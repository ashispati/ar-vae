import os
from tqdm import tqdm
import random
from random import shuffle
from scipy import stats
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from music21 import meter
from music21.abcFormat import ABCHandlerException

from data.dataloaders.bar_dataset_helpers import *
from utils.helpers import to_cuda_variable, to_cuda_variable_long, to_numpy

# set random seed
random.seed(0)


class BarDataset:
    def __init__(self, time_sig_num=4, time_sig_den=4, dataset_type='train', is_short=False):
        # specify pitch range
        self.pitch_range = [55, 84]
        # specify dataset type
        self.dataset_type = dataset_type
        # specify dataset lenght
        self.is_short = is_short
        # specify time sigature
        self.time_sig_num = time_sig_num
        self.time_sig_den = time_sig_den
        self.time_sig_str = str(self.time_sig_num) + 'by' + str(self.time_sig_den)
        # instantiate tick durations
        self.beat_subdivisions = len(TICK_VALUES)
        self.tick_durations = compute_tick_durations()
        # specify class name, this should be changed for child classes
        self.class_name = None
        self.cur_dir = os.path.dirname(__file__)
        # initialize index dicts
        self.note2index_dicts = None
        self.index2note_dicts = None

    def __repr__(self):
        return self.class_name

    def compute_dicts(self):
        raise NotImplementedError

    def update_index_dicts(self):
        """
        Update the note dictionaries
        """
        f = open(self.dict_path, 'w')
        f.write("%s\n" % self.index2note_dicts)
        f.write("%s\n" % self.note2index_dicts)
        f.close()

    def get_dataset(self):
        """
        Returns the dataset
        """
        return self.make_dataset()

    def make_dataset(self):
        raise NotImplementedError

    def get_tensor(self, score):
        """
        Returns the melody as a tensor

        :param score: music21 score object
        """
        eps = 1e-5
        notes = get_notes(score)
        if not is_score_on_ticks(score, TICK_VALUES):
            return None
        list_note_strings_and_pitches = [(n.nameWithOctave, n.pitch.midi)
                                         for n in notes
                                         if n.isNote]
        note2index = self.note2index_dicts
        index2note = self.index2note_dicts
        for note_name, pitch in list_note_strings_and_pitches:
            if note_name not in note2index:
                new_index = len(note2index)
                index2note.update({new_index: note_name})
                note2index.update({note_name: new_index})
                print('Warning: Entry ' + str(
                    {new_index: note_name}) + ' added to dictionaries')
                self.update_index_dicts()

        # construct sequence
        j = 0
        i = 0
        length = int(score.highestTime * self.beat_subdivisions)
        t = np.zeros((length, 2))
        is_articulated = True
        num_notes = len(notes)
        current_tick = 0
        while i < length:
            if j < num_notes - 1:
                if notes[j + 1].offset > current_tick + eps:
                    t[i, :] = [note2index[standard_name(notes[j])],
                               is_articulated]
                    i += 1
                    current_tick += self.tick_durations[
                        (i - 1) % len(TICK_VALUES)]
                    is_articulated = False
                else:
                    j += 1
                    is_articulated = True
            else:
                t[i, :] = [note2index[standard_name(notes[j])],
                           is_articulated]
                i += 1
                is_articulated = False
        lead = t[:, 0] * t[:, 1] + (1 - t[:, 1]) * note2index[SLUR_SYMBOL]
        # convert to torch tensor
        lead_tensor = torch.from_numpy(lead).long()[None, :]
        return lead_tensor  # , chord_tensor

    def split_tensor_to_bars(self, score_tensor):
        """
        Splits the score tensor to individual bars

        :param score_tensor: torch tensor (1, length)
        :return torch tensor, (num_bars, bar_seq_length)
        """
        batch_size, seq_length = score_tensor.size()
        assert(batch_size == 1)
        bar_seq_length = int(self.beat_subdivisions * self.time_sig_num)
        num_bars = int(np.floor(seq_length / bar_seq_length))
        # truncate sequence if needed
        score_tensor = score_tensor[:, :num_bars * bar_seq_length]
        bar_tensor_score = score_tensor.view(num_bars, bar_seq_length)
        return bar_tensor_score

    def empty_score_tensor(self, score_length):
        """
        Creates an empty score tensor

        :param score_length: int, length of the score in ticks
        :return: torch long tensor, initialized with start indices
        """
        start_symbols = self.note2index_dicts[SLUR_SYMBOL]
        start_symbols = torch.from_numpy(start_symbols).long().clone()
        start_symbols = start_symbols.repeat(score_length, 1).transpose(0, 1)
        return start_symbols

    def get_score_from_tensor(self, tensor_score):
        """
        Converts the given score tensor to a music21 score object
        :param tensor_score: torch tensor, (1, num_ticks)
        :return: music21 score object
        """
        slur_index = self.note2index_dicts[SLUR_SYMBOL]

        score = music21.stream.Score()
        part = music21.stream.Part()
        # LEAD
        dur = 0
        f = music21.note.Rest()
        tensor_lead_np = tensor_score.numpy().flatten()
        for tick_index, note_index in enumerate(tensor_lead_np):
            # if it is a played note
            if not note_index == slur_index:
                # add previous note
                if dur > 0:
                    f.duration = music21.duration.Duration(dur)
                    part.append(f)

                dur = self.tick_durations[tick_index % self.beat_subdivisions]
                f = standard_note(self.index2note_dicts[note_index])
            else:
                dur += self.tick_durations[tick_index % self.beat_subdivisions]
        # add last note
        f.duration = music21.duration.Duration(dur)
        part.append(f)
        score.insert(part)
        return score

    def data_loaders(self, batch_size, split=(0.85, 0.10)):
        """
        Returns three data loaders obtained by splitting
        self.dataset according to split
        :param batch_size:
        :param split:
        :return:
        """
        dataset = self.get_dataset()
        assert sum(split) < 1
        num_examples = len(dataset)
        a, b = split
        train_dataset = TensorDataset(*dataset[: int(a * num_examples)])
        val_dataset = TensorDataset(*dataset[int(a * num_examples):
                                             int((a + b) * num_examples)])
        eval_dataset = TensorDataset(*dataset[int((a + b) * num_examples):])

        train_dl = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        val_dl = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )

        eval_dl = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        return train_dl, val_dl, eval_dl

    def tensor_to_m21score(self, tensor_score):
        """
        Converts lead given as tensor_lead to a true music21 score
        :param tensor_score:
        :return:
        """
        slur_index = self.note2index_dicts[SLUR_SYMBOL]

        score = music21.stream.Score()
        part = music21.stream.Part()
        # LEAD
        dur = 0
        f = music21.note.Rest()
        tensor_lead_np = tensor_score.numpy().flatten()
        for tick_index, note_index in enumerate(tensor_lead_np):
            # if it is a played note
            if not note_index == slur_index:
                # add previous note
                if dur > 0:
                    f.duration = music21.duration.Duration(dur)
                    part.append(f)

                dur = self.tick_durations[tick_index % self.beat_subdivisions]
                f = standard_note(self.index2note_dicts[note_index])
            else:
                dur += self.tick_durations[tick_index % self.beat_subdivisions]
        # add last note
        f.duration = music21.duration.Duration(dur)
        part.append(f)
        score.insert(part)
        return score

    def concatenate_scores(self, scores_list):
        score = music21.stream.Score()
        part = music21.stream.Part()
        dur = 0.0
        for s in scores_list:
            notes = get_notes(s)
            note_dur = 0
            for n in notes:
                part.insert(dur + note_dur, n)
                note_dur += n.duration.quarterLength
            dur += 4.0
        score.insert(part)
        return score

    def all_transposition_intervals(self, score):
        """
        Finds all the possible transposition intervals from the score
        :param score: music21 score object
        :return: list of music21 interval objects
        """
        min_pitch, max_pitch = score_range(score)
        min_pitch_corpus, max_pitch_corpus = self.pitch_range

        min_transposition = min_pitch_corpus - min_pitch
        max_transposition = max_pitch_corpus - max_pitch

        transpositions = []
        for semi_tone in range(min_transposition, max_transposition + 1):
            interval_type, interval_nature = music21.interval.convertSemitoneToSpecifierGeneric(
                semi_tone)
            transposition_interval = music21.interval.Interval(
                str(interval_nature) + interval_type)
            transpositions.append(transposition_interval)
        return transpositions

    def get_transposed_tensor(self, score, trans_int):
        """
        Returns the transposed tensor

        :param score: music21 score object
        :param trans_int:  music21.interval.Interval object
        :return:
        """
        score_transposed = score.transpose(trans_int)
        return self.get_tensor(score_transposed)

    def get_tensor_with_padding(self, tensor, start_tick, end_tick):
        """

        :param tensor: (batch_size, length)
        :param start_tick:
        :param end_tick:
        :return: (batch_size, end_tick - start_tick)
        """
        assert start_tick < end_tick
        # assert end_tick > 0
        batch_size, length = tensor.size()
        symbol2index = self.note2index_dicts
        padded_tensor = []
        if start_tick < 0:
            start_symbols = np.array([symbol2index[START_SYMBOL]])
            start_symbols = torch.from_numpy(start_symbols).long().clone()
            start_symbols = start_symbols.repeat(batch_size, -start_tick)
            # start_symbols[-1] = symbol2index[START_SYMBOL]
            padded_tensor.append(start_symbols)

        slice_start = start_tick if start_tick > 0 else 0
        slice_end = end_tick if end_tick < length else length

        padded_tensor.append(tensor[:, slice_start: slice_end])

        if end_tick > length:
            end_symbols = np.array([symbol2index[END_SYMBOL]])
            end_symbols = torch.from_numpy(end_symbols).long().clone()
            end_symbols = end_symbols.repeat(batch_size, end_tick - length)
            # end_symbols[0] = symbol2index[END_SYMBOL]
            padded_tensor.append(end_symbols)

        padded_tensor = torch.cat(padded_tensor, 1)
        return padded_tensor

    # Measure attribute extractors
    def get_note_density_in_measure(self, measure_tensor):
        """
        Returns the number of notes in each measure of the input normalized by the
        length of the length of the measure representation
        :param measure_tensor: torch Variable,
                (batch_size, measure_seq_len)
        :return: torch Variable containing float tensor ,
                (batch_size)
        """
        _, measure_seq_len = measure_tensor.size()
        slur_index = self.note2index_dicts[SLUR_SYMBOL]
        start_index = self.note2index_dicts[START_SYMBOL]
        end_index = self.note2index_dicts[END_SYMBOL]
        rest_index = self.note2index_dicts['rest']
        slur_count = torch.sum(measure_tensor == slur_index, 1)
        rest_count = torch.sum(measure_tensor == rest_index, 1)
        start_count = torch.sum(measure_tensor == start_index, 1)
        end_count = torch.sum(measure_tensor == end_index, 1)
        note_count = measure_seq_len - (slur_count + rest_count + start_count + end_count)
        note_density = to_cuda_variable(note_count.float() / measure_seq_len)
        return note_density

    def get_pitch_range_in_measure(self, measure_tensor):
        """
        Returns the note range of each measure of the input normalized by the range
        the dataset
        :param measure_tensor: torch Variable,
                (batch_size, measure_seq_len)
        :return: torch Variable containing float tensor ,
                (batch_size)
        """
        batch_size, measure_seq_len = measure_tensor.size()
        index2note = self.index2note_dicts
        slur_index = self.note2index_dicts[SLUR_SYMBOL]
        rest_index = self.note2index_dicts['rest']
        none_index = self.note2index_dicts[None]
        start_index = self.note2index_dicts[START_SYMBOL]
        end_index = self.note2index_dicts[END_SYMBOL]
        nrange = to_cuda_variable(torch.zeros(batch_size))
        for i in range(batch_size):
            midi_notes = []
            for j in range(measure_seq_len):
                index = measure_tensor[i][j].item()
                if index not in (slur_index, rest_index, none_index, start_index, end_index):
                    midi_note = torch.Tensor([music21.pitch.Pitch(index2note[index]).midi])
                    midi_notes.append(midi_note)
            if len(midi_notes) == 0 or len(midi_notes) == 1:
                nrange[i] = 0
            else:
                midi_notes = torch.cat(midi_notes, 0)
                midi_notes = to_cuda_variable_long(midi_notes)
                nrange[i] = torch.max(midi_notes) - torch.min(midi_notes)
        return nrange / 26

    def get_rhythmic_entropy(self, measure_tensor):
        """
        Returns the rhytmic entropy in a measure of music
        :param measure_tensor: torch Variable,
                (batch_size, measure_seq_len)
        :return: torch Variable,
                (batch_size)
        """
        slur_index = self.note2index_dicts[SLUR_SYMBOL]
        start_index = self.note2index_dicts[START_SYMBOL]
        end_index = self.note2index_dicts[END_SYMBOL]
        rest_index = self.note2index_dicts['rest']
        none_index = self.note2index_dicts[None]
        beat_tensor = measure_tensor.clone()
        beat_tensor[beat_tensor == start_index] = slur_index
        beat_tensor[beat_tensor == end_index] = slur_index
        beat_tensor[beat_tensor == rest_index] = slur_index
        beat_tensor[beat_tensor == none_index] = slur_index
        beat_tensor[beat_tensor != slur_index] = 1
        beat_tensor[beat_tensor == slur_index] = 0
        ent = stats.entropy(np.transpose(to_numpy(beat_tensor)))
        ent = torch.from_numpy(np.transpose(ent))
        ent = to_cuda_variable(ent)
        return ent

    def get_beat_strength(self, measure_tensor):
        """
        Returns the normalized beat strength in a measure of music
        :param measure_tensor: torch Variable,
                (batch_size, measure_seq_len)
        :return: torch Variable,
                (batch_size)
        """
        if measure_tensor.is_cuda:
            measure_tensor_np = measure_tensor.cpu().numpy()
        else:
            measure_tensor_np = measure_tensor.numpy()
        slur_index = self.note2index_dicts[SLUR_SYMBOL]
        measure_tensor_np[measure_tensor_np != slur_index] = 1
        measure_tensor_np[measure_tensor_np == slur_index] = 0
        weights = np.array([1, 0.008, 0.008, 0.15, 0.008, 0.008])
        weights = np.tile(weights, 4)
        prod = weights * measure_tensor_np
        b_str = np.sum(prod, axis=1)
        if torch.cuda.is_available():
            b_str = torch.autograd.Variable(torch.from_numpy(b_str).cuda())
        else:
            b_str = torch.autograd.Variable(torch.from_numpy(b_str))
        return b_str

    def get_rhy_complexity(self, measure_tensor):
        """
        Returns the normalized rhythmic complexity of a batch of measures
        :param measure_tensor: torch Variable,
                (batch_size, measure_seq_len)
        :return: torch Variable,
                (batch_size)
        """
        slur_index = self.note2index_dicts[SLUR_SYMBOL]
        start_index = self.note2index_dicts[START_SYMBOL]
        end_index = self.note2index_dicts[END_SYMBOL]
        rest_index = self.note2index_dicts['rest']
        none_index = self.note2index_dicts[None]
        beat_tensor = torch.ones_like(measure_tensor)
        beat_tensor[measure_tensor == slur_index] = 0
        beat_tensor[measure_tensor == start_index] = 0
        beat_tensor[measure_tensor == end_index] = 0
        beat_tensor[measure_tensor == none_index] = 0
        beat_tensor[measure_tensor == rest_index] = 0
        beat_tensor = beat_tensor.float()
        num_measures = measure_tensor.shape[0]
        weights = to_cuda_variable(RHY_COMPLEXITY_COEFFS.view(1, -1).float())
        norm_coeff = torch.sum(weights, dim=1)
        weights = weights.repeat(num_measures, 1)
        h_product = weights * beat_tensor
        b_str = torch.sum(h_product, dim=1) / norm_coeff
        return b_str

    def get_contour(self, measure_tensor):
        """
        Returns the direction of the melodic contour of a batch of measures
        :param measure_tensor: torch Variable
                (batch_size, measure_seq_len)
        :return torch.Variable
                (batch_size)
        """
        batch_size, measure_seq_len = measure_tensor.size()
        index2note = self.index2note_dicts
        slur_index = self.note2index_dicts[SLUR_SYMBOL]
        rest_index = self.note2index_dicts['rest']
        none_index = self.note2index_dicts[None]
        start_index = self.note2index_dicts[START_SYMBOL]
        end_index = self.note2index_dicts[END_SYMBOL]
        contour = to_cuda_variable(torch.zeros(batch_size))
        for i in range(batch_size):
            midi_notes = []
            for j in range(measure_seq_len):
                index = measure_tensor[i][j].item()
                if index not in (slur_index, rest_index, none_index, start_index, end_index):
                    midi_note = torch.Tensor([music21.pitch.Pitch(index2note[index]).midi])
                    midi_notes.append(midi_note)
            if len(midi_notes) == 0 or len(midi_notes) == 1:
                contour[i] = 0
            else:
                midi_notes = torch.cat(midi_notes, 0)
                a = midi_notes[1:] - midi_notes[:-1]
                b = torch.sum(a)
                contour[i] = to_cuda_variable_long(b)
        return contour / 26

    def get_interval_entropy(self, measure_tensor):
        """
        Returns the normalized interval entropy of a batch of measures
        :param measure_tensor: torch Variable,
                (batch_size, measure_seq_len)
        :return: torch Variable,
                (batch_size)
        """
        batch_size, measure_seq_len = measure_tensor.size()
        index2note = self.index2note_dicts
        slur_index = self.note2index_dicts[SLUR_SYMBOL]
        rest_index = self.note2index_dicts['rest']
        none_index = self.note2index_dicts[None]
        start_index = self.note2index_dicts[START_SYMBOL]
        end_index = self.note2index_dicts[END_SYMBOL]
        has_note = False
        nrange = to_cuda_variable(torch.zeros(batch_size))
        for i in range(batch_size):
            midi_notes = []
            for j in range(measure_seq_len):
                index = measure_tensor[i][j].item()
                if index not in (slur_index, rest_index, none_index, start_index, end_index):
                    midi_note = torch.Tensor([music21.pitch.Pitch(index2note[index]).midi])
                    midi_notes.append(midi_note)
            if len(midi_notes) == 0 or len(midi_notes) == 1:
                nrange[i] = 0
            else:
                midi_notes = torch.cat(midi_notes, 0)
                midi_notes = to_cuda_variable_long(midi_notes)
                # compute intervals
                midi_notes = torch.abs(midi_notes[1:] - midi_notes[:-1])
                midi_notes = midi_notes % 12
                probs = to_cuda_variable(torch.zeros([12, 1]))
                for k in range(len(midi_notes)):
                    probs[midi_notes[k]] += 1
                # compute entropy of this interval vector
                b = nn.functional.softmax(probs, dim=0) * \
                    nn.functional.log_softmax(probs, dim=0)
                b = -1.0 * b.sum()
                nrange[i] = b
        return nrange


class ChoraleBarDataset(BarDataset):
    def __init__(self, time_sig_num=4, time_sig_den=4, dataset_type='train', is_short=False):
        """
        Init method for the ChraleBarDataset class

        :param time_sig_num: int, time signature numerator
        :param time_sig_den: int, time signature denominator
        :param dataset_type: str, 'train' or 'test'
        :param is_short: bool, smaller dataset if True
        """
        super().__init__(time_sig_num, time_sig_den, dataset_type, is_short)
        self.class_name = self.time_sig_str + '_ChoraleBarDataset_'
        if self.is_short:
            self.corpus_it_gen = music21.corpus.chorales.Iterator(highestNumber=10)
        else:
            self.corpus_it_gen = music21.corpus.chorales.Iterator()

        # compute and save index dicts
        self.dataset_dir_path = os.path.join(os.path.dirname(self.cur_dir), 'datasets')
        self.dict_path = os.path.join(self.dataset_dir_path, self.class_name + 'index_dicts.txt')
        self.index2note_dicts = None
        self.note2index_dicts = None
        self.compute_dicts()

        # create path to dataset
        self.dataset_path = os.path.join(self.dataset_dir_path, self.class_name + self.dataset_type)
        if is_short:
            self.dataset_path += '_short'

    def iterator_gen(self):
        return (
            chorale for chorale in self.corpus_it_gen if is_valid(chorale)
        )

    def compute_dicts(self):
        if os.path.exists(self.dict_path):
            print('Dictionaries already exist. Reading them now')
            f = open(self.dict_path, 'r')
            dicts = [line.rstrip('\n') for line in f]
            assert (len(dicts) == 2)  # must have 2 dictionaries
            self.index2note_dicts = eval(dicts[0])
            self.note2index_dicts = eval(dicts[1])
            return

        print('Computing note index dicts')
        self.index2note_dicts = {}
        self.note2index_dicts = {}
        # add slur symbol
        note_sets = set()
        note_sets.add(SLUR_SYMBOL)
        note_sets.add(START_SYMBOL)
        note_sets.add(END_SYMBOL)
        # get all notes
        for _, f in tqdm(enumerate(self.iterator_gen())):
            notes = get_notes(f)
            for n in notes:
                note_sets.add(standard_name(n))
        # update dictionary
        if 'rest' not in note_sets:
            note_sets.add('rest')
        for note_index, note_name in enumerate(note_sets):
            self.index2note_dicts.update({note_index: note_name})
            self.note2index_dicts.update({note_name: note_index})
        self.update_index_dicts()

    def make_dataset(self):
        """
        Creates the tensor dataset of musical bars

        :return: torch Dataset object
        """
        if os.path.exists(self.dataset_path):
            print('Dataset already created. Reading it now')
            return torch.load(self.dataset_path)

        if not os.path.exists(self.dict_path):
            self.compute_dicts()

        print('Making tensor dataset')
        bar_tensor_dataset = []
        for _, chorale in tqdm(enumerate(self.iterator_gen())):
            chorale = fix_and_expand_score(chorale)
            if chorale is not None:
                score_tensor = self.get_tensor(chorale)
                if score_tensor is not None:
                    local_tensor = self.split_tensor_to_bars(score_tensor)
                    bar_tensor_dataset.append(local_tensor.int())
        bar_tensor_dataset = torch.cat(bar_tensor_dataset, 0)
        dataset = TensorDataset(
            bar_tensor_dataset,
            bar_tensor_dataset  # TODO: add metadata tensor here
        )
        print('Dataset Size: ', bar_tensor_dataset.size())
        torch.save(dataset, self.dataset_path)
        return dataset


class ChoraleNBarDataset(ChoraleBarDataset):
    """
    Class to create n-bar sequences of chorale music
    """

    def __init__(self, time_sig_num=4, time_sig_den=4, dataset_type='train', is_short=False, num_bars=16):
        super().__init__(
            time_sig_num,
            time_sig_den,
            dataset_type,
            is_short
        )
        self.n_bars = num_bars
        self.class_name = self.time_sig_str + '_ChoraleNBarDataset_' + str(self.n_bars) + '_'
        if self.time_sig_num == 4:
            self.num_beats_per_bar = 4
        elif self.time_sig_num == 3:
            self.num_beats_per_bar = 3
        self.seq_size_in_beats = self.num_beats_per_bar * self.n_bars

        # create path to dataset
        self.dataset_path = os.path.join(self.dataset_dir_path, self.class_name + self.dataset_type)
        if is_short:
            self.dataset_path += '_short'

    def get_tensor_dataset(self, score, bar_tensor_dataset):
        """

        :param score: music21 score object
        :param bar_tensor_dataset: list
        :return: None
        """
        score = fix_and_expand_score(score)
        if score is None:
            return
        possible_transpositions = self.all_transposition_intervals(score)
        for trans_int in possible_transpositions:
            score_tensor = self.get_transposed_tensor(score, trans_int)
            if score_tensor is None:
                continue
            total_beats = int(score.highestTime)
            for offset_start in range(
                -self.num_beats_per_bar,
                total_beats,
                int(self.seq_size_in_beats)
            ):
                offset_end = offset_start + self.seq_size_in_beats
                local_score_tensor = self.get_tensor_with_padding(
                    tensor=score_tensor,
                    start_tick=offset_start * self.beat_subdivisions,
                    end_tick=offset_end * self.beat_subdivisions
                )
                # cast to int
                bar_tensor_dataset.append(
                    local_score_tensor.int()
                )

    def make_dataset(self):
        """

        :return: Tensor dataset
        """
        if os.path.exists(self.dataset_path):
            print('Dataset already created. Reading it now')
            return torch.load(self.dataset_path)

        if not os.path.exists(self.dict_path):
            self.compute_dicts()

        print('Making tensor dataset')
        bar_tensor_dataset = []
        # Parallel(n_jobs=NUM_CORES)(
        #    delayed(self.get_tensor_dataset)(f, bar_tensor_dataset) for _, f in tqdm(enumerate(self.dataset_filepaths))
        # )
        for _, chorale in tqdm(enumerate(self.iterator_gen())):
            self.get_tensor_dataset(chorale, bar_tensor_dataset)
        bar_tensor_dataset = torch.cat(bar_tensor_dataset, 0)
        num_datapoints = bar_tensor_dataset.size()[0]
        score_tensor_dataset = bar_tensor_dataset.view(
            num_datapoints, 1, -1
        )
        dataset = TensorDataset(
            bar_tensor_dataset,
            bar_tensor_dataset  # TODO: add metadata tensor here
        )
        print('Sizes: ', score_tensor_dataset.size())
        torch.save(dataset, self.dataset_path)
        return dataset


class FolkBarDataset(BarDataset):
    def __init__(self, time_sig_num=4, time_sig_den=4, dataset_type='train', is_short=False):
        """
        Init method for the FolkDarDataset class

        :param time_sig_num: int, time signature numerator
        :param time_sig_den: int, time signature denominator
        :param dataset_type: str, 'train' or 'test'
        :param is_short: bool, smaller dataset if True
        """
        super().__init__(time_sig_num, time_sig_den, dataset_type, is_short)
        self.class_name = self.time_sig_str + '_FolkBarDataset_'
        if self.is_short:
            self.max_num_files = 20
        else:
            self.max_num_files = 25000
        self.raw_datapath = os.path.join(os.path.dirname(self.cur_dir), 'folk_raw_data')
        self.dataset_dir_path = os.path.join(os.path.dirname(self.cur_dir), 'datasets')
        self.filelist = [
            os.path.join(self.raw_datapath, f) for f in os.listdir(self.raw_datapath)
            if os.path.join(self.raw_datapath, f).endswith('.abc')
        ]
        if len(self.filelist) == 0:
            print('No files in the raw data folder')
            return
        # specify time sigature
        self.time_sig_num = time_sig_num
        self.time_sig_den = time_sig_den
        self.time_sig_str = str(self.time_sig_num) + 'by' + str(self.time_sig_den)
        self.valid_filelist = self.time_sig_str + 'valid_filelist.txt'

        # compute and save valid file paths
        self.valid_filelist_path = os.path.join(self.dataset_dir_path, self.valid_filelist)
        if os.path.exists(self.valid_filelist_path):
            print('Valid file list already available. Reading them now.')
            f = open(self.valid_filelist_path, 'r')
            self.valid_filepaths = [
                os.path.join(self.raw_datapath, line.rstrip('\n')) for line in f
            ]
            f.close()
        else:
            self.valid_filepaths = self.get_valid_files()
            f = open(self.valid_filelist, 'w')
            for line in self.valid_filepaths:
                f.write("%s\n" % line)
            f.close()
        
        # compute and save dicts
        self.class_name = self.time_sig_str + '_FolkBarDataset_'
        self.dict_path = os.path.join(self.dataset_dir_path, self.class_name + 'index_dicts.txt')
        self.index2note_dicts = None
        self.note2index_dicts = None
        self.compute_dicts()

        # create training and testing splits
        shuffle(self.valid_filepaths)
        if is_short:
            self.valid_filepaths = self.valid_filepaths[:self.max_num_files]
        self.num_files = len(self.valid_filepaths)
        self.num_training = int(0.9 * self.num_files)
        self.num_testing = int(0.1 * self.num_files)
        if self.dataset_type == 'train':
            self.dataset_filepaths = self.valid_filepaths[:self.num_training]
        elif self.dataset_type == 'test':
            self.dataset_filepaths = self.valid_filepaths[self.num_training:]
        self.num_dataset_files = len(self.dataset_filepaths)

        # create path to dataset
        self.dataset_path = os.path.join(self.dataset_dir_path, self.class_name + self.dataset_type)
        if is_short:
            self.dataset_path += '_short'

    def make_dataset(self):
        """
        Creates the tensor dataset of musical bars

        :return: torch Dataset object
        """
        if os.path.exists(self.dataset_path):
            print('Dataset already created. Reading it now')
            return torch.load(self.dataset_path)

        if not os.path.exists(self.dict_path):
            self.compute_dicts()

        print('Making tensor dataset')
        bar_tensor_dataset = []
        for _, f in tqdm(enumerate(self.dataset_filepaths)):
            score = get_music21_score_from_path(f, fix_and_expand=True)
            score_tensor = self.get_tensor(score)
            local_tensor = self.split_tensor_to_bars(score_tensor)
            bar_tensor_dataset.append(local_tensor.int())
        bar_tensor_dataset = torch.cat(bar_tensor_dataset, 0)
        dataset = TensorDataset(
            bar_tensor_dataset,
            bar_tensor_dataset  # TODO: add metadata tensor here
        )
        print('Dataset Size: ', bar_tensor_dataset.size())
        torch.save(dataset, self.dataset_path)
        return dataset

    def compute_dicts(self):
        if os.path.exists(self.dict_path):
            print('Dictionaries already exist. Reading them now')
            f = open(self.dict_path, 'r')
            dicts = [line.rstrip('\n') for line in f]
            assert (len(dicts) == 2)  # must have 2 dictionaries
            self.index2note_dicts = eval(dicts[0])
            self.note2index_dicts = eval(dicts[1])
            return

        print('Computing note index dicts')
        self.index2note_dicts = {}
        self.note2index_dicts = {}
        # add slur symbol
        note_sets = set()
        note_sets.add(SLUR_SYMBOL)
        note_sets.add(START_SYMBOL)
        note_sets.add(END_SYMBOL)
        # get all notes
        for _, f in tqdm(enumerate(self.valid_filepaths)):
            score = get_music21_score_from_path(f)
            notes = get_notes(score)
            for n in notes:
                note_sets.add(standard_name(n))
        # update dictionary
        if 'rest' not in note_sets:
            note_sets.add('rest')
        for note_index, note_name in enumerate(note_sets):
            self.index2note_dicts.update({note_index: note_name})
            self.note2index_dicts.update({note_name: note_index})
        self.update_index_dicts()

    def get_valid_files(self):
        """
        Iterates through self.filelist and updates the valid filelist
        Checks for 4by4 scores 
        """
        valid_filelist = []
        for idx, f in tqdm(enumerate(self.filelist)):
            if idx >= self.max_num_files:
                break
            title = get_title(f)
            if title is None:
                continue
            if tune_is_multivoice(f):
                continue
            if tune_contains_chords(f):
                continue
            try:
                # read score from file
                score = get_music21_score_from_path(f)
                # ignore files with multiple time signatures
                time_sig = score.parts[0].recurse().getElementsByClass(meter.TimeSignature)
                if len(time_sig) > 1:
                    continue 
                # ignore files with non 4by4 time signatures
                time_sig_num = time_sig[0].numerator
                time_sig_den = time_sig[0].denominator
                if time_sig_den != self.time_sig_den and time_sig_num != self.time_sig_num:
                    continue 
                # ignore files with no  notes
                notes = get_notes(score)
                pitches = [n.pitch.midi for n in notes if n.isNote]
                if len(pitches) == 0:
                    continue
                # ignore files with too many notes
                if len(notes) > MAX_NOTES:
                    continue
                # ignore files with 32nd and 64th notes
                dur_list = [n.duration for n in notes if n.isNote]
                for dur in dur_list:
                    d = dur.type
                    if d == '32nd' or d == '64th':
                        break
                    elif d == 'complex':
                        if len(dur.components) > 2:
                            break
                # check if expand repeat works
                score = get_music21_score_from_path(f, fix_and_expand=True)
                # ignore files where notes are not on ticks
                if not is_score_on_ticks(score, TICK_VALUES):
                    continue
                valid_filelist.append(f)
            except (music21.abcFormat.ABCHandlerException,
                    music21.abcFormat.ABCTokenException,
                    music21.duration.DurationException,
                    music21.pitch.AccidentalException,
                    music21.meter.MeterException,
                    music21.repeat.ExpanderException,
                    music21.exceptions21.StreamException,
                    AttributeError,
                    IndexError,
                    UnboundLocalError,
                    ValueError,
                    ABCHandlerException):
                print('Error while parsing file', idx)

        return valid_filelist


class FolkNBarDataset(FolkBarDataset):
    """
    Class to create n-bar sequences of folk music
    """
    def __init__(self, time_sig_num=4, time_sig_den=4, dataset_type='train', is_short=False, num_bars=16):
        """
        Init method for the FolkDarDataset class

        :param time_sig_num: int, time signature numerator
        :param time_sig_den: int, time signature denominator
        :param dataset_type: str, 'train' or 'test'
        :param is_short: bool, smaller dataset if True
        :param num_bars: int, number of bars
        """
        super().__init__(
            time_sig_num,
            time_sig_den,
            dataset_type,
            is_short
        )
        self.n_bars = num_bars
        self.class_name = self.time_sig_str + '_FolkNBarDataset_' + str(self.n_bars) + '_'
        if self.time_sig_num == 4:
            self.num_beats_per_bar = 4
        elif self.time_sig_num == 3:
            self.num_beats_per_bar = 3
        self.seq_size_in_beats = self.num_beats_per_bar * self.n_bars

        # create path to dataset
        self.dataset_path = os.path.join(self.dataset_dir_path, self.class_name + self.dataset_type)
        if is_short:
            self.dataset_path += '_short'

    def get_tensor_dataset(self, f, bar_tensor_dataset):
        """

        :param f: str, filename
        :param bar_tensor_dataset: list
        :return: None
        """
        score = get_music21_score_from_path(f, fix_and_expand=True)
        possible_transpositions = self.all_transposition_intervals(score)
        for trans_int in possible_transpositions:
            score_tensor = self.get_transposed_tensor(score, trans_int)
            total_beats = int(score.highestTime)
            for offset_start in range(
                -self.num_beats_per_bar,
                total_beats,
                int(self.seq_size_in_beats)
            ):
                offset_end = offset_start + self.seq_size_in_beats
                local_score_tensor = self.get_tensor_with_padding(
                    tensor=score_tensor,
                    start_tick=offset_start * self.beat_subdivisions,
                    end_tick=offset_end * self.beat_subdivisions
                )
                # cast to int
                bar_tensor_dataset.append(
                    local_score_tensor.int()
                )

    def make_dataset(self):
        """

        :return: Tensor dataset
        """
        if os.path.exists(self.dataset_path):
            print('Dataset already created. Reading it now')
            return torch.load(self.dataset_path)

        if not os.path.exists(self.dict_path):
            self.compute_dicts()

        print('Making tensor dataset')
        bar_tensor_dataset = []
        # Parallel(n_jobs=NUM_CORES)(
        #    delayed(self.get_tensor_dataset)(f, bar_tensor_dataset) for _, f in tqdm(enumerate(self.dataset_filepaths))
        # )
        for _, f in tqdm(enumerate(self.dataset_filepaths)):
            self.get_tensor_dataset(f, bar_tensor_dataset)
        bar_tensor_dataset = torch.cat(bar_tensor_dataset, 0)
        num_datapoints = bar_tensor_dataset.size()[0]
        score_tensor_dataset = bar_tensor_dataset.view(
            num_datapoints, 1, -1
        )
        dataset = TensorDataset(
            bar_tensor_dataset,
            bar_tensor_dataset  # TODO: add metadata tensor here
        )
        print('Sizes: ', score_tensor_dataset.size())
        torch.save(dataset, self.dataset_path)
        return dataset


if __name__ == '__main__':
    # usage example
    is_short = False
    if is_short:
        batch_size = 10
    else:
        batch_size = 128
    bar_dataset = FolkNBarDataset(dataset_type='train', is_short=is_short, num_bars=1)
    (train_dataloader,
     val_dataloader,
     test_dataloader) = bar_dataset.data_loaders(
        batch_size=batch_size,
        split=(0.7, 0.2)
    )
    print('Num Train Batches: ', len(train_dataloader))
    print('Num Valid Batches: ', len(val_dataloader))
    print('Num Test Batches: ', len(test_dataloader))

