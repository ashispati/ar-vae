import music21
from torch import from_numpy
import numpy as np
from music21 import note
from fractions import Fraction


MAX_NOTES = 1000
SLUR_SYMBOL = '__'
START_SYMBOL = 'START'
END_SYMBOL = 'END'
TICK_VALUES = [
        0,
        Fraction(1, 4),
        Fraction(1, 3),
        Fraction(1, 2),
        Fraction(2, 3),
        Fraction(3, 4)
    ]

RHY_COMPLEXITY_COEFFS = from_numpy(
    np.array(
        [
            0.20, 1, 2, 0.5, 2, 1,
            0.67, 1, 2, 0.5, 2, 1,
            0.25, 1, 2, 0.5, 2, 1,
            0.67, 1, 2, 0.5, 2, 1
        ]
    )
)


def is_valid(chorale):
    """Checks if a chorale has 4-parts or not"""
    # We only consider 4-part chorales
    if not len(chorale.parts) == 4:
        return False
    return True


def compute_tick_durations():
    """
    Computes the tick durations
    """
    diff = [n - p
            for n, p in zip(TICK_VALUES[1:], TICK_VALUES[:-1])]
    diff = diff + [1 - TICK_VALUES[-1]]
    return diff


def fix_pick_up_measure_offset(score):
    """
    Adds rests to the pick-up measure (if-any)

    :param score: music21 score object
    """
    measures = score.recurse().getElementsByClass(music21.stream.Measure)
    num_measures = len(measures)
    # add rests in pick-up measures
    if num_measures > 0:
        m0_dur = measures[0].barDurationProportion()
        m1_dur = measures[1].barDurationProportion()
        if m0_dur != 1.0:
            if m0_dur + m1_dur != 1.0:
                offset = measures[0].paddingLeft
                measures[0].insertAndShift(0.0, music21.note.Rest(quarterLength=offset))
                for i, m in enumerate(measures):
                    # shift the offset of all other measures
                    if i != 0:
                        m.offset += offset
    return score


def fix_last_measure(score):
    """
    Adds rests to the last measure (if-needed)

    :param score: music21 score object
    """
    measures = score.recurse().getElementsByClass(music21.stream.Measure)
    num_measures = len(measures)
    # add rests in pick-up measures
    if num_measures > 0:
        m0_dur = measures[num_measures - 1].barDurationProportion()
        if m0_dur != 1.0:
            offset = measures[num_measures - 1].paddingRight
            measures[num_measures - 1].append(music21.note.Rest(quarterLength=offset))
    return score


def get_notes(score):
    """
    Returns the notes from the score object

    :return: list, of music21 note objects
    """
    notes = score.parts[0].flat.notesAndRests
    notes = [n for n in notes if not isinstance(n, music21.harmony.ChordSymbol)]
    return notes


def is_score_on_ticks(score, tick_values):
    """
    Checks if the notes are on ticks

    :param score: music21 score object
    :param tick_values:
    :return: boolean, True or False
    """
    notes = get_notes(score)
    eps = 1e-5
    for n in notes:
        _, d = divmod(n.offset, 1)
        flag = False
        for tick_value in tick_values:
            if tick_value - eps < d < tick_value + eps:
                flag = True
        if not flag:
            return False
    return True


def score_range(score):
    """

    :param score: music21 score object
    :return: tuple int, min and max midi pitch numbers
    """
    notes = get_notes(score)
    pitches = [n.pitch.midi for n in notes if n.isNote]
    min_pitch = min(pitches)
    max_pitch = max(pitches)
    return min_pitch, max_pitch


def standard_name(note_or_rest):
    """
    Convert music21 note objects to str

    :param note_or_rest: music21 note object
    :return: str,
    """
    if isinstance(note_or_rest, music21.note.Note):
        return note_or_rest.nameWithOctave
    if isinstance(note_or_rest, music21.note.Rest):
        return note_or_rest.name


def standard_note(note_or_rest_string):
    if note_or_rest_string == 'rest':
        return note.Rest()
    elif note_or_rest_string == SLUR_SYMBOL:
        return note.Rest()
    elif note_or_rest_string == START_SYMBOL:
        return note.Rest()
    elif note_or_rest_string == END_SYMBOL:
        return note.Rest()
    else:
        return note.Note(note_or_rest_string)


def fix_and_expand_score(score):
    try:
        score = score.expandRepeats()
        score = fix_pick_up_measure_offset(score)
        score = fix_last_measure(score)
    except music21.repeat.ExpanderException:
        score = None
    return score


def get_music21_score_from_path(filepath, fix_and_expand=False):
    """
    Extracts the music21 score from the provided filepath

    :param filepath: full path to the .xml file
    :param fix_and_expand: bool, True to expand repeats and add rests for pick-up bars
    :return: music21 score object
    """
    # TODO: add condition for .abc and .xml formats
    score = music21.converter.parseFile(filepath, format='abc')
    if fix_and_expand:
        score = fix_and_expand_score(score)
    return score


def get_title(filepath):
    """

    :param filepath: full path to the .abc tune
    :return: str, title of the .abc tune
    """
    for line in open(filepath):
        if line[:2] == 'T:':
            return line[2:]
    return None


def tune_contains_chords(tune_filepath):
    """

    :param tune_filepath: full path to the abc tune
    :return: bool, True if tune contains chords
    """
    # TODO: write it correctly
    for line in open(tune_filepath):
        if '"' in line:
            return True
    return False


def tune_is_multivoice(tune_filepath):
    """

    :param tune_filepath: full path to the abc tune
    :return: bool, True if tune has mutiple voices
    """
    for line in open(tune_filepath):
        if line[:3] == 'V:2':
            return True
        if line[:4] == 'V: 2':
            return True
        if line[:4] == 'V :2':
            return True
        if line[:5] == 'V : 2':
            return True
    return False
