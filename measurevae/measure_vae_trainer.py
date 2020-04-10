import os
import json
import torch
from tqdm import tqdm
import music21
from typing import Tuple

from utils.trainer import Trainer
from measurevae.measure_vae import MeasureVAE
from data.dataloaders.bar_dataset import FolkNBarDataset
from utils.helpers import to_cuda_variable_long, to_cuda_variable, to_numpy
from utils.plotting import plot_dim, plot_pianoroll_from_midi
from utils.evaluation import *

MUSIC_REG_TYPE = {
    'rhy_complexity': 0,
    'pitch_range': 1,
    'note_density': 2,
    'contour': 3
}


class MeasureVAETrainer(Trainer):
    def __init__(
            self,
            dataset,
            model: MeasureVAE,
            lr=1e-4,
            reg_type: Tuple[str] = None,
            reg_dim: Tuple[int] = 0,
            beta=0.001,
            gamma=1.0,
            capacity=0.0,
            rand=0,
            delta=10.0,
    ):
        super(MeasureVAETrainer, self).__init__(dataset, model, lr)
        if dataset.class_name[5:9] == 'Chor':
            self.dataset_type = 'bach'
        elif dataset.class_name[5:9] == 'Folk':
            self.dataset_type = 'folk'
        else:
            raise ValueError("Dataset Type not recognized")
        self.attr_dict = MUSIC_REG_TYPE
        self.reverse_attr_dict = {
            v: k for k, v in self.attr_dict.items()
        }
        self.metrics = {}
        self.beta = beta
        self.capacity = to_cuda_variable(torch.FloatTensor([capacity]))
        self.gamma = 0.0
        self.delta = 0.0
        self.cur_epoch_num = 0
        self.warm_up_epochs = 10
        self.reg_type = reg_type
        self.reg_dim = ()
        self.use_reg_loss = False
        self.rand_seed = rand
        torch.manual_seed(self.rand_seed)
        np.random.seed(self.rand_seed)
        self.trainer_config = f'_r_{self.rand_seed}_b_{self.beta}_'
        if capacity != 0.0:
            self.trainer_config += f'c_{capacity}_'
        self.model.update_trainer_config(self.trainer_config)
        if len(self.reg_type) != 0:
            self.use_reg_loss = True
            self.reg_dim = reg_dim
            self.gamma = gamma
            self.delta = delta
            self.trainer_config += f'g_{self.gamma}_d_{self.delta}_'
            reg_type_str = '_'.join(self.reg_type)
            self.trainer_config += f'{reg_type_str}_'
            self.model.update_trainer_config(self.trainer_config)

    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: tuple of Torch Variable objects
        """
        score_tensor, metadata_tensor = batch
        if isinstance(self.dataset, FolkNBarDataset):
            batch_size = score_tensor.size(0)
            score_tensor = score_tensor.view(batch_size, self.dataset.n_bars, -1)
            score_tensor = score_tensor.view(batch_size * self.dataset.n_bars, -1)
            metadata_tensor = metadata_tensor.view(batch_size, self.dataset.n_bars, -1)
            metadata_tensor = metadata_tensor.view(batch_size * self.dataset.n_bars, -1)
        # convert input to torch Variables
        batch_data = (
            to_cuda_variable_long(score_tensor),
            to_cuda_variable_long(metadata_tensor)
        )
        return batch_data

    def loss_and_acc_for_batch(self, batch, epoch_num=None, batch_num=None, train=True):
        """
        Computes the loss and accuracy for the batch
        Must return (loss, accuracy) as a tuple, accuracy can be None
        :param batch: torch Variable,
        :param epoch_num: int, used to change training schedule
        :param batch_num: int
        :param train: bool, True is backward pass is to be performed
        :return: scalar loss value, scalar accuracy value
        """
        if self.cur_epoch_num != epoch_num:
            flag = True
            self.cur_epoch_num = epoch_num
        else:
            flag = False

        # extract data
        score, metadata = batch

        # perform forward pass of model
        weights, samples, z_dist, prior_dist, z_tilde, z_prior = self.model(
            measure_score_tensor=score,
            measure_metadata_tensor=metadata,
            train=train
        )

        # compute reconstruction loss
        recons_loss = self.reconstruction_loss(x=score, x_recons=weights)

        # compute KLD loss
        dist_loss = self.compute_kld_loss(z_dist, prior_dist, self.beta)

        # add loses
        loss = recons_loss + dist_loss

        # compute and add regularization loss if needed
        if self.use_reg_loss:
            reg_loss = 0.0
            attr_labels = self.compute_attribute_labels(score)
            if type(self.reg_dim) == tuple:
                for dim in self.reg_dim:
                    labels = attr_labels[:, dim]
                    reg_loss += self.compute_reg_loss(
                        z_tilde, labels, dim, gamma=self.gamma, factor=self.delta
                    )
            else:
                raise TypeError("Regularization dimension must be a tuple of integers")
            loss += reg_loss
            if flag:
                self.writer.add_scalar('loss_split/recons_loss', recons_loss.item(), epoch_num)
                self.writer.add_scalar(
                    'loss_split/dist_loss', (dist_loss / self.beta).item(), epoch_num
                )
                self.writer.add_scalar(
                    'loss_split/reg_loss', (reg_loss / self.gamma).item(), epoch_num
                )
        else:
            if flag:
                self.writer.add_scalar(
                    'loss_split/recons_loss', recons_loss.item(), epoch_num
                )
                self.writer.add_scalar(
                    'loss_split/dist_loss', (dist_loss / self.beta).item(), epoch_num
                )

        # compute accuracy
        accuracy = self.mean_accuracy(
            weights=weights, targets=score
        )

        return loss, accuracy

    def compute_attribute_labels(self, score, attr_list=None):
        """
        Computes the attribute values given the regularization dimension
        """
        attr_tensor = []
        if attr_list is None:
            attr_list = [k for k in self.attr_dict.keys()]
        for i, attr_name in enumerate(attr_list):
            if attr_name == 'rhy_complexity':
                attr_tensor.append(self.dataset.get_rhy_complexity(score).view(1, -1))
            elif attr_name == 'pitch_range':
                attr_tensor.append(self.dataset.get_pitch_range_in_measure(score).view(1, -1))
            elif attr_name == 'note_density':
                attr_tensor.append(self.dataset.get_note_density_in_measure(score).view(1, -1))
            elif attr_name == 'contour':
                attr_tensor.append(self.dataset.get_contour(score).view(1, -1))
            else:
                raise ValueError('Invalid regularization attribute')
        attr_tensor = torch.cat(attr_tensor, 0)
        return attr_tensor.transpose(1, 0)

    def compute_representations(self, data_loader, num_batches=None):
        latent_codes = []
        attributes = []
        if num_batches is None:
            num_batches = 200
        for batch_id, batch in tqdm(enumerate(data_loader)):
            inputs, metadata = self.process_batch_data(batch)
            _, _, _, _, z_tilde, _ = self.model(inputs, metadata, train=False)
            latent_codes.append(to_numpy(z_tilde.cpu()))
            labels = self.compute_attribute_labels(inputs)
            attributes.append(to_numpy(labels))
            if batch_id == num_batches:
                break
        latent_codes = np.concatenate(latent_codes, 0)
        attributes = np.concatenate(attributes, 0)
        attr_list = [
            attr for attr in self.attr_dict.keys()
        ]
        return latent_codes, attributes, attr_list

    def eval_model(self, data_loader, epoch_num=0):
        if self.writer is not None:
            # evaluation takes time due to computation of metrics
            # so we skip it during training epochs
            metrics = None
        else:
            metrics = self.compute_eval_metrics()
        return metrics

    def compute_eval_metrics(self):
        """Returns the saved results as dict or computes them"""
        results_fp = os.path.join(
            os.path.dirname(self.model.filepath),
            'results_dict.json'
        )
        if os.path.exists(results_fp):
            with open(results_fp, 'r') as infile:
                self.metrics = json.load(infile)
        else:
            batch_size = 256
            _, _, data_loader = self.dataset.data_loaders(batch_size=batch_size)
            latent_codes, attributes, attr_list = self.compute_representations(data_loader)
            interp_metrics = compute_interpretability_metric(
                latent_codes, attributes, attr_list
            )
            self.metrics = {
                "interpretability": interp_metrics
            }
            self.metrics.update(compute_correlation_score(latent_codes, attributes))
            self.metrics.update(compute_modularity(latent_codes, attributes))
            self.metrics.update(compute_mig(latent_codes, attributes))
            self.metrics.update(compute_sap_score(latent_codes, attributes))
            self.metrics.update(self.test_model(batch_size=batch_size))
            with open(results_fp, 'w') as outfile:
                json.dump(self.metrics, outfile, indent=2)
        return self.metrics

    def plot_data_dist(self, latent_codes, attributes, attr_str, dim1=0, dim2=1):
        save_filename = os.path.join(
            Trainer.get_save_dir(self.model),
            'data_dist_' + attr_str + '.png'
        )
        img = plot_dim(
            latent_codes, attributes[:, self.attr_dict[attr_str]], save_filename, dim1=dim1, dim2=dim2,
            xlim=4.0, ylim=4.0
        )
        return img

    def plot_latent_interpolations(self, latent_codes, attr_str, num_points=10):
        n = min(num_points, latent_codes.shape[0])
        interp_dict = self.compute_eval_metrics()["interpretability"]
        dim = interp_dict[attr_str][0]
        for i in range(n):
            # compute output as music21 score
            original_score, _ = self.decode_latent_codes(torch.from_numpy(latent_codes[i:i+1, :]))
            org_save_filepath = os.path.join(
                Trainer.get_save_dir(self.model),
                f'original_{i}.mid'
            )
            original_score.write('midi', fp=org_save_filepath)
            score, tensor_score = self.compute_latent_interpolations(latent_codes[i:i+1, :], original_score, dim)

            attr_labels = self.compute_attribute_labels(tensor_score, [attr_str]).cpu().numpy().flatten()

            # write MIDI file
            save_filepath = os.path.join(
                Trainer.get_save_dir(self.model),
                f'latent_interpolations_{attr_str}_{i}.mid'
            )
            score.write('midi', fp=save_filepath)
            # plot MIDI
            plot_pianoroll_from_midi(save_filepath, attr_labels, attr_str, type=self.dataset_type)

    def decode_latent_codes(self, latent_codes):
        batch_size = latent_codes.size(0)
        dummy_score_tensor = to_cuda_variable(
            torch.zeros(batch_size, self.dataset.beat_subdivisions * self.dataset.seq_size_in_beats)
        )
        _, tensor_score = self.model.decoder(latent_codes, dummy_score_tensor, False)
        score = self.dataset.tensor_to_m21score(tensor_score)
        return score, tensor_score

    def compute_latent_interpolations(self, latent_code, original_score, dim1=0, num_points=5):
        assert num_points % 2 == 1
        x1 = torch.linspace(-4.0, 4.0, num_points)
        num_points = x1.size(0)
        z = to_cuda_variable(torch.from_numpy(latent_code))
        z = z.repeat(num_points, 1)
        z[:, dim1] = x1.contiguous()
        num_measures = z.size(0)
        score_list = []
        tensor_score_list = []
        for n in range(num_measures):
            score, tensor_score = self.decode_latent_codes(z[n:n+1, :])
            score_list.append(score)
            tensor_score_list.append(tensor_score)
        score_list[num_points // 2] = original_score
        concatenated_score = self.dataset.concatenate_scores(score_list)
        concatenated_tensor_score = torch.cat(tensor_score_list)
        concatenated_tensor_score = torch.squeeze(concatenated_tensor_score, dim=1)
        return concatenated_score, concatenated_tensor_score

    def plot_latent_surface(self, attr_str, dim1=0, dim2=1, grid_res=0.05):
        """
        Plots the value of an attribute over a surface defined by the dimensions
        :param dim1: int,
        :param dim2: int,
        :param grid_res: float,
        :return:
        """
        # create the dataspace
        x1 = torch.arange(-5., 5., grid_res)
        x2 = torch.arange(-5., 5., grid_res)
        z1, z2 = torch.meshgrid([x1, x2])
        num_points = z1.size(0) * z1.size(1)
        z = torch.randn(1, self.model.latent_space_dim)
        z = z.repeat(num_points, 1)
        z[:, dim1] = z1.contiguous().view(1, -1)
        z[:, dim2] = z2.contiguous().view(1, -1)
        z = to_cuda_variable(z)

        mini_batch_size = 500
        num_mini_batches = num_points // mini_batch_size
        attr_labels_all = []
        for i in tqdm(range(num_mini_batches)):
            z_batch = z[i * mini_batch_size:(i+1) * mini_batch_size, :]
            dummy_score_tensor = to_cuda_variable(
                torch.zeros(z_batch.size(0), self.dataset.beat_subdivisions * self.dataset.seq_size_in_beats)
            )
            _, samples = self.model.decoder(
                z=z_batch,
                score_tensor=dummy_score_tensor,
                train=False
            )
            samples = samples.view(z_batch.size(0), -1)
            labels = self.compute_attribute_labels(samples, attr_list=[attr_str])
            attr_labels_all.append(labels)

        attr_labels_all = to_numpy(torch.cat(attr_labels_all, 0))
        z = to_numpy(z)[:num_mini_batches*mini_batch_size, :]
        save_filename = os.path.join(
            Trainer.get_save_dir(self.model),
            f'latent_surface_{attr_str}.png'
        )
        plot_dim(z, attr_labels_all, save_filename, dim1=dim1, dim2=dim2)

    def test_model(self, batch_size):
        _, _, gen_test = self.dataset.data_loaders(batch_size)
        mean_loss_test, mean_accuracy_test = self.loss_and_acc_test(gen_test)
        print('Test Epoch:')
        print(
            '\tTest Loss: ', mean_loss_test, '\n'
            '\tTest Accuracy: ', mean_accuracy_test * 100
        )
        return {
            "test_loss": mean_loss_test,
            "test_acc": mean_accuracy_test,
        }

    def loss_and_acc_test(self, data_loader):
        mean_loss = 0
        mean_accuracy = 0

        for sample_id, batch in tqdm(enumerate(data_loader)):
            inputs, metadata_tensor = self.process_batch_data(batch)
            inputs = to_cuda_variable(inputs)
            # compute forward pass
            outputs, _, _, _, _, _ = self.model(
                measure_score_tensor=inputs,
                measure_metadata_tensor=metadata_tensor,
                train=False
            )
            # compute loss
            recons_loss = self.reconstruction_loss(
                x=inputs, x_recons=outputs
            )
            loss = recons_loss
            # compute mean loss and accuracy
            mean_loss += to_numpy(loss.mean())
            accuracy = self.mean_accuracy(
                weights=outputs,
                targets=inputs
            )
            mean_accuracy += to_numpy(accuracy)
        mean_loss /= len(data_loader)
        mean_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy
        )

    @staticmethod
    def reconstruction_loss(x, x_recons):
        return Trainer.mean_crossentropy_loss(weights=x_recons, targets=x)