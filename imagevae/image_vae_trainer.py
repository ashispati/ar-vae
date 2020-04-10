import os
import json
import numpy as np
import torch
import multiprocessing
from PIL import Image
from typing import Tuple
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from utils.trainer import Trainer
from utils.helpers import to_cuda_variable, to_numpy
from utils.plotting import plot_dim, save_gif, save_gif_from_list
from utils.evaluation import *
from imagevae.mnist_vae import MnistVAE
from imagevae.mnist_resnet import MnistResNet
from data.dataloaders.morphomnist.measure import measure_batch

MNIST_REG_TYPES = {
    "digit_identity": 0,
    "area": 1,
    "length": 2,
    "thickness": 3,
    "slant": 4,
    "width": 5,
    "height": 6
}

MNIST_NORMALIZATION_FACTORS = {
    "digit_identity": (0, 9),
    "area": (0, 350),
    "length": (0, 100),
    "thickness": (0, 15),
    "slant": (-1.2, 1.2),
    "width": (0, 30),
    "height": (0, 30)
}

DSPRITES_REG_TYPE = {
    "color": 0,
    "shape": 1,
    "scale": 2,
    "orientation": 3,
    "posx": 4,
    "posy": 5
}

DATASET_REG_TYPE_DICT = {
    'mnist': MNIST_REG_TYPES,
    'dsprites': DSPRITES_REG_TYPE
}


def get_reg_dim(attr_dict):
    reg_dim = []
    for r in attr_dict.keys():
        if r == 'digit_identity' or r == 'color':
            continue
        reg_dim.append(attr_dict[r])
    reg_dim = tuple(reg_dim)
    return reg_dim


class ImageVAETrainer(Trainer):
    def __init__(
            self,
            dataset,
            model: MnistVAE,
            lr=1e-4,
            reg_type: Tuple[str] = None,
            reg_dim: Tuple[int] = 0,
            dec_dist='bernoulli',
            beta=4.0,
            gamma=10.0,
            capacity=0.0,
            rand=0,
            delta=1.0,
    ):
        super(ImageVAETrainer, self).__init__(dataset, model, lr)
        if dataset.__class__.__name__ == 'MorphoMnistDataset':
            self.dataset_type = 'mnist'
        elif dataset.__class__.__name__ == 'DspritesDataset':
            self.dataset_type = 'dsprites'
        else:
            raise ValueError(f"Dataset type not recognized: {dataset.__class__.__name__}")
        self.attr_dict = DATASET_REG_TYPE_DICT[self.dataset_type]

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
        self.dec_dist = dec_dist
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
        if self.dataset_type == 'mnist':
            inputs, _, morpho_labels = batch
            inputs = to_cuda_variable(inputs)
            morpho_labels = to_cuda_variable(morpho_labels)
            return inputs, morpho_labels
        else:
            inputs, labels = batch
            inputs = to_cuda_variable(inputs)
            labels = to_cuda_variable(labels)
            return inputs, labels

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
        inputs, labels = batch

        # perform forward pass of model
        outputs, z_dist, prior_dist, z_tilde, z_prior = self.model(inputs)

        # compute reconstruction loss
        recons_loss = self.reconstruction_loss(inputs, outputs, self.dec_dist)

        # compute KLD loss
        dist_loss = self.compute_kld_loss(
            z_dist, prior_dist, beta=self.beta, c=self.capacity
        )

        # add losses
        loss = recons_loss + dist_loss

        # compute and add regularization loss if needed
        if self.use_reg_loss:
            reg_loss = 0.0
            if type(self.reg_dim) == tuple:
                for dim in self.reg_dim:
                    reg_loss += self.compute_reg_loss(
                        z_tilde, labels[:, dim], dim, gamma=self.gamma, factor=self.delta
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
            weights=torch.sigmoid(outputs),
            targets=inputs
        )

        if not train and batch_num == 0 and self.writer is not None:
            n = min(inputs.size(0), 16)
            recons = torch.sigmoid(outputs)
            comparison = torch.cat(
                [inputs[:n], recons[:n]]
            )
            image = make_grid(
                comparison.cpu(), nrow=n, pad_value=1.0
            )
            self.writer.add_image(
                'reconstruction', image, epoch_num
            )

        return loss, accuracy

    def eval_model(self, data_loader, epoch_num=0):
        if self.writer is not None:
            latent_codes, attributes, attr_list = self.compute_representations(data_loader)
            interp_metrics = compute_interpretability_metric(
                latent_codes, attributes, attr_list
            )
            metrics = {
                "interpretability": interp_metrics
            }
            for attr in interp_metrics.keys():
                self.writer.add_scalar(
                    'interpretability_metric/' + attr,
                    interp_metrics[attr][1],
                    epoch_num
                )
            if len(self.reg_dim) == 0:
                if self.dataset_type == 'mnist':
                    attr_str = 'slant'
                else:
                    attr_str = 'shape'
                dim1 = 0
                dim2 = 1
                interp = self.compute_latent_interpolations(latent_codes[:1, :], dim1)
            elif len(self.reg_dim) == 1:
                attr_str = self.reverse_attr_dict[self.reg_dim[0]]
                dim1 = self.reg_dim[0]
                dim2 = 0
                interp = self.compute_latent_interpolations(latent_codes[:1, :], dim1)
            else:
                attr_str = self.reverse_attr_dict[self.reg_dim[0]]
                dim1 = self.reg_dim[0]
                dim2 = self.reg_dim[1]
                interp = self.compute_latent_interpolations2d(latent_codes[:1, :], dim1, dim2)
            img = self.plot_data_dist(latent_codes, attributes, attr_str, dim1, dim2)
            img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
            self.writer.add_image(
                'attribute_distribution', img, epoch_num
            )
            self.writer.add_image(
                'interpolations', interp, epoch_num
            )
        else:
            metrics = self.compute_eval_metrics()
        return metrics

    def _extract_relevant_attributes(self, attributes):
        attr_list = [
            attr for attr in self.attr_dict.keys() if attr != 'digit_identity' and attr != 'color'
        ]
        attr_idx_list = [
            self.attr_dict[attr] for attr in attr_list
        ]
        attr_labels = attributes[:, attr_idx_list]
        return attr_labels, attr_list

    def compute_representations(self, data_loader):
        latent_codes = []
        attributes = []
        for sample_id, batch in tqdm(enumerate(data_loader)):
            inputs, labels = self.process_batch_data(batch)
            _, _, _, z_tilde, _ = self.model(inputs)
            latent_codes.append(to_numpy(z_tilde.cpu()))
            attributes.append(to_numpy(labels))
            if sample_id == 200:
                break
        latent_codes = np.concatenate(latent_codes, 0)
        attributes = np.concatenate(attributes, 0)
        attributes, attr_list = self._extract_relevant_attributes(attributes)
        return latent_codes, attributes, attr_list

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
            batch_size = 128
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
            if self.dataset_type == 'mnist':
                self.metrics.update(self.get_resnet_accuracy())
            with open(results_fp, 'w') as outfile:
                json.dump(self.metrics, outfile, indent=2)
        return self.metrics

    def get_resnet_accuracy(self):
        if self.dataset_type != 'mnist':
            return None
        # instantiate Resnet model
        resnet_model = MnistResNet()
        if torch.cuda.is_available():
            resnet_model.load()
            resnet_model.cuda()
        else:
            resnet_model.load(cpu=True)
        batch_size = 128
        _, _, data_loader = self.dataset.data_loaders(batch_size=batch_size)
        interp_dict = self.metrics["interpretability"]
        input_acc = 0
        recons_acc = 0
        interp_acc = 0
        for sample_id, batch in tqdm(enumerate(data_loader)):
            inputs, digit_labels, _ = batch
            inputs = to_cuda_variable(inputs)
            digit_labels = to_cuda_variable(digit_labels)
            recons, _, _, z, _ = self.model(inputs)
            recons = torch.sigmoid(recons)
            # compute input and reconstruction accuracy on resnet
            pred_inputs = self.compute_mnist_digit_identity(resnet_model, inputs)
            pred_recons = self.compute_mnist_digit_identity(resnet_model, recons)
            input_acc += self.mean_accuracy_pred(pred_inputs, digit_labels)
            recons_acc += self.mean_accuracy_pred(pred_recons, digit_labels)
            dummy = 0
            num_interps = 10
            z = z.repeat(num_interps, 1)
            for attr_str in interp_dict.keys():
                z_copy = z.clone()
                x1 = np.linspace(-4, 4.0, num_interps)
                x1 = x1.repeat(z.size(0) // num_interps)
                x1 = torch.from_numpy(x1)
                dim = interp_dict[attr_str][0]
                z_copy[:, dim] = x1.contiguous()
                outputs = torch.sigmoid(self.model.decode(z_copy))
                pred_outputs = self.compute_mnist_digit_identity(resnet_model, outputs)
                repeated_labels = digit_labels.repeat(10)
                dummy += self.mean_accuracy_pred(pred_outputs, repeated_labels)
            interp_acc += dummy / len(interp_dict.keys())
        num_batches = sample_id + 1
        return {
            'digit_pred_acc': {
                'inputs': input_acc.item() / num_batches,
                'recons': recons_acc.item() / num_batches,
                'interp': interp_acc.item() / num_batches
            }
        }

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

    def compute_latent_interpolations(self, latent_code, dim1=0, num_points=10):
        x1 = torch.linspace(-4., 4.0, num_points)
        num_points = x1.size(0)
        z = to_cuda_variable(torch.from_numpy(latent_code))
        z = z.repeat(num_points, 1)
        z[:, dim1] = x1.contiguous()
        outputs = torch.sigmoid(self.model.decode(z))
        interp = make_grid(outputs.cpu(), nrow=num_points, pad_value=1.0)
        return interp

    def compute_latent_interpolations2d(self, latent_code, dim1=0, dim2=1, num_points=10):
        x1 = torch.linspace(-4., 4.0, num_points)
        x2 = torch.linspace(-4., 4.0, num_points)
        z1, z2 = torch.meshgrid([x1, x2])
        num_points = z1.size(0) * z1.size(1)
        z = to_cuda_variable(torch.from_numpy(latent_code))
        z = z.repeat(num_points, 1)
        z[:, dim1] = z1.contiguous().view(1, -1)
        z[:, dim2] = z2.contiguous().view(1, -1)
        # z = torch.flip(z, dims=[0])
        outputs = torch.sigmoid(self.model.decode(z))
        interp = make_grid(outputs.cpu(), nrow=z1.size(0), pad_value=1.0)
        return interp

    def plot_latent_reconstructions(self, num_points=10):
        _, _, data_loader = self.dataset.data_loaders(batch_size=num_points)
        for sample_id, batch in tqdm(enumerate(data_loader)):
            inputs, labels = self.process_batch_data(batch)
            inputs = to_cuda_variable(inputs)
            recons, _, _, z, _ = self.model(inputs)
            recons = torch.sigmoid(recons)
            # save original image
            org_save_filepath = os.path.join(
                Trainer.get_save_dir(self.model),
                f'r_original_{sample_id}.png'
            )
            save_image(
                inputs.cpu(), org_save_filepath, nrow=num_points, pad_value=1.0
            )
            # save reconstruction
            recons_save_filepath = os.path.join(
                Trainer.get_save_dir(self.model),
                f'r_recons_{sample_id}.png'
            )
            save_image(
                recons.cpu(), recons_save_filepath, nrow=num_points, pad_value=1.0
            )
            break

    def create_latent_gifs(self, sample_id=9, num_points=10):
        x1 = torch.linspace(-4, 4.0, num_points)
        _, _, data_loader = self.dataset.data_loaders(batch_size=1)
        interp_dict = self.compute_eval_metrics()["interpretability"]
        for sid, batch in tqdm(enumerate(data_loader)):
            if sid == sample_id:
                inputs, labels = self.process_batch_data(batch)
                inputs = to_cuda_variable(inputs)
                _, _, _, z, _ = self.model(inputs)
                z = z.repeat(num_points, 1)
                outputs = []
                for attr_str in self.attr_dict.keys():
                    if attr_str == 'digit_identity' or attr_str == 'color':
                        continue
                    dim = interp_dict[attr_str][0]
                    z_copy = z.clone()
                    z_copy[:, dim] = x1.contiguous()

                    outputs.append(torch.sigmoid(self.model.decode(z_copy)))
                outputs = torch.unsqueeze(torch.cat(outputs, dim=1), dim=2)
                interps = []
                for n in range(outputs.shape[0]):
                    image_grid = make_grid(outputs[n], padding=2, pad_value=1.0).detach().cpu()
                    np_image = image_grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                    interps.append(Image.fromarray(np_image))
                # save gif
                gif_filepath = os.path.join(
                    Trainer.get_save_dir(self.model),
                    f'gif_interpolations_{self.dataset_type}_{sample_id}.gif'
                )
                save_gif_from_list(
                    interps, gif_filepath
                )
            if sid > sample_id:
                break

    def plot_latent_interpolations(self, attr_str='slant', num_points=10):
        x1 = torch.linspace(-4, 4.0, num_points)
        _, _, data_loader = self.dataset.data_loaders(batch_size=1)
        interp_dict = self.compute_eval_metrics()["interpretability"]
        dim = interp_dict[attr_str][0]
        for sample_id, batch in tqdm(enumerate(data_loader)):
            # for MNIST [5, 1, 30, 19, 23, 21, 17, 61, 9, 28]
            if sample_id in [5, 1, 30, 19, 23, 21, 17, 61, 9, 28]:
                inputs, labels = self.process_batch_data(batch)
                inputs = to_cuda_variable(inputs)
                recons, _, _, z, _ = self.model(inputs)
                recons = torch.sigmoid(recons)
                z = z.repeat(num_points, 1)
                z[:, dim] = x1.contiguous()
                outputs = torch.sigmoid(self.model.decode(z))
                # save interpolation
                save_filepath = os.path.join(
                    Trainer.get_save_dir(self.model),
                    f'latent_interpolations_{attr_str}_{sample_id}.png'
                )
                save_image(
                    outputs.cpu(), save_filepath, nrow=num_points, pad_value=1.0
                )
                # save original image
                org_save_filepath = os.path.join(
                    Trainer.get_save_dir(self.model),
                    f'original_{sample_id}.png'
                )
                save_image(
                    inputs.cpu(), org_save_filepath, nrow=1, pad_value=1.0
                )
                # save reconstruction
                recons_save_filepath = os.path.join(
                    Trainer.get_save_dir(self.model),
                    f'recons_{sample_id}.png'
                )
                save_image(
                    recons.cpu(), recons_save_filepath, nrow=1, pad_value=1.0
                )
            if sample_id == 62:
                break

    def plot_latent_interpolations2d(self, attr_str1, attr_str2, num_points=10):
        x1 = torch.linspace(-4., 4.0, num_points)
        x2 = torch.linspace(-4., 4.0, num_points)
        z1, z2 = torch.meshgrid([x1, x2])
        total_num_points = z1.size(0) * z1.size(1)
        _, _, data_loader = self.dataset.data_loaders(batch_size=1)
        interp_dict = self.compute_eval_metrics()["interpretability"]
        dim1 = interp_dict[attr_str1][0]
        dim2 = interp_dict[attr_str2][0]
        for sample_id, batch in tqdm(enumerate(data_loader)):
            if sample_id == 9:
                inputs, labels = self.process_batch_data(batch)
                inputs = to_cuda_variable(inputs)
                recons, _, _, z, _ = self.model(inputs)
                recons = torch.sigmoid(recons)
                z = z.repeat(total_num_points, 1)
                z[:, dim1] = z1.contiguous().view(1, -1)
                z[:, dim2] = z2.contiguous().view(1, -1)
                # z = torch.flip(z, dims=[0])
                outputs = torch.sigmoid(self.model.decode(z))
                save_filepath = os.path.join(
                    Trainer.get_save_dir(self.model),
                    f'latent_interpolations_2d_({attr_str1},{attr_str2})_{sample_id}.png'
                )
                save_image(
                    outputs.cpu(), save_filepath, nrow=num_points, pad_value=1.0
                )
                # save original image
                org_save_filepath = os.path.join(
                    Trainer.get_save_dir(self.model),
                    f'original_{sample_id}.png'
                )
                save_image(
                    inputs.cpu(), org_save_filepath, nrow=1, pad_value=1.0
                )
                # save reconstruction
                recons_save_filepath = os.path.join(
                    Trainer.get_save_dir(self.model),
                    f'recons_{sample_id}.png'
                )
                save_image(
                    recons.cpu(), recons_save_filepath, nrow=1, pad_value=1.0
                )
            if sample_id == 10:
                break

    def plot_latent_surface(self, attr_str, dim1=0, dim2=1, grid_res=0.1):
        # create the dataspace
        x1 = torch.arange(-5., 5., grid_res)
        x2 = torch.arange(-5., 5., grid_res)
        z1, z2 = torch.meshgrid([x1, x2])
        num_points = z1.size(0) * z1.size(1)
        z = torch.randn(1, self.model.z_dim)
        z = z.repeat(num_points, 1)
        z[:, dim1] = z1.contiguous().view(1, -1)
        z[:, dim2] = z2.contiguous().view(1, -1)
        z = to_cuda_variable(z)

        mini_batch_size = 500
        num_mini_batches = num_points // mini_batch_size
        attr_labels_all = []
        for i in tqdm(range(num_mini_batches)):
            z_batch = z[i * mini_batch_size:(i + 1) * mini_batch_size, :]
            outputs = torch.sigmoid(self.model.decode(z_batch))
            labels = self.compute_mnist_morpho_labels(outputs, attr_str)
            attr_labels_all.append(torch.from_numpy(labels))
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
            inputs, _ = self.process_batch_data(batch)
            inputs = to_cuda_variable(inputs)
            # compute forward pass
            outputs, _, _, _, _ = self.model(inputs)
            # compute loss
            recons_loss = self.reconstruction_loss(
                inputs, outputs, self.dec_dist
            )
            loss = recons_loss
            # compute mean loss and accuracy
            mean_loss += to_numpy(loss.mean())
            accuracy = self.mean_accuracy(
                weights=torch.sigmoid(outputs),
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
    def reconstruction_loss(x, x_recons, dist):
        batch_size = x.size(0)
        if dist == 'bernoulli':
            recons_loss = F.binary_cross_entropy_with_logits(
                x_recons, x, reduction='sum'
            ).div(batch_size)
        elif dist == 'gaussian':
            x_recons = torch.sigmoid(x_recons)
            recons_loss = F.mse_loss(
                x_recons, x, reduction='sum'
            ).div(batch_size)
        else:
            raise AttributeError("invalid dist")
        return recons_loss

    @staticmethod
    def mean_accuracy(weights, targets):
        """
        Evaluates the mean accuracy in prediction
        :param weights: torch Variable,
                (batch_size, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, seq_len)
        :return float, accuracy
        """
        predictions = torch.zeros_like(weights)
        predictions[weights >= 0.5] = 1
        binary_targets = torch.zeros_like(targets)
        binary_targets[targets >= 0.5] = 1
        correct = predictions == binary_targets
        acc = torch.sum(correct.float()) / binary_targets.view(-1).size(0)
        return acc

    @staticmethod
    def mean_accuracy_pred(pred_labels, gt_labels):
        correct = pred_labels.long() == gt_labels.long()
        return torch.sum(correct.float()) / pred_labels.size(0)

    @staticmethod
    def compute_mnist_digit_identity(resnet_model, outputs):
        resnet_model.eval()
        labels = torch.max(resnet_model(outputs), 1)[1]
        return labels

    def compute_mnist_morpho_labels(self, outputs, morpho_attr_str=None):
        with multiprocessing.Pool() as pool:
            a = outputs.detach().cpu().numpy().squeeze(axis=1)
            labels = measure_batch(a, pool=pool).values
        if morpho_attr_str is not None:
            labels = labels[:, self.attr_dict[morpho_attr_str] - 1]
        return labels
