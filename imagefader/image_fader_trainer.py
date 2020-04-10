import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from imagevae.image_vae_trainer import ImageVAETrainer, MNIST_NORMALIZATION_FACTORS
from utils.helpers import to_numpy, to_cuda_variable


class ImageFaderTrainer(ImageVAETrainer):
    def __init__(self, dataset_type, dataset, fader_model, disc_model, lr, beta):
        super(ImageFaderTrainer, self).__init__(
            dataset_type=dataset_type,
            dataset=dataset,
            model=fader_model,
            lr=lr,
            beta=beta,
            reg_type=(),
        )
        self.disc_model = disc_model
        self.disc_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.disc_model.parameters()),
            lr=lr
        )
        self.cur_epoch_num = 0
        self.curr_beta = 0.0
        self.num_ramp_steps = 3 * 1e4
        self.beta_delta = float(self.beta / self.num_ramp_steps)

    def cuda(self):
        """
        Convert the model to cuda
        """
        self.model.cuda()
        self.disc_model.cuda()

    def disc_zero_grad(self):
        """
        Zero the gradient of the discriminator
        """
        self.disc_optimizer.zero_grad()

    def disc_step(self):
        """
        Perform the backward pass and step update for the discriminator
        """
        self.disc_optimizer.step()

    def loss_and_acc_on_epoch(self, data_loader, epoch_num=None, train=True):
        """
        Computes the loss and accuracy for an epoch
        :param data_loader: torch dataloader object
        :param epoch_num: int, used to change training schedule
        :param train: bool, performs the backward pass and gradient descent if TRUE
        :return: loss values and accuracy percentages
        """
        mean_loss = 0
        mean_accuracy = 0
        for batch_num, batch in tqdm(enumerate(data_loader)):
            log = False
            if train and self.writer is not None:
                if self.cur_epoch_num != epoch_num:
                    log = True
                    self.cur_epoch_num = epoch_num
            # process batch data
            batch_data = self.process_batch_data(batch)
            inputs, labels = batch_data
            norm_labels = self.normalize_labels(labels.clone())
            flipped_norm_labels = 1.0 - norm_labels.clone()

            # Encode data
            z = self.model.encode(inputs)

            # TRAIN DISCRIMINATOR
            # zero the disc gradients
            self.disc_zero_grad()
            # compute loss for discriminator
            disc_loss = self.disc_loss_for_batch(
                z.detach(), norm_labels, epoch_num, log
            )
            # compute backward and step discriminator if train
            if train:
                disc_loss.backward()
                self.disc_step()

            # TRAIN FADER MODEL
            # zero the model gradients
            self.zero_grad()
            # compute fader model loss
            fader_loss, accuracy = self.fader_loss_for_batch(
                inputs, z, norm_labels, flipped_norm_labels, epoch_num, log
            )
            # compute backward and step if train
            if train:
                fader_loss.backward()
                self.step()

            # compute mean loss and accuracy
            mean_loss += to_numpy(fader_loss.mean())
            if accuracy is not None:
                mean_accuracy += to_numpy(accuracy)

            # update beta
            self.curr_beta += self.beta_delta

        mean_loss /= len(data_loader)
        mean_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy
        )

    def disc_loss_for_batch(
            self,
            z,
            norm_labels,
            epoch_num=None,
            log=False
    ):
        """
        Computes the discriminator loss
        :param z: torch Variable, latent code
        :param norm_labels: torch Variable, normalized attribute labels
        :param epoch_num: int, used to change training schedule
        :param log: bool, log to tensorboard if True
        """
        # predict attributes
        pred_labels = self.disc_model(z.detach())
        # compute loss
        disc_loss = self.compute_disc_loss(pred_labels, norm_labels)

        # add to tensorboard writer for visualization
        if log:
            self.writer.add_scalar('loss_split/disc_loss', disc_loss.item(), epoch_num)

        return disc_loss

    def fader_loss_for_batch(
            self,
            inputs,
            z,
            norm_labels,
            flipped_norm_labels,
            epoch_num=None,
            log=False):
        """
        Computes the loss and accuracy for the batch
        Must return (loss, accuracy) as a tuple, accuracy can be None
        :pram inputs: torch.Variable
        :param z: torch Variable,
        :param flipped_norm_labels: flipped normalized labels
        :param epoch_num: int, used to change training schedule
        :return: scalar loss value, scalar accuracy value
        :param log: bool, log to tensorboard if True
        """
        pred_labels = self.disc_model(z)
        # compute output
        outputs = self.model.decode(torch.cat((z, norm_labels), 1))
        # compute reconstruction loss
        rec_loss = self.reconstruction_loss(inputs, outputs, self.dec_dist)
        # compute fader loss
        beta = min(self.beta, self.curr_beta)
        beta = self.beta
        adv_loss = beta * self.compute_disc_loss(pred_labels, flipped_norm_labels)
        fader_loss = rec_loss + adv_loss

        # add to tensorboard writer for visualization
        if log:
            self.writer.add_scalar('loss_split/recons_loss', rec_loss.item(), epoch_num)
            self.writer.add_scalar('loss_split/adv_loss', adv_loss.item(), epoch_num)
            self.writer.add_scalar('params/beta', beta, epoch_num)

        # compute accuracy
        accuracy = self.mean_accuracy(
            weights=torch.sigmoid(outputs),
            targets=inputs
        )

        return fader_loss, accuracy

    def compute_representations(self, data_loader):
        latent_codes = []
        attributes = []
        for sample_id, batch in tqdm(enumerate(data_loader)):
            inputs, labels = self.process_batch_data(batch)
            norm_labels = self.normalize_labels(labels)
            z = self.model.encode(inputs)
            latent_codes.append(to_numpy(z.cpu()))
            attributes.append(to_numpy(norm_labels))
            if sample_id == 200:
                break
        latent_codes = np.concatenate(latent_codes, 0)
        attributes = np.concatenate(attributes, 0)
        attr_list = self._extract_relevant_attributes(attributes)
        return latent_codes, attributes, attr_list

    def _extract_relevant_attributes(self, attributes):
        attr_list = [
            attr for attr in self.attr_dict.keys() if attr != 'digit_identity' and attr != 'color'
        ]
        return attr_list

    def eval_model(self, data_loader, epoch_num=0):
        latent_codes, attributes, attr_list = self.compute_representations(data_loader)
        if self.writer is not None:
            for i, attr in enumerate(attr_list):
                interp = self.compute_latent_interpolations(
                    latent_codes, attributes, dim1=i
                )
                self.writer.add_image(
                    'fader_' + attr, interp, epoch_num
                )

    def compute_latent_interpolations(self, latent_code, labels, dim1=1):
        x1 = torch.arange(0., 1.01, 0.1)
        num_points = x1.size(0)
        z = to_cuda_variable(torch.from_numpy(latent_code[:1, :]))
        z = z.repeat(num_points, 1)
        l = labels[:1, :]
        l = l.repeat(num_points, 0)
        l[:, dim1] = x1.contiguous()
        l = to_cuda_variable(torch.from_numpy(l))
        inputs = torch.cat((z, l), 1)
        outputs = torch.sigmoid(self.model.decode(inputs))
        interp = make_grid(outputs.cpu(), nrow=1, pad_value=1.0)
        return interp

    def normalize_labels(self, labels):
        """
        Normalize labels between  0 to 1
        """
        if self.dataset_type == 'mnist':
            norm_labels = torch.clone(labels)
            for i, attr in enumerate(MNIST_NORMALIZATION_FACTORS.keys()):
                min_attr, max_attr = MNIST_NORMALIZATION_FACTORS[attr]
                norm_labels[:, i] = (labels[:, i] - min_attr) / (max_attr - min_attr)
            return norm_labels[:, 1:]
        elif self.dataset_type == 'dpsrites':
            raise ValueError("normalization method not defined for dsprites dataset")
        else:
            raise ValueError("Invalid dataset type. Should be `mnist` or `dpsrites`")

    @staticmethod
    def compute_disc_loss(pred_labels, ground_truth):
        batch_size = pred_labels.size(0)
        disc_loss = F.mse_loss(
            pred_labels, ground_truth, reduction='sum'
        ).div(batch_size)
        return disc_loss
