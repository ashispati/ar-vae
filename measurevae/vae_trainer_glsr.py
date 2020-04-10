import torch

from data.dataloaders.bar_dataset_helpers import (
    RHY_COMPLEXITY_COEFFS, START_SYMBOL, END_SYMBOL, SLUR_SYMBOL
)
from measurevae.measure_vae import MeasureVAE
from measurevae.measure_vae_trainer import MeasureVAETrainer
from utils.helpers import to_cuda_variable


class MeasureVAETrainerGLSR(MeasureVAETrainer):
    def __init__(
            self, dataset,
            model: MeasureVAE,
            lr=1e-4,
            has_reg_loss=False,
            reg_type=None,
            reg_dim=0,
    ):
        super(MeasureVAETrainerGLSR, self).__init__(
            dataset,
            model,
            lr,
            has_reg_loss,
            reg_type,
            reg_dim
        )
        self.trainer_config += 'GLSR'
        self.model.update_trainer_config(self.trainer_config)
        self.note_tensor = self.is_note_tensor()

    def is_note_tensor(self):
        note2index = self.dataset.note2index_dicts
        slur_index = note2index[SLUR_SYMBOL]
        rest_index = note2index['rest']
        none_index = note2index[None]
        start_index = note2index[START_SYMBOL]
        end_index = note2index[END_SYMBOL]
        not_notes_indexes = [slur_index, rest_index, none_index, start_index, end_index]
        num_notes = len(note2index)
        is_note_tensor = torch.ones(num_notes)
        is_note_tensor[not_notes_indexes] = 0
        return is_note_tensor

    def compute_reg_loss(self, z, score, epsilon=1e-3):
        """
        Compute the GLSR regularization loss
        :param z:
        :param score:
        :param epsilon:
        :return:
        """
        d_z = torch.zeros_like(z)
        deltas = (1 + torch.rand(z.size(0))) * epsilon
        deltas = to_cuda_variable(deltas)
        d_z[: self.reg_dim] = deltas
        z_plus = z + d_z
        z_minus = z - d_z

        dummy_score_tensor = to_cuda_variable(
            torch.zeros(z.size(0), self.model.num_ticks_per_measure)
        )
        weights_plus, _ = self.model.decoder(
            z=z_plus,
            score_tensor=dummy_score_tensor,
            train=False
        )
        weights_minus, _ = self.model.decoder(
            z=z_minus,
            score_tensor=dummy_score_tensor,
            train=False
        )

        softmax_weights_plus = F.softmax(weights_plus, dim=2)
        softmax_weights_minus = F.softmax(weights_minus, dim=2)
        grad_softmax = softmax_weights_plus - softmax_weights_minus

        grad_attr = self.compute_grad_attr(grad_softmax)
        grad_attr = grad_attr / (2 * deltas)

        prior_mean = to_cuda_variable(torch.ones_like(grad_attr) * 100)
        prior_std = to_cuda_variable(torch.ones_like(grad_attr))
        reg_loss = -torch.distributions.Normal(prior_mean, prior_std).log_prob(grad_attr)
        return reg_loss.mean()

    def compute_grad_attr(self, softmax_weights):
        mask = to_cuda_variable(
            self.note_tensor[None, None, :].expand(
                softmax_weights.size()).detach()
        )
        if self.reg_type == 'rhy_complexity':
            metrical_weights = RHY_COMPLEXITY_COEFFS
            metrical_weights = to_cuda_variable(
                metrical_weights[None, :, None].expand(
                    softmax_weights.size()).detach()
            ).float()
            rhy_complexity = (softmax_weights * metrical_weights * mask).sum(2).sum(1) / sum(RHY_COMPLEXITY_COEFFS)
            return rhy_complexity
        elif self.reg_type == 'num_notes':
            measure_seq_len = softmax_weights.size(1)
            num_notes = (softmax_weights * mask).sum(2).sum(1) / measure_seq_len
            return num_notes
        else:
            raise ValueError('Invalid regularization type')
