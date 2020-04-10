import os
import click
import json
import numpy as np
import torch
from tqdm import tqdm

from measurevae.measure_vae import MeasureVAE
from measurevae.measure_vae_trainer import MeasureVAETrainer, MUSIC_REG_TYPE
from data.dataloaders.bar_dataset import FolkNBarDataset, ChoraleNBarDataset


@click.command()
@click.option('--dataset_type', '-d', default='folk',
              help='dataset to be used, `bach` or `folk`')
@click.option('--note_embedding_dim', default=10,
              help='size of the note embeddings')
@click.option('--metadata_embedding_dim', default=2,
              help='size of the metadata embeddings')
@click.option('--num_encoder_layers', default=2,
              help='number of layers in encoder RNN')
@click.option('--encoder_hidden_size', default=128,
              help='hidden size of the encoder RNN')
@click.option('--encoder_dropout_prob', default=0.5,
              help='float, amount of dropout prob between encoder RNN layers')
@click.option('--has_metadata', default=False,
              help='bool, True if data contains metadata')
@click.option('--latent_space_dim', default=32,
              help='int, dimension of latent space parameters')
@click.option('--num_decoder_layers', default=2,
              help='int, number of layers in decoder RNN')
@click.option('--decoder_hidden_size', default=128,
              help='int, hidden size of the decoder RNN')
@click.option('--decoder_dropout_prob', default=0.5,
              help='float, amount got dropout prob between decoder RNN layers')
@click.option('--batch_size', default=256,
              help='training batch size')
@click.option('--num_epochs', default=30,
              help='number of training epochs')
@click.option('--lr', default=1e-4,
              help='learning rate')
@click.option('--beta', default=0.001,
              help='parameter for weighting KLD loss')
@click.option('--capacity', default=0.0,
              help='parameter for beta-VAE capacity')
@click.option('--gamma', default=1.0,
              help='parameter for weighting regularization loss')
@click.option('--delta', default=10.0,
              help='parameter for controlling the spread')
@click.option('--train/--test', default=True,
              help='train or test the specified model')
@click.option('--log/--no_log', default=False,
              help='log the results for tensorboard')
@click.option(
    '--rand',
    default=None,
    help='random seed for the random number generator'
)
@click.option(
    '--reg_type',
    '-r',
    default=None,
    multiple=True,
    help='attribute name string to be used for regularization'
)
def main(
        dataset_type,
        note_embedding_dim,
        metadata_embedding_dim,
        num_encoder_layers,
        encoder_hidden_size,
        encoder_dropout_prob,
        latent_space_dim,
        num_decoder_layers,
        decoder_hidden_size,
        decoder_dropout_prob,
        has_metadata,
        batch_size,
        num_epochs,
        lr,
        beta,
        capacity,
        gamma,
        delta,
        train,
        log,
        rand,
        reg_type,
):

    is_short = False
    num_bars = 1

    if dataset_type == 'bach':
        dataset = ChoraleNBarDataset(
            dataset_type='train',
            is_short=is_short,
            num_bars=num_bars
        )
    elif dataset_type == 'folk':
        dataset = FolkNBarDataset(
            dataset_type='train',
            is_short=is_short,
            num_bars=num_bars
        )
    else:
        raise ValueError("Invalid dataset_type. Choose between `folk` and `bach`")

    attr_dict = MUSIC_REG_TYPE
    if len(reg_type) != 0:
        if len(reg_type) == 1:
            if reg_type[0] == 'all':
                reg_dim = []
                for r in attr_dict.keys():
                    reg_dim.append(attr_dict[r])
            else:
                reg_dim = [attr_dict[reg_type]]
        else:
            reg_dim = []
            for r in reg_type:
                reg_dim.append(attr_dict[r])
    else:
        reg_dim = [0]
    reg_dim = tuple(reg_dim)

    if rand is None:
        rand = range(0, 10)
    else:
        rand = [int(rand)]
    for r in rand:
        # instantiate trainer
        model = MeasureVAE(
            dataset=dataset,
            note_embedding_dim=note_embedding_dim,
            metadata_embedding_dim=metadata_embedding_dim,
            num_encoder_layers=num_encoder_layers,
            encoder_hidden_size=encoder_hidden_size,
            encoder_dropout_prob=encoder_dropout_prob,
            latent_space_dim=latent_space_dim,
            num_decoder_layers=num_decoder_layers,
            decoder_hidden_size=decoder_hidden_size,
            decoder_dropout_prob=decoder_dropout_prob,
            has_metadata=has_metadata,
            dataset_type=dataset_type,
        )

        trainer = MeasureVAETrainer(
            dataset=dataset,
            model=model,
            lr=lr,
            reg_type=reg_type,
            reg_dim=reg_dim,
            beta=beta,
            capacity=capacity,
            gamma=gamma,
            delta=delta,
            rand=r
        )

        if train:
            if torch.cuda.is_available():
                trainer.cuda()
            trainer.train_model(
                batch_size=batch_size,
                num_epochs=num_epochs,
                log=log,
            )

        trainer.load_model()
        trainer.writer = None
        metrics = trainer.compute_eval_metrics()
        interp_dict = metrics["interpretability"]
        print(json.dumps(metrics, indent=2))

        # data_loader, _, _ = trainer.dataset.data_loaders(batch_size=256)
        # rhy_comp = []
        # not_den = []
        # for batch_num, batch in tqdm(enumerate(data_loader)):
        #     score_tensor, _ = trainer.process_batch_data(batch)
        #     rhy_comp.append(trainer.compute_attribute_labels(score_tensor, attr_list=['rhy_complexity']).numpy())
        #     not_den.append(trainer.compute_attribute_labels(score_tensor, attr_list=['note_density']).numpy())
        # rhy_comp = np.concatenate(rhy_comp)
        # not_den = np.concatenate(not_den)
        # from scipy.stats import spearmanr
        # rho, p = spearmanr(rhy_comp, not_den)
        # print(f'Coeff:{rho}, p-value:{p}')

        # _, _, data_loader = trainer.dataset.data_loaders(batch_size=128)
        # latent_codes, attributes, attr_list = trainer.compute_representations(
        #     data_loader=data_loader,
        #     num_batches=min(100, len(data_loader))
        # )
        # attr_dims = [interp_dict[attr][0] for attr in trainer.attr_dict.keys()]
        # non_attr_dims = [a for a in range(trainer.model.latent_space_dim) if a not in attr_dims]
        # for attr in trainer.attr_dict.keys():
        #     dim1 = interp_dict[attr][0]
        #     trainer.plot_data_dist(latent_codes, attributes, attr, dim1, non_attr_dims[-1])

        _, _, data_loader = trainer.dataset.data_loaders(batch_size=1)
        latent_codes, attributes, attr_list = trainer.compute_representations(data_loader=data_loader,)
        attr_dims = [interp_dict[attr][0] for attr in trainer.attr_dict.keys()]
        non_attr_dims = [a for a in range(trainer.model.latent_space_dim) if a not in attr_dims]
        for attr in trainer.attr_dict.keys():
            # dim1 = interp_dict[attr][0]
            # trainer.plot_latent_surface(
            #     attr,
            #     dim1=dim1,
            #     dim2=non_attr_dims[-1],
            #     grid_res=0.05,
            # )
            trainer.plot_latent_interpolations(latent_codes, attr_str=attr, num_points=20)


if __name__ == '__main__':
    main()
