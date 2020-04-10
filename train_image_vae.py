import os
import click
import torch
import json
from data.dataloaders.mnist_dataset import MorphoMnistDataset
from data.dataloaders.dsprites_dataset import DspritesDataset
from imagevae.mnist_vae import MnistVAE
from imagevae.dsprites_vae import DspritesVAE
from imagevae.image_vae_trainer import ImageVAETrainer, MNIST_REG_TYPES, DSPRITES_REG_TYPE


@click.command()
@click.option('--dataset_type', '-d', default='mnist',
              help='dataset to be used, `mnist` or `dsprites`')
@click.option('--batch_size', default=128,
              help='training batch size')
@click.option('--num_epochs', default=100,
              help='number of training epochs')
@click.option('--lr', default=1e-4,
              help='learning rate')
@click.option('--beta', default=4.0,
              help='parameter for weighting KLD loss')
@click.option('--capacity', default=0.0,
              help='parameter for beta-VAE capacity')
@click.option('--gamma', default=10.0,
              help='parameter for weighting regularization loss')
@click.option('--delta', default=1.0,
              help='parameter for controlling the spread')
@click.option('--dec_dist', default='bernoulli',
              help='distribution of the decoder')
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
        batch_size,
        num_epochs,
        lr,
        beta,
        capacity,
        gamma,
        delta,
        dec_dist,
        train,
        log,
        rand,
        reg_type,
):
    if dataset_type == 'mnist':
        dataset = MorphoMnistDataset()
        model = MnistVAE()
        attr_dict = MNIST_REG_TYPES
    elif dataset_type == 'dsprites':
        dataset = DspritesDataset()
        model = DspritesVAE()
        attr_dict = DSPRITES_REG_TYPE
    else:
        raise ValueError("Invalid dataset_type. Choose between mnist and dsprites")

    if len(reg_type) != 0:
        if len(reg_type) == 1:
            if reg_type[0] == 'all':
                reg_dim = []
                for r in attr_dict.keys():
                    if r == 'digit_identity' or r == 'color':
                        continue
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
        trainer = ImageVAETrainer(
            dataset=dataset,
            model=model,
            lr=lr,
            reg_type=reg_type,
            reg_dim=reg_dim,
            beta=beta,
            capacity=capacity,
            gamma=gamma,
            delta=delta,
            dec_dist=dec_dist,
            rand=r
        )

        # train if needed
        if train:
            if torch.cuda.is_available():
                trainer.cuda()
            trainer.train_model(
                batch_size=batch_size,
                num_epochs=num_epochs,
                log=log
            )

        # compute and print evaluation metrics
        trainer.load_model()
        trainer.writer = None
        metrics = trainer.compute_eval_metrics()
        print(json.dumps(metrics, indent=2))

        for sample_id in [0, 1, 4]:
            trainer.create_latent_gifs(sample_id=sample_id)

        # interp_dict = metrics['interpretability']
        # if dataset_type == 'mnist':
        #     attr_dims = [interp_dict[attr][0] for attr in trainer.attr_dict.keys() if attr != 'digit_identity']
        #     non_attr_dims = [a for a in range(trainer.model.z_dim) if a not in attr_dims]
        #     for attr in interp_dict.keys():
        #         dim1 = interp_dict[attr][0]
        #         trainer.plot_latent_surface(
        #             attr,
        #             dim1=dim1,
        #             dim2=non_attr_dims[-1],
        #             grid_res=0.05,
        #         )

        # # plot interpolations
        # trainer.plot_latent_reconstructions()
        # for attr_str in trainer.attr_dict.keys():
        #     if attr_str == 'digit_identity' or attr_str == 'color':
        #         continue
        #     trainer.plot_latent_interpolations(attr_str)

        # if dataset_type == 'mnist':
        #     trainer.plot_latent_interpolations2d('slant', 'thickness')
        # else:
        #     trainer.plot_latent_interpolations2d('posx', 'posy')


if __name__ == '__main__':
    main()
