import os
import click
import numpy as np
import pandas as pd
import torch
import json
from data.dataloaders.mnist_dataset import MorphoMnistDataset
from data.dataloaders.dsprites_dataset import DspritesDataset
from imagevae.mnist_vae import MnistVAE
from imagevae.dsprites_vae import DspritesVAE
from imagevae.image_vae_trainer import ImageVAETrainer, MNIST_REG_TYPES, DSPRITES_REG_TYPE, get_reg_dim
from utils.evaluation import EVAL_METRIC_DICT
from utils.plotting import create_scatter_plot


@click.command()
@click.option('--dataset_type', '-d', default='mnist',
              help='dataset to be used, `mnist` or `dsprites`')
@click.option('--batch_size', default=128,
              help='training batch size')
@click.option('--num_epochs', default=100,
              help='number of training epochs')
@click.option('--lr', default=1e-4,
              help='learning rate')
@click.option('--capacity', default=0.0,
              help='parameter for beta-VAE capacity')
@click.option('--dec_dist', default='bernoulli',
              help='distribution of the decoder')
@click.option('--train/--test', default=True,
              help='train or test the specified model')
@click.option('--log/--no_log', default=False,
              help='log the results for tensorboard')
def main(
        dataset_type,
        batch_size,
        num_epochs,
        lr,
        capacity,
        dec_dist,
        train,
        log,
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

    reg_type = (['all'])
    reg_dim = get_reg_dim(attr_dict)

    gamma = [0.01, 0.1, 1.0, 2.0, 5.0, 10.0, 100.0]
    delta = [100.0, 10.0, 1.0, 0.1, 0.01]
    results_list = list()
    for g in gamma:
        for d in delta:
            # instantiate trainer
            trainer = ImageVAETrainer(
                dataset=dataset,
                model=model,
                lr=lr,
                reg_type=reg_type,
                reg_dim=reg_dim,
                beta=1.0,
                capacity=capacity,
                gamma=g,
                delta=d,
                dec_dist=dec_dist,
                rand=0
            )
            file_exists = os.path.exists(trainer.model.filepath)
            # train if needed
            if not file_exists:
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

                    # plot interpolations
                    trainer.plot_latent_reconstructions()
                    for attr_str in trainer.attr_dict.keys():
                        if attr_str == 'digit_identity' or attr_str == 'color':
                            continue
                        trainer.plot_latent_interpolations(attr_str)

                    if dataset_type == 'mnist':
                        trainer.plot_latent_interpolations2d('slant', 'thickness')
                    else:
                        trainer.plot_latent_interpolations2d('posx', 'posy')
            else:
                temp_list = list()
                temp_list.append(f'={str(g)}')
                temp_list.append(f'={str(d)}')

                # fetch and store results
                trainer.load_model()
                trainer.writer = None
                r = trainer.compute_eval_metrics()
                for k in EVAL_METRIC_DICT.keys():
                    if k == 'interpretability':
                        temp_list.append(r[k]['mean'][1])
                    else:
                        temp_list.append(r[k])
                temp_list.append(r['test_acc'] * 100)
                results_list.append(temp_list)
    results_list = np.stack(results_list, axis=1)
    columnlist = [r'$\gamma$', r'$\delta$']
    columnlist += [EVAL_METRIC_DICT[k] for k in EVAL_METRIC_DICT.keys()]
    columnlist.append('Reconstruction Accuracy (in %)')
    df = pd.DataFrame(columns=columnlist, data=results_list.T)
    for k in EVAL_METRIC_DICT.keys():
        df[EVAL_METRIC_DICT[k]] = df[EVAL_METRIC_DICT[k]].astype(float)
    df['Reconstruction Accuracy (in %)'] = df['Reconstruction Accuracy (in %)'].astype(float)
    save_path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)), 'plots', f'hyper_param.pdf'
    )
    create_scatter_plot(
        df,
        x_axis='Interpretability',
        y_axis='Reconstruction Accuracy (in %)',
        grouping=r'$\gamma$',
        size=r'$\delta$',
        save_path=save_path
    )


if __name__ == '__main__':
    main()
