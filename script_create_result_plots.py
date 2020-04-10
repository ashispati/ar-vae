import click
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from data.dataloaders.mnist_dataset import MorphoMnistDataset
from data.dataloaders.dsprites_dataset import DspritesDataset
from data.dataloaders.bar_dataset import ChoraleNBarDataset, FolkNBarDataset
from imagevae.mnist_vae import MnistVAE
from imagevae.dsprites_vae import DspritesVAE
from measurevae.measure_vae import MeasureVAE
from imagevae.image_vae_trainer import ImageVAETrainer, MNIST_REG_TYPES, DSPRITES_REG_TYPE, get_reg_dim
from measurevae.measure_vae_trainer import MeasureVAETrainer, MUSIC_REG_TYPE
from utils.plotting import create_box_plot, create_pair_plot
from utils.evaluation import EVAL_METRIC_DICT


def main():
    dataset_dict = {
        'dsprites': {
            'repr': '2-d sprites',
            'attr_dict': DSPRITES_REG_TYPE,
            'dataset': DspritesDataset(),
            'model': DspritesVAE(),
            'trainer': ImageVAETrainer,
            'model_dict': {
                r'$\beta$-VAE': {
                    'metric_dlist': [],
                    'reg_type': (),
                    'reg_dim': tuple([0]),
                    'beta': 4.0,
                    'capacity': 0.0,
                    'gamma': 0.0
                },
                'AR-VAE': {
                    'metric_dlist': [],
                    'reg_type': (['all']),
                    'reg_dim': get_reg_dim(DSPRITES_REG_TYPE),
                    'beta': 1.0,
                    'capacity': 0.0,
                    'gamma': 10.0
                }
            },
        },
        'mnist': {
            'repr': 'Morpho-MNIST',
            'attr_dict': MNIST_REG_TYPES,
            'dataset': MorphoMnistDataset(),
            'model': MnistVAE(),
            'trainer': ImageVAETrainer,
            'model_dict': {
                r'$\beta$-VAE': {
                    'metric_dlist': [],
                    'reg_type': (),
                    'reg_dim': tuple([0]),
                    'beta': 4.0,
                    'capacity': 0.0,
                    'gamma': 0.0
                },
                'AR-VAE': {
                    'metric_dlist': [],
                    'reg_type': (['all']),
                    'reg_dim': get_reg_dim(MNIST_REG_TYPES),
                    'beta': 1.0,
                    'capacity': 0.0,
                    'gamma': 10.0
                }
            },
        },
        'bach': {
            'repr': 'Bach Chorales',
            'attr_dict': MUSIC_REG_TYPE,
            'dataset': ChoraleNBarDataset(
                dataset_type='train',
                is_short=False,
                num_bars=1,
            ),
            'model': MeasureVAE(
                dataset=ChoraleNBarDataset(
                    dataset_type='train',
                    is_short=False,
                    num_bars=1,
                ),
                note_embedding_dim=10,
                metadata_embedding_dim=2,
                num_encoder_layers=2,
                encoder_hidden_size=128,
                encoder_dropout_prob=0.5,
                latent_space_dim=32,
                num_decoder_layers=2,
                decoder_hidden_size=128,
                decoder_dropout_prob=0.5,
                has_metadata=False,
                dataset_type='bach',
            ),
            'trainer': MeasureVAETrainer,
            'model_dict': {
                r'$\beta$-VAE': {
                    'metric_dlist': [],
                    'reg_type': (),
                    'reg_dim': tuple([0]),
                    'beta': 0.001,
                    'capacity': 0.0,
                    'gamma': 0.0
                },
                'AR-VAE': {
                    'metric_dlist': [],
                    'reg_type': (['all']),
                    'reg_dim': get_reg_dim(MUSIC_REG_TYPE),
                    'beta': 0.001,
                    'capacity': 0.0,
                    'gamma': 1.0
                },
            },
        },
        'folk': {
            'repr': 'Folk Music',
            'attr_dict': MUSIC_REG_TYPE,
            'dataset': FolkNBarDataset(
                dataset_type='train',
                is_short=False,
                num_bars=1,
            ),
            'model': MeasureVAE(
                dataset=FolkNBarDataset(
                    dataset_type='train',
                    is_short=False,
                    num_bars=1,
                ),
                note_embedding_dim=10,
                metadata_embedding_dim=2,
                num_encoder_layers=2,
                encoder_hidden_size=128,
                encoder_dropout_prob=0.5,
                latent_space_dim=32,
                num_decoder_layers=2,
                decoder_hidden_size=128,
                decoder_dropout_prob=0.5,
                has_metadata=False,
                dataset_type='folk',
            ),
            'trainer': MeasureVAETrainer,
            'model_dict': {
                r'$\beta$-VAE': {
                    'metric_dlist': [],
                    'reg_type': (),
                    'reg_dim': tuple([0]),
                    'beta': 0.001,
                    'capacity': 0.0,
                    'gamma': 0.0
                },
                'AR-VAE': {
                    'metric_dlist': [],
                    'reg_type': (['all']),
                    'reg_dim': get_reg_dim(MUSIC_REG_TYPE),
                    'beta': 0.001,
                    'capacity': 0.0,
                    'gamma': 1.0
                },
            },
        },
    }

    rand = range(0, 10)
    for d in dataset_dict.keys():
        for m in dataset_dict[d]['model_dict'].keys():
            for r in rand:
                # instantiate trainer
                trainer = dataset_dict[d]['trainer'](
                    dataset=dataset_dict[d]['dataset'],
                    model=dataset_dict[d]['model'],
                    lr=1e-4,
                    reg_type=dataset_dict[d]['model_dict'][m]['reg_type'],
                    reg_dim=dataset_dict[d]['model_dict'][m]['reg_dim'],
                    beta=dataset_dict[d]['model_dict'][m]['beta'],
                    capacity=dataset_dict[d]['model_dict'][m]['capacity'],
                    gamma=dataset_dict[d]['model_dict'][m]['gamma'],
                    rand=r
                )
                # compute and print evaluation metrics
                trainer.load_model()
                trainer.writer = None
                dataset_dict[d]['model_dict'][m]['metric_dlist'].append(
                    trainer.compute_eval_metrics()
                )

    # Plot Digit Prediction Plot
    digit_pred_data = []
    digit_pred_crit = {
        'recons': 'Reconstructed',
        'interp': 'Interpolated',
    }
    for k in digit_pred_crit:
        for m in dataset_dict['mnist']['model_dict'].keys():
            temp_list = list()
            temp_list.append(
                [r['digit_pred_acc'][k]*100 for r in dataset_dict['mnist']['model_dict'][m]['metric_dlist']]
            )
            n = len(dataset_dict['mnist']['model_dict'][m]['metric_dlist'])
            temp_list.append(n * [digit_pred_crit[k]])
            temp_list.append(n * [m])
            digit_pred_data.append(temp_list)
    digit_pred_data = np.concatenate(digit_pred_data, axis=1)
    df = pd.DataFrame(columns=['Accuracy (in %)', 'Criteria', 'Model'], data=digit_pred_data.T)
    df['Accuracy (in %)'] = df['Accuracy (in %)'].astype(float)
    save_path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)), 'plots', 'digit_pred_acc.pdf'
    )
    fig, ax = create_box_plot(df, 'Criteria', 'Accuracy (in %)', 'Model', width=0.5)
    plt.plot(0.5, 96.15, 'x', color='k')
    ax.annotate(r'MNIST Test Set', (0.4, 97.55))
    plt.savefig(save_path)

    # Plot Reconstruction accuracy plots
    recons_data = []
    for d in dataset_dict.keys():
        for m in dataset_dict[d]['model_dict'].keys():
            temp_list = list()
            temp_list.append([r['test_acc']*100 for r in dataset_dict[d]['model_dict'][m]['metric_dlist']])
            n = len(dataset_dict[d]['model_dict'][m]['metric_dlist'])
            temp_list.append(n * [dataset_dict[d]['repr']])
            temp_list.append(n * [m])
            recons_data.append(temp_list)

    recons_data = np.concatenate(recons_data, axis=1)
    df = pd.DataFrame(columns=['Reconstruction Accuracy (in %)', 'Datasets', 'Model'], data=recons_data.T)
    df['Reconstruction Accuracy (in %)'] = df['Reconstruction Accuracy (in %)'].astype(float)
    save_path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)), 'plots', 'reconstruction.pdf'
    )
    create_box_plot(df, 'Datasets', 'Reconstruction Accuracy (in %)', 'Model', save_path=save_path)

    # Plot Disentanglement Evaluation metrics
    data = {}
    for k in EVAL_METRIC_DICT.keys():
        data[k] = []
        for d in dataset_dict.keys():
            for m in dataset_dict[d]['model_dict'].keys():
                temp_list = list()
                if k == 'interpretability':
                    temp_list.append([r[k]['mean'][1] for r in dataset_dict[d]['model_dict'][m]['metric_dlist']])
                else:
                    temp_list.append([r[k] for r in dataset_dict[d]['model_dict'][m]['metric_dlist']])
                n = len(dataset_dict[d]['model_dict'][m]['metric_dlist'])
                temp_list.append(n * [dataset_dict[d]['repr']])
                temp_list.append(n * [m])
                data[k].append(temp_list)

        data[k] = np.concatenate(data[k], axis=1)
        df = pd.DataFrame(columns=[EVAL_METRIC_DICT[k], 'Datasets', 'Model'], data=data[k].T)
        df[EVAL_METRIC_DICT[k]] = df[EVAL_METRIC_DICT[k]].astype(float)
        save_path = os.path.join(
            os.path.realpath(os.path.dirname(__file__)), 'plots', f'evaluation_{EVAL_METRIC_DICT[k]}.pdf'
        )
        create_box_plot(df, 'Datasets', EVAL_METRIC_DICT[k], 'Model', save_path=save_path)

    # Plot Pairplot
    pairplot_data = []
    for d in dataset_dict.keys():
        for m in dataset_dict[d]['model_dict'].keys():
            N = len(dataset_dict[d]['model_dict'][m]['metric_dlist'])
            for n in range(N):
                temp_list = list()
                for k in EVAL_METRIC_DICT.keys():
                    if k == 'interpretability':
                        temp_list.append(dataset_dict[d]['model_dict'][m]['metric_dlist'][n][k]['mean'][1])
                    else:
                        temp_list.append(dataset_dict[d]['model_dict'][m]['metric_dlist'][n][k])
                if d == 'folk' or d == 'bach':
                    temp_list.append(f'{m}:Music')
                else:
                    temp_list.append(f'{m}:Image')
                pairplot_data.append(temp_list)
    pairplot_data = np.stack(pairplot_data, axis=1)
    columnlist = [EVAL_METRIC_DICT[k] for k in EVAL_METRIC_DICT.keys()]
    columnlist.append('Model')
    df = pd.DataFrame(columns=columnlist, data=pairplot_data.T)
    for k in EVAL_METRIC_DICT.keys():
        df[EVAL_METRIC_DICT[k]] = df[EVAL_METRIC_DICT[k]].astype(float)
    save_path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)), 'plots', f'pair_plot.pdf'
    )
    create_pair_plot(df, 'Model', save_path=save_path)


if __name__ == '__main__':
    main()
