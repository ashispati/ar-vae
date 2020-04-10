import os
import click
import torch
import json
from data.dataloaders.mnist_dataset import MorphoMnistDataset
from data.dataloaders.dsprites_dataset import DspritesDataset
from imagefader.image_fader import MnistFaderNetwork, DspritesFaderNetwork, ImageFaderDiscriminator
from imagefader.image_fader_trainer import ImageFaderTrainer
from imagevae.image_vae_trainer import MNIST_REG_TYPES, DSPRITES_REG_TYPE


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
@click.option('--train/--test', default=True,
              help='train or test the specified model')
@click.option('--log/--no_log', default=False,
              help='log the results for tensorboard')
def main(
        dataset_type,
        batch_size,
        num_epochs,
        lr,
        beta,
        train,
        log,
):
    if dataset_type == 'mnist':
        dataset = MorphoMnistDataset()
        model = MnistFaderNetwork()
    elif dataset_type == 'dsprites':
        dataset = DspritesDataset()
        model = DspritesFaderNetwork()
    else:
        raise ValueError("Invalid dataset_type. Choose between mnist and dsprites")
    disc_model = ImageFaderDiscriminator(num_attributes=model.num_attributes)

    trainer = ImageFaderTrainer(
        dataset_type=dataset_type,
        dataset=dataset,
        fader_model=model,
        disc_model=disc_model,
        lr=lr,
        beta=beta,
    )

    if train:
        if torch.cuda.is_available():
            trainer.cuda()
        trainer.train_model(
            batch_size=batch_size,
            num_epochs=num_epochs,
            log=log
        )

    trainer.load_model()
    trainer.writer = None
    _, _, data_loader = trainer.dataset.data_loaders(batch_size=batch_size)
    # metrics = trainer.eval_model(data_loader)
    # print(json.dumps(metrics, indent=2))
    # file_name = os.path.join(
    #     os.path.dirname(trainer.model.filepath),
    #     'results_dict.json'
    # )
    # with open(file_name, 'w') as outfile:
    #     json.dump(metrics, outfile, indent=2)


if __name__ == '__main__':
    main()
