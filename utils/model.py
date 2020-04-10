import os
import torch
from torch import nn


class Model(torch.nn.Module):
    """
    Abstract model class
    """
    def __init__(self, filepath=None):
        super(Model, self).__init__()
        self.filepath = filepath
        self.trainer_config = ''

    def forward(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def update_filepath(self):
        """
        Updates the filepath
        :return:
        """
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.filepath = os.path.join(
            os.path.dirname(cur_dir),
            'models',
            self.__repr__(),
            self.__repr__() + '.pt'
        )

    def update_trainer_config(self, config):
        """
        Update the trainer configuration string
        :param config: str,
        :return:
        """
        self.trainer_config = config
        self.update_filepath()

    def save(self):
        """
        Saves the model
        :return: None
        """
        save_dir = os.path.dirname(self.filepath)
        # create save directory if needed
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.state_dict(), self.filepath)
        print(f'Model {self.__repr__()} saved')

    def save_checkpoint(self, epoch_num):
        """
        Saves the model checkpoints
        :param epoch_num: int,
        :return: None
        """
        save_dir = os.path.dirname(self.filepath)
        # create save directory if needed
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(
            os.path.dirname(self.filepath),
            self.__repr__() + '_' + str(epoch_num) + '.pt'
        )
        torch.save(self.state_dict(), filename)
        # print(f'Model checkpoint {self.__repr__()} saved for epoch')

    def load(self, cpu=False):
        """
        Loads the model
        :param cpu: bool, specifies if the model should be loaded on the CPU
        :return: None
        """
        if cpu:
            self.load_state_dict(
                torch.load(
                    self.filepath,
                    map_location=lambda storage,
                    loc: storage
                )
            )
        else:
            self.load_state_dict(torch.load(self.filepath))
        # print(f'Model {self.__repr__()} loaded')

    def xavier_initialization(self):
        """
        Initializes the network params
        :return:
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)