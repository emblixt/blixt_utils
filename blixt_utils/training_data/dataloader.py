import numpy as np
import os
import datetime
import logging
from torch.utils.data import Dataset

from blixt_utils.utils import arrange_logging
from blixt_utils.training_data.datagen import datagen


class DataLoader(Dataset):
    """
    This data loader creates the data as we iterate
    We want the validation data to be the same in each epoch, which is why the seeds/indexes for generating
    validation data is determined upon initialization.
    We save the first 1000 indexes for validation data.
    """
    def __init__(self, root_dir, validation=None, length=None, validation_length=None, seed=None):
        if validation is None:
            validation = False
        if length is None:
            length = 200
        if validation_length is None:
            validation_length = 20

        if not os.path.exists(root_dir):
            raise OSError('The folder {} is not found'.format(root_dir))

        this_date = datetime.datetime.now().strftime('%Y-%m-%dT%H.%M')
        arrange_logging(False, os.path.join(root_dir, '{} training.log'.format(this_date)))
        logging.info('Generating data on {}'.format(this_date))

        max_validation_index = 1000
        _rng = np.random.default_rng(seed)

        if validation_length >= max_validation_index:
            raise ValueError('Number of validation data must be less than {}'.format(max_validation_index))

        self.validation_length = validation_length
        self.max_validation_index = max_validation_index
        self.root_dir = root_dir
        self.validation = validation
        self.length = length
        self.validation_idxs = _rng.integers(max_validation_index, size=validation_length)

    def __len__(self):
        if self.validation:
            return self.validation_length
        else:
            return self.length

    def __getitem__(self, idx):
        if self.validation:
            # Return an index (= seed to the random generator) from the same list in each epoch
            if idx >= self.validation_length:
                raise ValueError('Index, {}, is larger than number of validation data, {}'.format(
                    idx, self.validation_length))
            this_rng = np.random.default_rng(self.validation_idxs[idx])
            _ = datagen(self.root_dir, self.validation_idxs[idx], 'Validation')
            return self.validation_idxs[idx], this_rng.integers(10)
        else:
            # Return an index (seed) randomly chosen in each iteration of each epoch
            this_rng = np.random.default_rng()
            _ = datagen(self.root_dir, self.max_validation_index + this_rng.integers(1000000), 'Training')
            return idx, this_rng.integers(10)


def test():
    val_data = DataLoader('C:\\tmp', validation=True)
    train_data = DataLoader('C:\\tmp')

    print(len(val_data), len(train_data))

    for i in range(2):
       _ = val_data[i]

    for i in range(2):
        _ = train_data[i]


if __name__ == '__main__':
    test()



