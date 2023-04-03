import csv
import gc
import random

import numpy as np
import torch
from torch.utils.data import Dataset

# Set all random seeds so training is deterministic
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


class M4_Dataset(Dataset):
    def __init__(self, backcast_length: int, forecast_length: int, Lh_sampling_len: int) -> None:

        # Set this to path of input data
        train_dataset_path = "/data/M4DataSet/Monthly-train.csv"
        test_dataset_path = "/data/M-test-set/Monthly-test.csv"

        # List containing which lines in training aren't long enough to run on
        # True if we skip that line in data because it is not long enough
        tossed_data_mask = []

        # Min series length needed for training is:
        # val data + Lh sampling window + backcast from earliest sample in window
        # Min length = (len validation) + (Lh * (len forecast)) + (len backcast)
        min_length: int = int(forecast_length + Lh_sampling_len + backcast_length)

        # Instantiate empty arrays for train, val, and test data
        # Train will be Lh sampling len + backcast len long
        # Both val and test will be forecast long
        self.train = np.array([]).reshape(0, (Lh_sampling_len + backcast_length))
        self.val = np.array([]).reshape(0, forecast_length)
        self.test = np.array([]).reshape(0, forecast_length)

        # TRAIN READ
        # Go through csv file, read in data
        with open(train_dataset_path, "r") as file:
            reader = csv.reader(file, delimiter=",")
            next(reader, None)  # Skip first header row

            for line in reader:
                # Don't read in first column (series id)
                # Get array of just series values and toss nulls ("")
                series_array = np.array([s for s in line[1:] if s != ""], dtype=np.float32)

                # Toss all lines less than "min_length" long
                # If tossing, add to mask so we can remove these lines from the test set
                if len(series_array) < min_length:
                    tossed_data_mask.append(True)
                    continue

                # Train data is from [min_length: val start]
                # Validation data is just last forecast window of train series
                train_data = series_array[-(min_length):-forecast_length]
                validation_data = series_array[-forecast_length:]

                # Stack together the forecast and backcast values per series
                # x and y are matrix of backcast or forecast for full dataset
                self.train = np.vstack((self.train, train_data))
                self.val = np.vstack((self.val, validation_data))
                tossed_data_mask.append(False)

        # TEST READ
        # Go through csv file, read in data
        with open(test_dataset_path, "r") as file:
            reader = csv.reader(file, delimiter=",")
            next(reader, None)  # Skip first header row

            for idx, line in enumerate(reader):
                # If we tossed this row in training, skip it in test
                if tossed_data_mask[idx]:
                    continue

                # Don't read in first column (series id)
                self.test = np.vstack((self.test, np.array(line[1:], dtype=np.float32)))

        print(f"The input data had {len(tossed_data_mask)} series")
        print(f"The shape of the train dataset is {self.train.shape}")
        print(f"The shape of the val dataset is {self.val.shape}")
        print(f"The shape of the test dataset is {self.test.shape}")

        # Not sure if I need to gc but doing it just in case
        gc.collect()

    def __len__(self):
        return self.train.shape[0]

    def __getitem__(self, idx):
        # Return the idx'th element of train, val, and test
        return self.train[idx], self.val[idx], self.test[idx]