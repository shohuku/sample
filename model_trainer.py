import numpy as np
import torch
from torch import nn


def sMAPE(predictions, actuals):
    predictions = predictions.float()
    actuals = actuals.float()
    sumf = torch.sum(
        2 * torch.abs(predictions - actuals) / (torch.abs(predictions) + torch.abs(actuals))
    )
    return (100 / predictions.shape[1]) * sumf


class N_Beats_Trainer(nn.Module):
    def __init__(self, model, dl, backcast_length, forecast_length, Lh_sampling_len, device):
        super(N_Beats_Trainer, self).__init__()

        # Job configs and settings
        self.device = device
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.dl = dl
        self.epochs = 0
        self.test_loss = 0
        self.Lh_sampling_len = Lh_sampling_len

        # Move model to correct device if needed
        self.model = model.to(self.device)

        # Define optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-3)

        # Define loss function of sMAPE
        self.criterion = lambda output, target: sMAPE(output, target)

    def train(self):
        # Instantiate variables we want to track for one epoch of training
        self.model.train()
        total_epoch_loss = 0
        num_samples = 0

        # Go through full dataloader per epoch
        for (train, val, test) in self.dl:

            # For training, we sample an "anchor" point between [-Lh sampling window: -forecast_len] of train set
            # Then, we set forecast as the next forecast_length points after the anchor
            # Backcast will be the backcast_length before the anchor
            anchor_value = np.random.randint(
                self.forecast_length, self.Lh_sampling_len
            )  # What point in the past to pick as anchor

            # Backcast is the backcast_length before the anchor
            # Forecast is the forecast_length after the anchor
            backcast = train[
                :, -(self.backcast_length + anchor_value + 1) : -(anchor_value + 1)
            ].to(self.device)
            forecast = train[
                :, -(anchor_value + 1) : -(anchor_value - self.forecast_length + 1)
            ].to(self.device)

            # Run a batch of data through the model
            _, forecast_hat = self.model(backcast)

            # Calculate loss on predictions and backpropogate through model
            loss = self.criterion(forecast_hat.to(self.device), forecast)
            loss.backward()

            # Take step with optimizer, and then zero gradient after so its ready for next step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # We will use this to calculate epoch level loss after all batches
            total_epoch_loss += loss.item()
            num_samples += train.shape[0]  # Record number of series per batch

        # Calculate loss at epoch level for reporting
        avg_epoch_loss = total_epoch_loss / num_samples
        self.epochs += 1

        print(f"Epoch {self.epochs} Loss: {avg_epoch_loss:.2f}")

        return avg_epoch_loss

    def val(self):
        # Instantiate variables we want to track for one epoch of validation
        self.model.eval()
        total_epoch_loss = 0
        num_samples = 0

        # Go through full dataloader per epoch
        for (train, val, test) in self.dl:

            # For validation, we will grab backcast_len of data at the end of train and predict on val
            backcast = train[:, -self.backcast_length :].to(self.device)
            forecast = val.to(self.device)

            # Run a batch of data through the model
            _, forecast_hat = self.model(backcast)

            # Calculate loss on predictions
            loss = self.criterion(forecast_hat.to(self.device), forecast)

            # We will use this to calculate epoch level loss after all batches
            total_epoch_loss += loss.item()
            num_samples += backcast.shape[0]  # Record number of series per batch

        # Calculate loss at epoch level for reporting
        avg_epoch_loss = total_epoch_loss / num_samples

        print(f"******** Val Loss: {avg_epoch_loss:.2f} ********")

        return avg_epoch_loss

    def predict(self):
        # Instantiate variables we want to track for one epoch of validation
        self.model.eval()
        total_epoch_loss = 0
        num_samples = 0

        # Go through full dataloader per epoch
        for (train, val, test) in self.dl:

            # For testing, we will grab backcast_len of data at the end of val and train and predict on test
            # From train, we need backcast - len(val) = backcast - forecast
            backcast_train_part = train[:, -(self.backcast_length - self.forecast_length) :]
            backcast_val_part = val
            backcast = np.hstack((backcast_train_part, backcast_val_part))
            backcast = torch.Tensor(backcast).to(self.device)
            forecast = test.to(self.device)

            # Run a batch of data through the model
            _, forecast_hat = self.model(backcast)

            # Calculate loss on predictions
            loss = self.criterion(forecast_hat.to(self.device), forecast)

            # We will use this to calculate epoch level loss after all batches
            total_epoch_loss += loss.item()
            num_samples += backcast.shape[0]  # Record number of series per batch

        # Calculate loss at epoch level for reporting
        avg_epoch_loss = total_epoch_loss / num_samples
        self.test_loss = avg_epoch_loss

        print(f"******** Test Loss: {avg_epoch_loss:.2f} ********")

        return avg_epoch_loss