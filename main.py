import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from n_beats.data_loading import M4_Dataset
from n_beats.model import N_Beats
from n_beats.model_trainer import N_Beats_Trainer

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


def main():
    # Job configs
    forecast_length = 18
    backcast_length = 5 * forecast_length
    batch_size = 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 50
    val_every_epochs = 5

    # Model configs
    stacks = {"trend": 2, "seasonality": 8}
    num_blocks = 4
    share_weights_in_stack = True
    hidden_layer_dims = 256

    # Lh sampling is just a multiplier times the forecasting length
    Lh_sampling_len: int = int(2 * forecast_length)

    # Instantiate dataset and data loader
    m4_data_train = M4_Dataset(backcast_length, forecast_length, Lh_sampling_len)
    dataloader = DataLoader(m4_data_train, batch_size=batch_size, shuffle=True)

    # Instantiate model with correct configs
    n_beats_model = N_Beats(
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        device=device,
        stacks=stacks,
        num_blocks=num_blocks,
        share_weights_in_stack=share_weights_in_stack,
        hidden_layer_dims=hidden_layer_dims,
    )

    # Instantiate model trainer object
    trainer = N_Beats_Trainer(
        model=n_beats_model,
        dl=dataloader,
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        Lh_sampling_len=Lh_sampling_len,
        device=device,
    )

    print("--- Training ---")
    best_loss = 1e8
    best_model = None
    for epoch in range(num_epochs):
        _ = trainer.train()
        if epoch % val_every_epochs == 0:
            val_loss = trainer.val()
            if val_loss < best_loss:  # If loss is better, save model and update best
                best_model = copy.deepcopy(trainer.model)
                best_loss = val_loss

    # Reset to best model and get test loss
    trainer.model = best_model
    _ = trainer.predict()


if __name__ == "__main__":
    main()