
   
import numpy as np
import torch
from torch import nn
from typing import Dict


class N_Beats(nn.Module):
    def __init__(
        self,
        backcast_length: int,
        forecast_length: int,
        device: str,
        stacks: Dict[str, int],
        num_blocks: int,
        share_weights_in_stack: bool,
        hidden_layer_dims: int,
    ):
        super(N_Beats, self).__init__()

        # Job configs
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.device = device

        # Model params
        self.stacks = stacks
        self.num_blocks = num_blocks
        self.share_weights_in_stack = share_weights_in_stack
        self.hidden_layer_dims = hidden_layer_dims

        # Instantiate a param list that PyTorch's optimizer needs
        self.parameters = []

        # A model is made up of multiple stacks, and each stack is made up of blocks
        # We loop through each stack and create all necessary blocks within it
        self.stacks_list = []
        for stack_type, basis_dims in self.stacks.items():
            self.stacks_list.append(self.create_stack(stack_type=stack_type, basis_dims=basis_dims))

        # Convert the param list into a form PyTorch can use
        self.parameters = nn.ParameterList(self.parameters)

    def create_stack(self, stack_type: str, basis_dims: int):
        """Helper function that creates all blocks within a specific stack and returns a list of blocks
        Args:
            stack_type (str): What type of stack we are creating (can be "trend" or "seasonality")
            basis_dims (int): Size of theta dimension inside each block, used for fitting coefficients to functional forms
        Returns:
            block_list (List[Block]): A stack of blocks that comprise one part of the N-Beats model
        """
        # Create the appropriate number of blocks within each stack
        block_list = []

        for block_id in range(self.num_blocks):
            # Create the appropriate block for each stack type
            # If we are sharing weights, just append the same block over and over
            if self.share_weights_in_stack and block_id != 0:
                block = block_list[-1]
            else:  # Otherwise, create a new block with appropriate configurations
                block = Block(
                    forecast_length=self.forecast_length,
                    backcast_length=self.backcast_length,
                    device=self.device,
                    stack_type=stack_type,
                    basis_dims=basis_dims,
                    hidden_layer_dims=self.hidden_layer_dims,
                )

            # Need to add the parameters of a certain block to this list
            # This is so the optimizer knows the model params during training
            self.parameters.extend(block.parameters())

            # Add created or old block to the list
            block_list.append(block)

        return block_list

    def forward(self, backcast: torch.Tensor):
        # Create empty array of forecasts that is num series * forecast length
        # Also cast backcast to float type, and move both to proper device
        forecast = torch.zeros(backcast.shape[0], self.forecast_length, dtype=torch.float).to(
            self.device
        )
        backcast = backcast.float().to(self.device)

        # Create empty array of predicted backcast so we can use it for verificaiton purposes
        backcast_hat = torch.zeros(backcast.shape[0], self.backcast_length, dtype=torch.float).to(
            self.device
        )

        # Need to run given backcast through each stack
        for stack in self.stacks_list:
            # Within each stack, run the backcast through each block
            for block in stack:
                # For each block, given a backcast, we will produce a residual backcast and a forecast
                # The backcast is subtractive (output of model is removed from input)
                # The forecast is addative (output of the model is added to final predicted output)
                backcast_predicted, forecast_predicted = block(backcast)

                backcast = backcast - backcast_predicted
                forecast = forecast + forecast_predicted
                backcast_hat = backcast_hat + backcast_predicted

        return backcast_hat, forecast


class Block(nn.Module):
    def __init__(
        self, forecast_length, backcast_length, device, stack_type, basis_dims, hidden_layer_dims
    ):
        super(Block, self).__init__()

        # Read job and model configs in
        self.hidden_layer_dims = hidden_layer_dims
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.device = device
        self.model_type = stack_type

        # This is the number of polynomial dims if using trend or number of sin and cos in fourier fit
        self.basis_layer_dim = basis_dims

        # Instantiate FF model of four linear and ReLU layers
        self.ff_model = nn.Sequential(
            nn.Linear(self.backcast_length, self.hidden_layer_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_dims, self.hidden_layer_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_dims, self.hidden_layer_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_dims, self.hidden_layer_dims),
            nn.ReLU(),
        )

        # Create the array of backcast and forecast time intervals
        self.backcast_linspace, self.forecast_linspace = self.linspace()

        # We will use the same linear projection of ff_model to both forecast and backcast
        self.ff_model_to_theta = nn.Linear(self.hidden_layer_dims, self.basis_layer_dim, bias=False)

    def forward(self, x):
        """The forward pass within each block goes as follows:
        First, we take an input backcast to a block and run it through a feed forward network.
        This will embed the input backcast to self.hidden_layer_dims.
        Next, we will linearly project the output of the feed forward to self.basis_layer_dim.
        Finally, depending on what the type of the stack is, we will either run the self.basis_layer_dim
        as the coefficients to a Fourier series or as the coefficients to a polynomial fit.
        We will use the same linear projection and Fourier/polynomial fit to calculate a forecast
        and a backcast for a given input backcast.
        Args:
            x (List[float]): Input backcast to a block
        Raises:
            ValueError: A value error is raised if the stack type is not trend or seasonality
        Returns:
            backcast (List[float]): Prediction of the given backcast using given constraints on the block
            forecast (List[float]): Prediction of the forecast using given constraints on the block
        """
        # First, run input backcast through a feed forward
        ff_output = self.ff_model(x)
        theta = self.ff_model_to_theta(ff_output)

        if self.model_type.upper() == "TREND":
            backcast = self.trend_model(theta, self.backcast_linspace)
            forecast = self.trend_model(theta, self.forecast_linspace)
        elif self.model_type.upper() == "SEASONALITY":
            backcast = self.seasonality_model(theta, self.backcast_linspace)
            forecast = self.seasonality_model(theta, self.forecast_linspace)
        else:
            raise ValueError("Not a valid model type!")

        return backcast, forecast

    def seasonality_model(self, thetas, t_range):
        """For the seasonality model, we want to fit a Fourier series to our time series.
        From the n-beats paper:
        Typical characteristic of seasonality is that it is a regular, cyclical, recurring
        fluctuation. Therefore, to model seasonality, we propose to constrain [our model]... A natural choice for the
        basis to model periodic function is the Fourier series.
        Args:
            thetas (List[float]): Output of the feed forward part of a block. This can be interpreted as a
                low dimensional representation of the input data series, and is compressed down to a
                Fourier series' coefficients. Matrix size is (batch_size x degree of Fourier series)
            t_range (List[float]): Range of backcast or forecast time points to "predict" on. This will have
                dimensions of backcast or forecast horizion length. Matrix size is (backcast/forecast len)
        Returns:
            fourier_fit (List[float]): A seasonality fit on the input low dimensional backcast or forecast series
        """
        # How many waves we want in each Fourier series
        num_sin_and_cos = self.basis_layer_dim
        if num_sin_and_cos % 2 != 0:
            raise ValueError("Not supporting uneven number of sin and cos waves right now")

        # For the seasonality block, we will fit an equal number of sin and cos waves
        num_sin_functions = num_sin_and_cos // 2
        num_cos_functions = num_sin_functions

        # Fit sin and cos waves with different periodicities
        # Both tensors have shape (num_cos/sin_functions, backcast/forecast len)
        cos_funcs = torch.tensor(
            [np.cos(2 * np.pi * i * t_range) for i in range(num_sin_functions)]
        ).float()
        sin_funcs = torch.tensor(
            [np.sin(2 * np.pi * i * t_range) for i in range(num_cos_functions)]
        ).float()

        # Concat them together so both are in the same matrix
        # Matrix has shape (num_sin_and_cos, backcast/forecast len)
        S = torch.cat([cos_funcs, sin_funcs]).to(self.device)

        # Do mat mult to get polynomial fit of data
        # Matrix size is (batch_size, backcast/forecast len)
        fourier_fit = thetas.mm(S)
        return fourier_fit

    def trend_model(self, thetas, t_range):
        """For the trend model, we want to fit a low degree polynomial (polynomial_degree <= 3 ish).
        From the n-beats paper:
        A typical characteristic of trend is that most of the time it is a monotonic function, or
        at least a slowly varying function. In order to mimic this behaviour we propose to constrain g
        to be a polynomial of small degree p, a function slowly varying across forecast window:
        Args:
            thetas (List[float]): Output of the feed forward part of a block. This can be interpreted as a
                low dimensional representation of the input data series, and is compressed down to a
                low degree polynomial size. Matrix size is (batch_size x polynomical_degree)
            t_range (List[float]): Range of backcast or forecast time points to "predict" on. This will have
                dimensions of backcast or forecast horizion length. Matrix size is (backcast/forecast len)
        Returns:
            poly_fit (List[float]): A trend fit on the input low dimensional backcast or forecast series
        """
        # What degree polynomial we want to fit
        polynomial_degree = self.basis_layer_dim

        # Calculate a matrix of powers of t_range
        # Since we are doing this for every power of the polynomial, each row is one
        # of the terms for the polynomial of a certain series
        # Matrix size is (polynomial_degree x backcast/forecast length)
        T = torch.tensor([t_range ** i for i in range(polynomial_degree)]).float()

        # Do mat mult to get polynomial fit of data
        # Matrix size is (batch_size, backcast/forecast len)
        poly_fit = thetas.mm(T.to(self.device))

        return poly_fit

    def linspace(self):
        lin_space = np.linspace(
            -self.backcast_length, self.forecast_length, self.backcast_length + self.forecast_length
        )
        b_ls = lin_space[: self.backcast_length]
        f_ls = lin_space[self.backcast_length :]
        return b_ls, f_ls

