from functools import partial

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..emulator import Emulator


class Net(nn.Module):
    def __init__(self, idim, odim, hidden=[128, 128], activation=torch.sigmoid):
        """
        Simple fully connected neural network with ReLU activation

        Parameters
        ----------
        idim : int
            Input dimension
        odim : int
            Output dimension
        hidden : list of int
            List of hidden layer sizes
        """
        super(Net, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(idim, hidden[0]))
        for i in range(1, len(hidden)):
            self.layers.append(nn.Linear(hidden[i - 1], hidden[i]))
        self.layers.append(nn.Linear(hidden[-1], odim))

    def forward(self, out):
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)
        return out


class TorchEmulator(Emulator):
    def __init__(
        self,
        NNClass=Net,
        epochs=500,
        optimizer=None,
        criterion=None,
        lr=0.01,
        weight_decay=0,
        momentum=0.9,
        NN_kwargs={},
        scheduler=None,
        batch_size=32,
        device=None,
        **kwargs,
    ):
        """
        A class used to emulate data using a neural network based on the PyTorch library.

        Parameters
        ----------
        NNClass : nn.Module, optional
            The neural network class to use. Default is Net.
        epochs : int, optional
            The number of epochs to train the neural network. Default is 500.
        optimizer : torch.optim.Optimizer, optional
            The optimizer to use for training the neural network. If None, uses Stochastic Gradient Descent (SGD) optimizer with default parameters.
        criterion : torch.nn.modules.loss._Loss, optional
            The loss function to use for training the neural network. If None, uses Mean Squared Error (MSE) loss function.
        lr : float, optional
            The learning rate for the optimizer. Default is 0.01.
        weight_decay : float, optional
            The weight decay (L2 penalty) for the optimizer. Default is 0.
        momentum : float, optional
            The momentum factor for the optimizer. Default is 0.9.
        NN_kwargs : dict, optional
            The keyword arguments to pass to the neural network class constructor. Default is an empty dictionary.
        **kwargs
            Additional keyword arguments to pass to the base Emulator class constructor.

        Attributes
        ----------
        NNClass : callable
            A callable that returns an instance of the neural network class with the given keyword arguments.
        epochs : int
            The number of epochs to train the neural network.
        optimizer : torch.optim.Optimizer
            The optimizer to use for training the neural network.
        criterion : torch.nn.modules.loss._Loss
            The loss function to use for training the neural network.
        """
        super(TorchEmulator, self).__init__(**kwargs)
        self.NNClass = partial(NNClass, **NN_kwargs)
        self.epochs = epochs
        self.optimizer = (
            optimizer
            if optimizer is not None
            else partial(optim.SGD, lr=lr, momentum=momentum, weight_decay=weight_decay)
        )
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self.scheduler = scheduler
        self.batch_size = batch_size
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def _train(self, X_train, Y_train):
        """
        Train the emulator

        Parameters
        ----------
        X_train : np.ndarray
            Training input
        Y_train : np.ndarray
            Training output
        """
        device = self.device
        # convert the data to PyTorch tensors and send to GPU if available
        X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)
        Y_train = torch.as_tensor(Y_train, dtype=torch.float32, device=device)

        # create TensorDatasets and DataLoaders
        train_data = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        # build model, optimizer and loss function
        model = self.NNClass(X_train.shape[-1], Y_train.shape[-1]).to(device)
        optimizer = self.optimizer(model.parameters())
        scheduler = self.scheduler(optimizer) if self.scheduler is not None else None

        # train the model
        print("Training emulator...")
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            running_loss = 0.0
            for i, (inputs, Y) in enumerate(train_loader):
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = self.criterion(outputs, Y)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            pbar.set_postfix({"loss": f"{running_loss/(i+1):.5f}"})
            if scheduler is not None:
                scheduler.step()

        self.model = model

    def _test(self, X_test, Y_test):
        """
        Test the emulator on a test set

        Parameters
        ----------
        X_test : array-like
            The test set input data
        Y_test : array-like
            The test set output data
        """
        device = self.device
        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
        Y_test = torch.as_tensor(Y_test, dtype=torch.float32, device=device)
        # Test the model
        if len(X_test) > 0:
            with torch.no_grad():
                test_loss = 0.0
                for i, data in enumerate(zip(X_test, Y_test)):
                    inputs, Y = data
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, Y)
                    test_loss += loss.item()
                avg_test_loss = test_loss / (i + 1)
            print("Ave Test loss: {:.3f}".format(avg_test_loss))

    def _predict(self, x):
        """
        Predict the output of the emulator for a given input

        Parameters
        ----------
        x : array-like
            The input data

        Returns
        -------
        np.ndarray
            The predicted output
        """
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        res = self.model(x).detach().cpu().numpy()
        return res

    def _save(self, store):
        """
        Save the emulator model to a file

        Parameters
        ----------
        store: dict
            The dictionary that will be stored to disk

        """
        store["model"] = self.model.state_dict()

    def _load(self, store):
        """
        Load the emulator model from a file

        Parameters
        ----------
        store: dict
            The dictionary that will be stored to disk
        """
        self.model = self.NNClass(store["in"], store["out"])
        self.model.load_state_dict(store["model"])
        self.model.to(self.device)
