from collections import OrderedDict
from functools import partial

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self, idim, odim, hidden=[64, 64]):
        super(Net, self).__init__()
        self.layer0 = nn.Linear(idim, hidden[0])
        for i in range(1, len(hidden)):
            self.add_module("layer" + str(i), nn.Linear(hidden[i - 1], hidden[i - 1]))
        self.add_module("layer" + str(len(hidden)), nn.Linear(hidden[-1], odim))

    def forward(self, out):
        for i in range(len(self._modules) - 1):
            out = self._modules["layer" + str(i)](out)
            out = F.relu(out)
        out = self._modules["layer" + str(len(self._modules) - 1)](out)
        return out


class Emulator:
    def __init__(
        self,
        test_size=0.2,
        preconditioner=None,
    ):
        self.test_size = test_size
        self.preconditioner = preconditioner  # need to have forward and backward methods for transf and inverse transf

    def train(self, X_train, Y_train):
        raise NotImplementedError

    def test(self, X_test, Y_test):
        raise NotImplementedError

    def build_func(self, param_keys):
        raise NotImplementedError

    def emulate(self, func, samples):
        X = np.vstack([np.array(v) for _, v in samples.items()]).T

        # run all combinations and gather results
        res_list = []
        for i in range(X.shape[0]):  # samples has shape (n_samples, n_parameters)
            res = func(**{k: v for k, v in zip(samples.keys(), X[i, :])})
            res_list.append(res)
        Y = np.array(res_list)

        # optionally apply a transformation
        if self.preconditioner is not None:
            Y = self.preconditioner.forward(Y)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)

        # split into train and test
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=self.test_size
        )

        self.train(X_train, Y_train)
        self.test(X_test, Y_test)
        emulated_func = self.build_func(samples.keys())

        return self.reverse_precondition(emulated_func)

    def reverse_precondition(self, func):
        if self.preconditioner is not None:

            def wrapped_func(*args, **kwargs):
                res = func(*args, **kwargs)
                return self.preconditioner.backward(res)

            return wrapped_func
        return func


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
        **kwargs,
    ):
        super(TorchEmulator, self).__init__(**kwargs)
        self.NNClass = partial(NNClass, **NN_kwargs)
        self.epochs = epochs
        self.optimizer = (
            optimizer
            if optimizer is not None
            else partial(optim.SGD, lr=lr, momentum=momentum, weight_decay=weight_decay)
        )
        self.criterion = criterion if criterion is not None else nn.MSELoss()

    def train(self, X_train, Y_train):
        # now we can train a model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # convert the data to PyTorch tensors and send to GPU if available
        X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)
        Y_train = torch.as_tensor(Y_train, dtype=torch.float32, device=device)

        # create TensorDatasets and DataLoaders
        train_data = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        # build model, optimizer and loss function
        model = self.NNClass(X_train.shape[-1], Y_train.shape[-1]).to(device)
        optimizer = self.optimizer(model.parameters())

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
            pbar.set_postfix({"loss": f"{running_loss/(i+1):.3f}"})

        self.model = model

    def test(self, X_test, Y_test):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def build_func(self, param_keys):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def emulated_func(**kwargs):
            values = [kwargs.get(k, None) for k in param_keys]
            if None in values:
                raise ValueError("Missing argument")
            values = torch.as_tensor(values, dtype=torch.float, device=device)
            res = self.model(values).detach().cpu().numpy()
            return res

        return emulated_func


# a lazy person's emulator interface
class emulate:
    def __init__(
        self,
        samples=None,
        emulator_class=TorchEmulator,
        **kwargs,
    ):
        self.samples = samples
        self.em = emulator_class(**kwargs)

    def __call__(self, func):
        return self.em.emulate(func, self.samples)
