from collections import OrderedDict
from functools import partial

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.metrics import mean_squared_error
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
        self.model = None

    def _prepare_data(self, func, samples):
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

        return X_train, X_test, Y_train, Y_test

    def _train(self, X_train, Y_train):
        raise NotImplementedError

    def _test(self, X_test, Y_test):
        raise NotImplementedError

    def _predict(self, x):
        raise NotImplementedError

    def _build_func(self, param_keys):
        def emulated_func(**kwargs):
            values = [kwargs.get(k, None) for k in param_keys]
            if None in values:
                raise ValueError("Missing argument")
            res = self._predict(values)
            if self.preconditioner is not None:
                res = self.preconditioner.backward(res)
            return res

        # attach a reference to the model and emulator instance in the function
        emulated_func.model = self.model
        emulated_func.emulator = self
        return emulated_func

    def emulate(self, func, samples):
        X_train, X_test, Y_train, Y_test = self._prepare_data(func, samples)
        self._train(X_train, Y_train)
        self._test(X_test, Y_test)
        emulated_func = self._build_func(samples.keys())
        return emulated_func


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

    def _train(self, X_train, Y_train):
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

    def _test(self, X_test, Y_test):
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

    def _predict(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        res = self.model(x).detach().cpu().numpy()
        return res


class SklearnEmulator(Emulator):
    def __init__(self, model=None, kernel_kwargs={"n_restarts_optimizer": 5}, **kwargs):
        super(SklearnEmulator, self).__init__(**kwargs)
        if model is None:
            # use a gaussian process model with RKB kernel from sklearn if none is provided
            model = GaussianProcessRegressor(kernel=kernels.RBF(**kernel_kwargs))
        self.model = model

    def _train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def _test(self, X_test, Y_test):
        if len(X_test) > 0:
            test_loss = mean_squared_error(Y_test, self.model.predict(X_test))
            print("Ave Test loss: {:.3f}".format(test_loss))

    def _predict(self, x):
        res = self.model.predict([x])[0]
        return res


# the main emulator interface
class emulate:
    def __init__(
        self,
        samples=None,
        emulator_class=None,
        backend="torch",
        **kwargs,
    ):
        self.samples = samples

        # select the backend
        if backend == "torch":
            self.emulator_class = TorchEmulator
        elif backend == "sklearn":
            self.emulator_class = SklearnEmulator
        else:
            raise ValueError(f"Unknown backend {backend}")

        # if custom emulator class is provided, use it instead
        if emulator_class is not None:
            self.emulator_class = emulator_class

        self.em = self.emulator_class(**kwargs)

    def __call__(self, func):
        return self.em.emulate(func, self.samples)
