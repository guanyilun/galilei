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
    def __init__(self, idim, odim, hidden=[64, 64], activation=F.relu):
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


class Emulator:
    def __init__(
        self,
        test_size=0.2,
        preconditioner=None,
    ):
        """
        Base class for emulators

        Parameters
        ----------
        test_size : float
            Fraction of samples to use for testing
        preconditioner : object
            Implement forward and backward methods for transforming function results before
            training and after prediction
        """
        self.test_size = test_size
        self.preconditioner = preconditioner  # need to have forward and backward methods for transf and inverse transf
        self.model = None

        # input, output dimensions, and parameter names
        self.idim = None
        self.odim = None
        self.param_keys = None  # TODO: retrieve from function signature instead

        self.X_test = None
        self.Y_test = None

    def _prepare_data(self, func, samples, precomputed=None):
        """
        Prepare data for training and testing

        Parameters
        ----------
        func : callable
            Function to be emulated
        samples : dict
            Dictionary of parameter names and values
        precomputed: array-like
            Precomputed outputs of the function. If provided, the function will not be
            evaluated.

        Returns
        -------
        X_train : np.ndarray
            Training input
        X_test : np.ndarray
            Test input
        Y_train : np.ndarray
            Training output
        Y_test : np.ndarray
            Test output
        """
        X = np.vstack([np.array(v) for _, v in samples.items()]).T

        # run all combinations and gather results
        if precomputed is None:
            res_list = []
            for i in range(X.shape[0]):  # samples has shape (n_samples, n_parameters)
                res = func(**{k: v for k, v in zip(samples.keys(), X[i, :])})
                res_list.append(res)
            Y = np.array(res_list)
        else:
            Y = np.array(precomputed)

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

        # save input, output dimensions, and parameter names
        self.idim = X_train.shape[1]
        self.odim = Y_train.shape[1]
        self.param_keys = list(
            samples.keys()
        )  # TODO: retrieve from function signature instead

        # save a reference to test data for future use
        self.X_test = X_test
        self.Y_test = Y_test

        return X_train, X_test, Y_train, Y_test

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
        raise NotImplementedError

    def _test(self, X_test, Y_test):
        """
        Test the emulator

        Parameters
        ----------
        X_test : np.ndarray
            Test input
        Y_test : np.ndarray
            Test output
        """
        raise NotImplementedError

    def _predict(self, x):
        """
        Predict the output for a given input

        Parameters
        ----------
        x : np.ndarray
            Input parameters for emulated function

        Returns
        -------
        y : np.ndarray
            Output of emulated function
        """
        raise NotImplementedError

    def _build_func(self):
        """
        Build a function that can be used to emulate the original function

        Parameters
        ----------
        param_keys : list of str
            List of parameter names

        Returns
        -------
        emulated_func : callable
            Function that can be used to emulate the original function. The function has a
            reference to the emulator instance in the attribute `emulator`.
        """

        def emulated_func(**kwargs):
            values = [kwargs.get(k, None) for k in self.param_keys]
            if None in values:
                raise ValueError("Missing argument")
            res = self._predict(values)
            if self.preconditioner is not None:
                res = self.preconditioner.backward(res)
            return res

        # attach a reference to the emulator instance in the function
        emulated_func.emulator = self
        return emulated_func

    def emulate(self, func, samples, pretrained=None, precomputed=None):
        """
        Emulate a function

        Parameters
        ----------
        func : callable
            Function to be emulated
        samples : dict
            Dictionary of parameter names and values, where the values are lists of
            parameter values to evaluate the function at to build the emulator
        precomputed: array-like
            Precomputed outputs of the function. If provided, the function will not be
            evaluated at the parameter values in `samples`, and the outputs in
            `precomputed_outputs` will be used instead. This is useful if the function
            is expensive to evaluate, and the outputs have already been computed.


        Returns
        -------
        emulated_func : callable
            Function that can be used to emulate the original function. The function has a
            reference to the emulator instance in the attribute `emulator`.
        """
        if pretrained is not None:
            self.load(pretrained)
            return self._build_func()
        X_train, X_test, Y_train, Y_test = self._prepare_data(
            func, samples, precomputed=precomputed
        )
        self._train(X_train, Y_train)
        self._test(X_test, Y_test)
        emulated_func = self._build_func()
        return emulated_func

    def _save(self, store):
        """
        Save the emulator to a datastore

        Parameters
        ----------
        store : dict
            DataStore to save the emulator to
        """
        raise NotImplementedError

    def _load(self, store):
        """
        Load the emulator from a datastore

        Parameters
        ----------
        store : dict
            DataStore to load the emulator from
        """
        raise NotImplementedError

    def save(self, filename):
        """
        Save the emulator to a file

        Parameters
        ----------
        filename : str
            Filename to save the emulator to
        """
        # always save the input and output dimensions
        store = {
            "in": self.idim,
            "out": self.odim,
            "param_keys": self.param_keys,
        }
        self._save(store)
        if self.preconditioner is not None:
            if not hasattr(self.preconditioner, "_save"):
                raise ValueError("Cannot save a preconditioner without a _save method")
            store["preconditioner"] = {}
            self.preconditioner._save(store["preconditioner"])
        torch.save(store, filename)  # shorter to write than pickle

    def load(self, filename):
        """
        Load the emulator from a file

        Parameters
        ----------
        filename : str
            Filename to load the emulator from
        """
        store = torch.load(filename)

        self.idim = store["in"]
        self.odim = store["out"]
        self.param_keys = store["param_keys"]

        if "preconditioner" in store:
            if self.preconditioner is None:
                raise ValueError(
                    "Cannot load a preconditioner into an emulator without one"
                )
            self.preconditioner._load(store["preconditioner"])
        self._load(store)


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
            pbar.set_postfix({"loss": f"{running_loss/(i+1):.4f}"})
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.NNClass(store["in"], store["out"])
        self.model.load_state_dict(store["model"])
        self.model.to(device)


class SklearnEmulator(Emulator):
    def __init__(
        self,
        model=None,
        kernel_kwargs={},
        model_kwargs={"n_restarts_optimizer": 10},
        **kwargs,
    ):
        """
        A class used to emulate data using a machine learning model based on the scikit-learn library.

        Parameters
        ----------
        model : sklearn.base.TransformerMixin, optional
            The machine learning model to use. If None, uses a Gaussian Process model with an RBF kernel from scikit-learn with default parameters.
        kernel_kwargs : dict, optional
            The keyword arguments to pass to the RBF kernel constructor if a Gaussian Process model is used. Default is an empty dictionary.
        model_kwargs : dict, optional
            The keyword arguments to pass to the machine learning model constructor if one is provided. Default sets n_restarts_optimizer to 5.
        **kwargs
            Additional keyword arguments to pass to the base Emulator class constructor.

        Attributes
        ----------
        model : sklearn.base.TransformerMixin
            The machine learning model used for emulation.
        """

        super(SklearnEmulator, self).__init__(**kwargs)
        if model is None:
            # use a gaussian process model with RKB kernel from sklearn if none is provided
            model = GaussianProcessRegressor(
                kernel=kernels.RBF(**kernel_kwargs), **model_kwargs
            )
        self.model = model

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
        self.model.fit(X_train, Y_train)

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
        if len(X_test) > 0:
            test_loss = mean_squared_error(Y_test, self.model.predict(X_test))
            print("Ave Test loss: {:.3f}".format(test_loss))

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
        res = self.model.predict([x])[0]
        return res

    def _save(self, store):
        """
        Save the emulator model to a file

        Parameters
        ----------
        store: dict
            The dictionary that will be stored to disk

        """
        store["model"] = self.model

    def _load(self, store):
        """
        Load the emulator model from a file

        Parameters
        ----------
        store: dict
            The dictionary that will be stored to disk
        """
        self.model = store["model"]


class GPyEmulator(Emulator):
    def __init__(
        self,
        kernel_kwargs={},
        model_kwargs={},
        **kwargs,
    ):
        """
        A class used to emulate data using a machine learning model based on the GPy library.

        Parameters
        ----------
        kernel_kwargs : dict, optional
            The keyword arguments to pass to the RBF kernel constructor if a Gaussian Process model is used. Default is an empty dictionary.
        model_kwargs : dict, optional
            The keyword arguments to pass to the machine learning model constructor if one is provided. Default is an empty dictionary.
        **kwargs
            Additional keyword arguments to pass to the base Emulator class constructor.

        Attributes
        ----------
        model : GPy.models.GPRegression
            The machine learning model used for emulation.
        """
        super(GPyEmulator, self).__init__(**kwargs)
        self.kernel_kwargs = kernel_kwargs
        self.model_kwargs = model_kwargs
        self.model = None

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
        import GPy

        kernel = GPy.kern.RBF(input_dim=X_train.shape[-1], **self.kernel_kwargs)
        model = GPy.models.GPRegression(X_train, Y_train, kernel, **self.model_kwargs)
        model.optimize()
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
        if len(X_test) > 0:
            test_loss = np.mean((Y_test - self.model.predict(X_test)[0]) ** 2)
            print("Ave Test loss: {:.3f}".format(test_loss))

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
        res = self.model.predict(np.array([x]))
        return res[0][0]

    def _save(self, store):
        """
        Save the emulator model to a file

        Parameters
        ----------
        store: dict
            The dictionary that will be stored to disk
        """
        store["model"] = self.model.to_dict()

    def _load(self, store):
        """
        Load the emulator model from a file

        Parameters
        ----------
        store: dict
            The dictionary that will be stored to disk
        """
        import GPy

        self.model = GPy.models.GPRegression.from_dict(store["model"])


# the main emulator interface
class emulate:
    def __init__(
        self,
        samples=None,
        emulator_class=None,
        backend="torch",
        save=None,
        load=None,
        precomputed=None,
        **kwargs,
    ):
        """
        A class used to emulate a function using a machine learning model.

        Parameters
        ----------
        samples : np.ndarray or pd.DataFrame, optional
            The samples to use for training the machine learning model. If None, the emulator will not be trained and cannot be used for emulation.
        emulator_class : type, optional
            The class of the emulator to use. If None, the default emulator for the specified backend will be used.
        backend : str, optional
            The backend library to use for the emulator. Currently supported options are "torch" and "sklearn". Default is "torch".
        save : str, optional
            The path to the file to save the emulator to. If None, the emulator will not be saved. Default is None.
        load : str, optional
            The path to the file to load the emulator from. If None, the emulator will not be loaded. Default is None.
        **kwargs
            Additional keyword arguments to pass to the emulator constructor.

        Attributes
        ----------
        samples : np.ndarray or pd.DataFrame
            The samples used for training the machine learning model.
        emulator_class : type
            The class of the emulator used for emulation.
        em : emulator_class
            An instance of the emulator used for emulation.
        save : str
            The path to the file to save the emulator to.
        load : str
            The path to the file to load the emulator from.

        Methods
        -------
        __call__(func)
            Call the emulator to emulate the specified function with the specified input samples.
        """

        self.samples = samples
        self.precomputed = precomputed
        self.save = save
        self.load = load

        # select the backend
        if backend == "torch":
            self.emulator_class = TorchEmulator
        elif backend == "sklearn":
            self.emulator_class = SklearnEmulator
        elif backend == "gpy":
            try:
                import GPy
            except ImportError:
                raise ImportError(
                    "GPy is not installed. Please install it to use the GPy backend."
                )
            self.emulator_class = GPyEmulator
        else:
            raise ValueError(f"Unknown backend {backend}")

        # if custom emulator class is provided, use it instead
        if emulator_class is not None:
            self.emulator_class = emulator_class

        self.em = self.emulator_class(**kwargs)

    def __call__(self, func):
        """
        Call the emulator to emulate the specified function with the specified input samples.

        Parameters
        ----------
        func : callable
            The function to emulate.

        Returns
        -------
        callable
            The emulator function.
        """
        emulated = self.em.emulate(
            func, self.samples, pretrained=self.load, precomputed=self.precomputed
        )
        if self.save is not None:
            self.em.save(self.save)
        return emulated
