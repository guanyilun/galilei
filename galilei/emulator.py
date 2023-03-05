import pickle

import numpy as np
from sklearn.model_selection import train_test_split


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

    def _prepare_data(self, func, samples, precomputed=None, collection=None):
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
        collection: str
            Path to a pickle file containing precomputed collection of data. If
            provided, the function will not be evaluated. It takes precedence over
            precomputed which contains only the function outputs. The collection
            is expected to be the saved product of the `collect` function.

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
        # load everything from collection if it's given
        if collection is not None:
            with open(collection, "rb") as f:
                collection_ = pickle.load(f)
            # use the loaded values instead of computing
            X = collection_["X"]
            Y = collection_["Y"]
            samples = collection_["samples"]
        else:
            X = np.vstack([np.array(v) for _, v in samples.items()]).T

            # run all combinations and gather results
            if precomputed is None:
                res_list = []
                for i in range(
                    X.shape[0]
                ):  # samples has shape (n_samples, n_parameters)
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

        def emulated_func(*args, **kwargs):
            # check if the function has been called with the correct number of arguments
            if len(args) > 0:
                if len(args) != len(self.param_keys):
                    raise ValueError("Incorrect number of arguments")
                values = np.array(args)
            else:
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

    def emulate(
        self, func, samples, pretrained=None, precomputed=None, collection=None
    ):
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
            evaluated at the parameter values in `samples`

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
            func, samples, precomputed=precomputed, collection=collection
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

        with open(filename, "wb") as f:
            pickle.dump(store, f)

    def load(self, filename):
        """
        Load the emulator from a file

        Parameters
        ----------
        filename : str
            Filename to load the emulator from
        """
        with open(filename, "rb") as f:
            store = pickle.load(f)

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


# the main emulator interface
class emulate:
    def __init__(
        self,
        samples=None,
        emulator_class=None,
        backend="jax",
        save=None,
        load=None,
        precomputed=None,
        collection=None,
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
            The backend library to use for the emulator. Currently supported options are "torch", "sklearn", "gpy", "jax". Default is "jax".
        save : str, optional
            The path to the file to save the emulator to. If None, the emulator will not be saved. Default is None.
        load : str, optional
            The path to the file to load the emulator from. If None, the emulator will not be loaded. Default is None.
        precomputed : np.ndarray, optional
            precomputed outputs of the function. If provided, the function will not be evaluated at the parameter values
            in `samples`.
        collection : str, optional
            A dictionary of precomputed collection of the function. If provided, the function will not be evaluated
            at the parameter values in `samples`, and `X`, `Y`, `samples` stored in `collection` dictionary will be used
            instead. This works with collection saved by the `collect` decorator.
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
        self.collection = collection

        # select the backend
        if backend == "torch":
            try:
                from .backends.torch import TorchEmulator
            except ImportError:
                raise ImportError(
                    "PyTorch is not installed. Please install it to use the PyTorch backend."
                )
            self.emulator_class = TorchEmulator
        elif backend == "sklearn":
            from .backends.sklearn import SklearnEmulator  # installed by default

            self.emulator_class = SklearnEmulator
        elif backend == "gpy":
            try:
                from .backends.GPy import GPyEmulator
            except ImportError:
                raise ImportError(
                    "GPy is not installed. Please install it to use the GPy backend."
                )
            self.emulator_class = GPyEmulator
        elif backend == "jax":
            try:
                from .backends.jax import JaxEmulator
            except ImportError:
                raise ImportError(
                    "Jax is not installed. Please install jax, optax, and flax to use the Jax backend."
                )
            self.emulator_class = JaxEmulator
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
            func,
            self.samples,
            pretrained=self.load,
            precomputed=self.precomputed,
            collection=self.collection,
        )
        if self.save is not None:
            self.em.save(self.save)
        return emulated
