import GPy
import numpy as np

from ..emulator import Emulator


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
            print("Ave Test loss: {:.5f}".format(test_loss))

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
