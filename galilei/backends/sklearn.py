from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.metrics import mean_squared_error

from ..emulator import Emulator


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
