from typing import List

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from tqdm import tqdm

from ..emulator import Emulator


class Net(nn.Module):
    """
    A class representing a simple fully-connected neural network.

    Parameters
    ----------
    odim: int
        Dimensionality of the output.
    hidden: list of int
        List of hidden layer sizes.
    """

    odim: int
    hidden: List[int]

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = nn.sigmoid(x)
        x = nn.Dense(self.odim)(x)
        return x


class JaxEmulator(Emulator):
    def __init__(
        self,
        NNClass=Net,
        epochs=500,
        optimizer=None,
        lr=0.01,
        NN_kwargs={},
        batch_size=128,
        seed=42,
        **kwargs,
    ):
        """
        A class for emulating models using JAX.

        Parameters
        ----------
        NNClass: class, optional (default=Net)
            A class that implements the neural network architecture.
        epochs: int, optional (default=500)
            Number of training epochs.
        optimizer: str or callable, optional (default=None)
            Name of the optimizer or a custom optimizer function.
        lr: float, optional (default=0.01)
            Learning rate for the optimizer.
        NN_kwargs: dict, optional (default={})
            Additional keyword arguments to pass to the neural network constructor.
        batch_size: int, optional (default=128)
            Batch size for training.
        seed: int, optional (default=42)
            Seed for the random number generator.
        **kwargs: dict
            Additional keyword arguments to pass to the `Emulator` constructor.

        """
        super(JaxEmulator, self).__init__(**kwargs)
        self.NNClass = NNClass
        self.epochs = epochs
        self.optimizer = (
            optimizer if optimizer is not None else optax.adam(learning_rate=lr)
        )
        self.batch_size = batch_size
        self.seed = seed
        self.NN_kwargs = NN_kwargs

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
        X_train = jnp.array(X_train)
        Y_train = jnp.array(Y_train)

        # update default NN kwargs if necessary
        NN_kwargs = {"odim": Y_train.shape[1], "hidden": [128, 128]}
        NN_kwargs.update(self.NN_kwargs)
        model = self.NNClass(**NN_kwargs)
        params = model.init(
            jax.random.PRNGKey(self.seed), jnp.ones((1, X_train.shape[1]))
        )
        opt_state = self.optimizer.init(params)

        # minibatch setup
        n_samples = X_train.shape[0]
        n_batches = n_samples // self.batch_size
        if n_batches > 0:
            X_train = jnp.split(X_train[: n_batches * self.batch_size], n_batches)
            Y_train = jnp.split(Y_train[: n_batches * self.batch_size], n_batches)
        else:
            X_train = [X_train]
            Y_train = [Y_train]

        # loss function and gradient for optimization
        @jax.jit
        def compute_loss(params, X, Y):
            loss = optax.l2_loss(model.apply(params, X), Y)
            loss = jnp.mean(loss)
            return loss

        loss_grad = jax.jit(jax.grad(compute_loss))

        @jax.jit
        def train_step(params, opt_state, X_train, Y_train):
            for X, Y in zip(X_train, Y_train):
                grads = loss_grad(params, X, Y)
                updates, opt_state = self.optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
            return params, opt_state

        # training loop
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            params, opt_state = train_step(params, opt_state, X_train, Y_train)
            # use one sample as a proxy for the whole epoch loss
            running_loss = compute_loss(params, X_train[0], Y_train[0])
            pbar.set_postfix({"loss": f"{running_loss:.5f}"})
        self.model = model
        self.params = params

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
        X_test = jnp.array(X_test)
        Y_test = jnp.array(Y_test)
        Y_pred = self.model.apply(self.params, X_test)
        loss = jnp.mean(optax.l2_loss(Y_pred, Y_test))
        print(f"Test loss: {loss:.5f}")
        return loss

    def _predict(self, x):
        """
        Predict the output of the emulator for a given input

        Parameters
        ----------
        x : array-like
            The input data

        Returns
        -------
        The predicted output
        """
        return self.model.apply(self.params, x)

    def _save(self, store):
        """
        Save the emulator model to a file

        Parameters
        ----------
        store: dict
            The dictionary that will be stored to disk

        """
        store["model"] = self.model
        store["params"] = self.params

    def _load(self, store):
        """
        Load the emulator model from a file

        Parameters
        ----------
        store: dict
            The dictionary that will be stored to disk
        """
        self.model = store["model"]
        self.params = store["params"]

    def _build_func(self):
        """
        Build a function that can be used to emulate the original function

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
                values = jnp.array(args)
            else:
                values = [kwargs.get(k, None) for k in self.param_keys]
                if None in values:
                    raise ValueError("Missing argument")
                values = jnp.array(values)
            res = self._predict(values)
            if self.preconditioner is not None:
                res = self.preconditioner.backward(res)
            return res

        # attach a reference to the emulator instance in the function
        emulated_func.emulator = self
        return emulated_func
