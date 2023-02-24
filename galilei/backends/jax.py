from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from tqdm import tqdm

from ..emulator import Emulator


class Net(nn.Module):
    idim: int
    odim: int
    hidden = [128, 128]

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
        super(JaxEmulator, self).__init__(**kwargs)
        self.NNClass = partial(NNClass, **NN_kwargs)
        self.epochs = epochs
        self.optimizer = (
            optimizer if optimizer is not None else optax.adam(learning_rate=lr)
        )
        self.batch_size = batch_size
        self.seed = seed

    def _train(self, X_train, Y_train):
        X_train = jnp.array(X_train)
        Y_train = jnp.array(Y_train)

        model = self.NNClass(idim=X_train.shape[1], odim=Y_train.shape[1])
        model = Net(idim=X_train.shape[1], odim=Y_train.shape[1])
        params = model.init(
            jax.random.PRNGKey(self.seed), jnp.ones((1, X_train.shape[1]))
        )
        opt_state = self.optimizer.init(params)

        # minibatch setup
        n_samples = X_train.shape[0]
        n_batches = n_samples // self.batch_size
        X_train = jnp.split(X_train[: n_batches * self.batch_size], n_batches)
        Y_train = jnp.split(Y_train[: n_batches * self.batch_size], n_batches)

        # loss function and gradient for optimization
        def compute_loss(params, X, Y):
            loss = optax.l2_loss(model.apply(params, X), Y)
            loss = jnp.mean(loss)
            return loss

        loss_grad = jax.grad(compute_loss)

        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            for X, Y in zip(X_train, Y_train):  # minibatch
                grads = loss_grad(params, X, Y)
                updates, opt_state = self.optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
            # use one sample as a proxy for the whole epoch loss
            running_loss = compute_loss(params, X_train[0], Y_train[0])
            pbar.set_postfix({"loss": f"{running_loss:.5f}"})
        self.model = model
        self.params = params

    def _test(self, X_test, Y_test):
        X_test = jnp.array(X_test)
        Y_test = jnp.array(Y_test)
        Y_pred = self.model.apply(self.params, X_test)
        return jnp.mean(optax.l2_loss(Y_pred, Y_test))

    def _predict(self, x):
        return self.model.apply(self.params, x)

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
