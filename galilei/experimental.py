import pickle

import numpy as np


class collect:
    def __init__(self, samples, save="data.pkl", mpi=False):
        """
        collect data from a function

        Parameters
        ----------
        """
        self.samples = samples
        self.mpi = mpi
        self.save = save

        if self.mpi:
            from mpi4py import MPI

            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1

    def __call__(self, func):
        X = np.vstack([np.array(v) for _, v in self.samples.items()]).T
        res_list = []
        for i in range(
            self.rank, X.shape[0], self.size
        ):  # samples has shape (n_samples, n_parameters)
            res = func(**{k: v for k, v in zip(self.samples.keys(), X[i, :])})
            res_list.append(res)
        Y = np.array(res_list)

        if self.mpi:
            Y = self.comm.gather(Y, root=0)
            if self.rank == 0:
                Y = np.vstack(Y)

        if self.rank == 0:
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            if len(Y.shape) == 1:
                Y = Y.reshape(-1, 1)

            # save output
            to_save = {
                "X": X,
                "Y": Y,
            }
            with open(self.save, "wb") as f:
                pickle.dump(to_save, f)
