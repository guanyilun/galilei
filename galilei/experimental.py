import pickle

import numpy as np


class collect:
    def __init__(self, samples, save="data.pkl", mpi=False):
        """
        collect data from a function

        Parameters
        ----------
        samples: dict
            dictionary of samples
        save: str
            path to save the data
        mpi: bool
            use mpi to collect data, default False, only works when
            mpi4py is installed
        """
        self.samples = samples
        self.mpi = mpi
        self.save = save

        if self.mpi:
            try:
                from mpi4py import MPI

                self.comm = MPI.COMM_WORLD
                self.rank = self.comm.Get_rank()
                self.size = self.comm.Get_size()
            except ImportError:
                raise ImportError(
                    "mpi4py is not installed, " "please install it to use mpi"
                )
        else:
            self.comm = None
            self.rank = 0
            self.size = 1

    def __call__(self, func):
        """
        collect data from a function

        Parameters
        ----------
        func: function
            function to collect data from
        """
        X = np.vstack([np.array(v) for _, v in self.samples.items()]).T
        res_list = []
        for i in range(self.rank, X.shape[0], self.size):
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
                "samples": self.samples,
                "X": X,
                "Y": Y,
            }
            with open(self.save, "wb") as f:
                pickle.dump(to_save, f)
