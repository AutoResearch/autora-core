import sys
from copy import deepcopy
from random import randint, random

import numpy as np
import pandas as pd

# from mcmc import *
from mcmc import OPS, Tree
from numpy import exp


class Parallel:
    """The Parallel class for parallel tempering."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        Ts,
        ops=OPS,
        variables=["x"],
        parameters=["a"],
        max_size=50,
        prior_par={},
        x=None,
        y=None,
    ):
        # All trees are initialized to the same tree but with different BT
        Ts.sort()
        self.Ts = [str(T) for T in Ts]
        self.trees = {
            "1": Tree(
                ops=ops,
                variables=deepcopy(variables),
                parameters=deepcopy(parameters),
                prior_par=deepcopy(prior_par),
                x=x,
                y=y,
                max_size=max_size,
                BT=1,
            )
        }
        self.t1 = self.trees["1"]
        for BT in [T for T in self.Ts if T != 1]:
            treetmp = Tree(
                ops=ops,
                variables=deepcopy(variables),
                parameters=deepcopy(parameters),
                prior_par=deepcopy(prior_par),
                x=x,
                y=y,
                root_value=str(self.t1),
                max_size=max_size,
                BT=float(BT),
            )
            self.trees[BT] = treetmp
            # Share fitted parameters and representative with other trees
            self.trees[BT].fit_par = self.t1.fit_par
            self.trees[BT].representative = self.t1.representative
        return

    # -------------------------------------------------------------------------
    def mcmc_step(self, verbose=False, p_rr=0.05, p_long=0.45):
        """Perform a MCMC step in each of the trees."""
        # Loop over all trees
        for T, tree in list(self.trees.items()):
            # MCMC step
            tree.mcmc_step(verbose=verbose, p_rr=p_rr, p_long=p_long)
        self.t1 = self.trees["1"]
        # Done
        return

    # -------------------------------------------------------------------------
    def tree_swap(self):
        # Choose Ts to swap
        nT1 = randint(0, len(self.Ts) - 2)
        nT2 = nT1 + 1
        t1 = self.trees[self.Ts[nT1]]
        t2 = self.trees[self.Ts[nT2]]
        # The temperatures and energies
        BT1, BT2 = t1.BT, t2.BT
        # EB1, EB2, EP1, EP2 = t1.EB, t2.EB, t1.EP, t2.EP
        EB1, EB2 = t1.EB, t2.EB
        # The energy change
        DeltaE = np.float(EB1) * (1.0 / BT2 - 1.0 / BT1) + np.float(EB2) * (
            1.0 / BT1 - 1.0 / BT2
        )
        if DeltaE > 0:
            paccept = exp(-DeltaE)
        else:
            paccept = 1.0
        # Accept/reject change
        if random() < paccept:
            self.trees[self.Ts[nT1]] = t2
            self.trees[self.Ts[nT2]] = t1
            t1.BT = BT2
            t2.BT = BT1
            self.t1 = self.trees["1"]
            return self.Ts[nT1], self.Ts[nT2]
        else:
            return None, None

    # -------------------------------------------------------------------------
    def anneal(self, n=1000, factor=5):
        # Heat up
        for t in list(self.trees.values()):
            t.BT *= factor
        for kk in range(n):
            print(
                "# Annealing heating at %g: %d / %d" % (self.trees["1"].BT, kk, n),
                file=sys.stderr,
            )
            self.mcmc_step()
            self.tree_swap()
        # Cool down (return to original temperatures)
        for BT, t in list(self.trees.items()):
            t.BT = float(BT)
        for kk in range(2 * n):
            print(
                "# Annealing cooling at %g: %d / %d" % (self.trees["1"].BT, kk, 2 * n),
                file=sys.stderr,
            )
            self.mcmc_step()
            self.tree_swap()
        # Done
        return

    # -------------------------------------------------------------------------
    def trace_predict(
        self,
        x,
        burnin=5000,
        thin=100,
        samples=10000,
        anneal=100,
        annealf=5,
        verbose=True,
        write_files=True,
        progressfn="progress.dat",
        reset_files=True,
    ):
        # Burnin
        if verbose:
            sys.stdout.write("# Burning in\t")
            sys.stdout.write("[%s]" % (" " * 50))
            sys.stdout.flush()
            sys.stdout.write("\b" * (50 + 1))
        for i in range(burnin):
            self.mcmc_step()
            if verbose and (i % (burnin / 50) == 0):
                sys.stdout.write("=")
                sys.stdout.flush()
        # MCMC
        if write_files:
            if reset_files:
                progressf = open(progressfn, "w")
            else:
                progressf = open(progressfn, "a")
        if verbose:
            sys.stdout.write("\n# Sampling\t")
            sys.stdout.write("[%s]" % (" " * 50))
            sys.stdout.flush()
            sys.stdout.write("\b" * (50 + 1))
        ypred = {}
        last_swap = dict([(T, 0) for T in self.Ts[:-1]])
        max_inactive_swap = 0
        for s in range(samples):
            # MCMC updates
            ready = False
            while not ready:
                for kk in range(thin):
                    self.mcmc_step()
                    BT1, BT2 = self.tree_swap()
                    if BT1 is not None:
                        last_swap[BT1] = s
                # Predict for this sample (prediction must be finite;
                # otherwise, repeat
                ypred[s] = self.trees["1"].predict(x)
                ready = True not in np.isnan(
                    np.array(ypred[s])
                ) and True not in np.isinf(np.array(ypred[s]))
            # Output
            if verbose and (s % (samples / 50) == 0):
                sys.stdout.write("=")
                sys.stdout.flush()
            if write_files:
                progressf.write(
                    "%s %d %s %lf %lf %d %s\n"
                    % (
                        list(x.index),
                        s,
                        str(list(ypred[s])),
                        self.trees["1"].E,
                        self.trees["1"].bic,
                        max_inactive_swap,
                        self.trees["1"],
                    )
                )
                progressf.flush()
            # Anneal if the some configuration is stuck
            max_inactive_swap = max([s - last_swap[T] for T in last_swap])
            if max_inactive_swap > anneal:
                self.anneal(n=anneal * thin, factor=annealf)
                last_swap = dict([(T, s) for T in self.Ts[:-1]])

        # Done
        if verbose:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return pd.DataFrame.from_dict(ypred)
