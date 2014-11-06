# -*- coding: utf-8 -*-

from __future__ import division

import multiprocessing

import numpy as np
import emcee
from george import GP

from .pca import PCA

__all__ = 'MultiGP',


class _GPProcess(multiprocessing.Process):
    """
    Run a GP in a separate process.

    """
    def __init__(self, x, y, kernel):
        self._x = x
        self._y = y
        self._kernel = kernel
        self._in_pipe, self._out_pipe = multiprocessing.Pipe()
        super(_GPProcess, self).__init__()

    def run(self):
        gp = GP(self._kernel)
        gp.compute(self._x)
        y = self._y
        pipe = self._out_pipe

        for cmd, args, kwargs in iter(pipe.recv, None):
            if cmd == 'predict':
                result = gp.predict(y, *args, **kwargs)

            elif cmd == 'train':
                prior, nwalkers, nsteps = args

                def log_post(pars):
                    log_prior = prior.logpdf(pars)
                    if not np.isfinite(log_prior):
                        return -np.inf
                    gp.kernel.pars = pars
                    return log_prior + gp.lnlikelihood(y, quiet=True)

                ndim = len(gp.kernel)
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post)

                # sample random initial position from prior
                pos0 = prior.rvs(nwalkers)

                # run burn-in chain
                pos1, *_ = sampler.run_mcmc(pos0, nsteps, storechain=False)
                sampler.reset()

                # run production chain
                sampler.run_mcmc(pos1, nsteps)

                # set hyperparameters to max posterior point from chain
                gp.kernel.pars = sampler.flatchain[
                    sampler.lnprobability.argmax()
                ]

                result = None

            elif cmd == 'get_sampler_attr':
                try:
                    result = getattr(sampler, args[0])
                except NameError:
                    result = RuntimeError(
                        'Training sampler has not been created yet.'
                    )
                except AttributeError as e:
                    result = e

            else:
                result = ValueError('Unknown command: {}.'.format(cmd))

            pipe.send(result)

    def send_cmd(self, cmd, *args, **kwargs):
        self._in_pipe.send((cmd, args, kwargs))
        return self

    def stop(self):
        self._in_pipe.send(None)

    def get_result(self):
        res = self._in_pipe.recv()
        if isinstance(res, Exception):
            raise res
        return res


class MultiGP(object):
    """
    Multidimensional Gaussian process using principal component analysis.

    x: (nsamples, ndim)
        Design matrix.
    y: (nsamples, nfeatures)
        Model outputs.
    kernel: george kernel
        Template kernel for each GP.
    npc: number of principal components to use
        Passed directly to the PCA constructor.  A separate process will be
        created for each PC.

    """
    def __init__(self, x, y, kernel, npc=None):
        x = np.atleast_2d(x)
        self._x_min = x.min(axis=0)
        self._x_range = x.ptp(axis=0)
        x = self._standardize(x)
        self._ndim = x.shape[1]

        self._pca = PCA(y, npc=npc, normalize=True)

        self._procs = tuple(
            _GPProcess(x, y, kernel)
            for y in self._pca.transform().T
        )
        for p in self._procs:
            p.start()

    def __del__(self):
        for p in self._procs:
            p.stop()

    def _standardize(self, x):
        """
        Scale x to the unit hypercube [0, 1]^ndim.

        """
        z = np.copy(x)
        z -= self._x_min
        z /= self._x_range

        return z

    def train(self, prior, nwalkers, nsteps):
        """
        Train the GPs, i.e. estimate the optimal hyperparameters via MCMC.

        prior: Prior object
            Priors for the kernel hyperparameters.
        nwalkers: number of MCMC walkers
        nsteps: number of MCMC steps per walker
            Both the burn-in and production chains will have nsteps.

        """
        for p in self._procs:
            p.send_cmd('train', prior, nwalkers, nsteps)
        for p in self._procs:
            # wait for results
            p.get_result()

    def get_training_sampler_attr(self, n, attr):
        """
        Retrieve an attribute from a hyperparameter training sampler.

        n : integer
            Index of sampler to access.
        attr : string
            Attribute name.

        """
        return self._procs[n].send_cmd('get_sampler_attr', attr).get_result()

    def _predict_pc(self, t):
        """
        Predict principal components at test points.

        t : (ntest, ndim)
            Test points.  Must already be standardized.

        """
        for p in self._procs:
            p.send_cmd('predict', t, mean_only=True)
        z = np.column_stack([p.get_result() for p in self._procs])

        return z

    def predict(self, t):
        """
        Calculate predictions at test points.

        t : (ntest, ndim)

        """
        t = self._standardize(np.atleast_2d(t))
        z = self._predict_pc(t)

        return self._pca.inverse(z)

    def calibrate(self, yexp, yerr, prior, nwalkers, nsteps):
        """
        Calibrate GP input parameters to data.

        yexp: (nfeatures,)
            Experimental/calibration data.
        yerr: float
            Fractional error on experimental data.
        prior: Prior object
            Priors for input parameters.
        nwalkers: number of MCMC walkers
        nsteps: number of MCMC steps per walker
            Both the burn-in and production chains will have nsteps.

        """
        zexp = self._pca.transform(yexp)
        zerrsq = np.square(yerr*zexp)

        def log_post(theta):
            log_prior = prior.logpdf(theta)
            if not np.isfinite(log_prior):
                return -np.inf
            zmodel = self._predict_pc(theta)
            log_prob = -.5*np.sum(np.square(zmodel-zexp)/zerrsq)
            return log_prior + log_prob

        sampler = emcee.EnsembleSampler(nwalkers, self._ndim, log_post)

        # sample random initial position from prior
        pos0 = prior.rvs(nwalkers)

        # run burn-in chain
        pos1, *_ = sampler.run_mcmc(pos0, nsteps, storechain=False)
        sampler.reset()

        # run production chain
        sampler.run_mcmc(pos1, nsteps)
