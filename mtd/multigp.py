# -*- coding: utf-8 -*-

from __future__ import division

import multiprocessing

import numpy as np
import emcee
from george import GP

from .pca import PCA
from .util import atleast_2d_column

__all__ = 'MultiGP',


class _GPProcess(multiprocessing.Process):
    """
    Run a GP in a separate process.

    """
    def __init__(self, x, y, kernel, name=None):
        self._x = x
        self._y = y
        self._kernel = kernel
        self._in_pipe, self._out_pipe = multiprocessing.Pipe()
        super(_GPProcess, self).__init__(name=name)

    def run(self):
        gp = GP(self._kernel)
        gp.compute(self._x)
        y = self._y
        pipe = self._out_pipe

        for cmd, args, kwargs in iter(pipe.recv, None):
            if cmd == 'predict':
                result = gp.predict(y, *args, **kwargs)

            elif cmd == 'train':
                prior, nwalkers, nsteps, nburnsteps, verbose = args

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

                if verbose:
                    print(self.name, 'starting training burn-in')

                # run burn-in chain
                nburnsteps = nburnsteps or nsteps
                pos1 = sampler.run_mcmc(pos0, nburnsteps,
                                        storechain=verbose)[0]

                if verbose:
                    print(self.name, 'burn-in complete')
                    _print_sampler_stats(sampler)
                    print(self.name, 'starting production')

                # run production chain
                sampler.reset()
                sampler.run_mcmc(pos1, nsteps)

                if verbose:
                    print(self.name, 'training complete')
                    _print_sampler_stats(sampler)

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
        x = atleast_2d_column(x)
        self._x_min = x.min(axis=0)
        self._x_range = x.ptp(axis=0)
        x = self._standardize(x)
        self._ndim = x.shape[1]

        y = atleast_2d_column(y)
        if x.shape[0] != y.shape[0]:
            raise ValueError('x and y must have same number of samples.')

        self._pca = PCA(y, npc=npc)

        self._procs = tuple(
            _GPProcess(x, y, kernel, name='GP{}'.format(n))
            for n, y in enumerate(self._pca.transform().T)
        )
        for p in self._procs:
            p.start()

    def __len__(self):
        return self._pca.npc

    def __del__(self):
        try:
            for p in self._procs:
                p.stop()
        except AttributeError:
            pass

    def _standardize(self, x):
        """
        Scale x to the unit hypercube [0, 1]^ndim.

        """
        z = np.array(x, copy=True)
        z -= self._x_min
        z /= self._x_range

        return z

    def _destandardize(self, z):
        """
        Scale z back from the unit hypercube.

        """
        x = np.array(z, copy=True)
        x *= self._x_range
        x += self._x_min

        return x

    def train(self, prior, nwalkers, nsteps, nburnsteps=None, verbose=False):
        """
        Train the GPs, i.e. estimate the optimal hyperparameters via MCMC.

        prior: Prior object
            Priors for the kernel hyperparameters.
        nwalkers: number of MCMC walkers
        nsteps, nburnsteps: number of MCMC steps per walker
            nsteps must be specified, nburnsteps is optional.  If only
            nburnsteps is not given, both the burn-in and production chains
            will have nsteps; if nburnsteps is given, the burn-in chain will
            have nburnsteps and the production chain will have nsteps.
        verbose : boolean
            Whether to output status info.

        """
        for p in self._procs:
            p.send_cmd('train', prior, nwalkers, nsteps, nburnsteps, verbose)
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
        t = self._standardize(atleast_2d_column(t))
        z = self._predict_pc(t)

        return self._pca.inverse(z)

    def calibrate(self, yexp, yerr, prior,
                  nwalkers, nsteps, nburnsteps=None,
                  verbose=False):
        """
        Calibrate GP input parameters to data.

        yexp: (nfeatures,)
            Experimental/calibration data.
        yerr: float
            Fractional error on experimental data.
        prior: Prior object
            Priors for input parameters.
        nwalkers: number of MCMC walkers
        nsteps, nburnsteps: number of MCMC steps per walker
            nsteps must be specified, nburnsteps is optional.  If only
            nburnsteps is not given, both the burn-in and production chains
            will have nsteps; if nburnsteps is given, the burn-in chain will
            have nburnsteps and the production chain will have nsteps.
        verbose : boolean
            Whether to output status info.

        """
        zexp = self._pca.transform(np.atleast_2d(yexp))
        zerrsq = np.square(yerr*zexp)

        def log_post(theta):
            log_prior = prior.logpdf(theta)
            if not np.isfinite(log_prior):
                return -np.inf
            zmodel = self._predict_pc(np.atleast_2d(theta))
            log_prob = -.5*np.sum(np.square(zmodel-zexp)/zerrsq)
            return log_prior + log_prob

        sampler = emcee.EnsembleSampler(nwalkers, self._ndim, log_post)

        # sample random initial position from prior
        pos0 = prior.rvs(nwalkers)

        if verbose:
            print('starting calibration burn-in')

        # run burn-in chain
        nburnsteps = nburnsteps or nsteps
        pos1 = sampler.run_mcmc(pos0, nburnsteps, storechain=verbose)[0]

        if verbose:
            print('burn-in complete')
            _print_sampler_stats(sampler)
            print('starting production')

        # run production chain
        sampler.reset()
        sampler.run_mcmc(pos1, nsteps)

        if verbose:
            print('calibration complete')
            _print_sampler_stats(sampler)

        # delete ref. to log_post()
        sampler.lnprobfn = None

        self._sampler = sampler

    def get_calibration_chain(self, flat=True):
        """
        Retrieve the calibration MCMC chain.

        flat : boolean, default True
            Whether to return the chain flattened to shape
            (nwalkers*nsteps, ndim) or with per-walker shape
            (nwalkers, nsteps, ndim).

        """
        chain = self._sampler.flatchain if flat else self._sampler.chain
        chain = self._destandardize(chain)

        return chain


def _print_sampler_stats(sampler, fmt_str='{:.3g}'):
    """
    Output MCMC sampler statistics.

    """
    afrac = sampler.acceptance_fraction
    print('  acceptance fraction',
          ' ± '.join(fmt_str.format(i) for i in (afrac.mean(), afrac.std())))
    window = min(50, int(sampler.chain.shape[1]/2))
    acor = sampler.get_autocorr_time(window)
    print('  autocorrelation times',
          ' '.join(fmt_str.format(i) for i in acor))
