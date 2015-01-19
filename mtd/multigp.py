# -*- coding: utf-8 -*-

from __future__ import division

import multiprocessing
import pickle

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
                if len(result) == 2:
                    # only return the diagonal of the covariance matrix
                    result = result[0], result[1].diagonal()

            elif cmd == 'get_kernel_pars':
                result = gp.kernel.pars

            elif cmd == 'set_kernel_pars':
                gp.kernel.pars = args[0]
                result = None

            elif cmd == 'train':
                prior, nwalkers, nsteps, nburnsteps, verbose = args

                def log_post(pars):
                    if np.any(pars < 0.):
                        return -np.inf
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

                # set hyperparameters to median
                median = np.median(sampler.flatchain, axis=0)
                gp.kernel.pars = median

                if verbose:
                    print(self.name, 'training complete')
                    _print_sampler_stats(sampler)
                    print('  median',
                          _format_number_list(log_post(median), *median))

                # delete ref. to log_post()
                sampler.lnprobfn = None

                # return sampler and delete reference so memory will be cleared
                # at next command
                result = sampler
                del sampler

            else:
                result = ValueError('Unknown command: {}.'.format(cmd))

            pipe.send(result)

    def send_cmd(self, cmd, *args, **kwargs):
        self._in_pipe.send((cmd, args, kwargs))
        return self

    def send_bytes(self, buffer):
        self._in_pipe.send_bytes(buffer)
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

    @property
    def pca(self):
        """
        The underlying PCA object.

        """
        return self._pca

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

    def _send_cmd_all_procs(self, cmd, *args, **kwargs):
        """
        Send the same command to each subprocess and retrieve results.

        """
        buffer = pickle.dumps((cmd, args, kwargs))

        for p in self._procs:
            p.send_bytes(buffer)

        return tuple(p.get_result() for p in self._procs)

    def get_kernel_pars(self, n):
        """
        Retrieve the kernel hyperparameters for the nth GP.

        """
        return self._procs[n].send_cmd('get_kernel_pars').get_result()

    def set_kernel_pars(self, n, pars):
        """
        Set the kernel hyperparameters for the nth GP.

        """
        self._procs[n].send_cmd('set_kernel_pars', pars).get_result()

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
        self._training_samplers = self._send_cmd_all_procs(
            'train', prior, nwalkers, nsteps, nburnsteps, verbose
        )

    @property
    def training_samplers(self):
        """
        A tuple of the training samplers for each GP.

        """
        try:
            return self._training_samplers
        except AttributeError:
            raise RuntimeError('Training has not run yet.')

    def _predict_pc(self, t, mean_only=True, out=None):
        """
        Predict principal components at test points.

        t : (ntest, ndim)
            Test points.  Must already be standardized.
        mean_only : boolean, default True
            Whether to predict the GP mean only or also the variance.
        out : (ntest, npc), optional
            Array to write mean results into.

        """
        results = self._send_cmd_all_procs('predict', t, mean_only=mean_only)

        if out is None:
            out = np.empty((t.shape[0], len(self)))

        if mean_only:
            # results contains only the mean
            out.T[:] = results
            return out
        else:
            # results contains (mean, var) tuples -- unpack with zip
            out_var = np.empty_like(out)
            out.T[:], out_var.T[:] = zip(*results)
            return out, out_var

    def predict(self, t, mean_only=True):
        """
        Calculate predictions at test points.

        t : (ntest, ndim)
        mean_only : boolean, default True
            Whether to predict the GP mean only or also the variance.  The
            covariance is *not* computed between the test points t, only the
            variance of each output.

        """
        t = self._standardize(atleast_2d_column(t))
        z = self._predict_pc(t, mean_only=mean_only)

        if mean_only:
            return self._pca.inverse(z)
        else:
            return self._pca.inverse(z[0], var=z[1], y_cov=False)

    def calibrate(self, yexp, yerr,
                  nwalkers, nsteps, nburnsteps=None,
                  prior=None, verbose=False):
        """
        Calibrate GP input parameters to data.

        yexp: (nfeatures,)
            Experimental/calibration data.
        yerr: float
            Fractional error on experimental data.
        nwalkers: number of MCMC walkers
        nsteps, nburnsteps: number of MCMC steps per walker
            nsteps must be specified, nburnsteps is optional.  If only
            nburnsteps is not given, both the burn-in and production chains
            will have nsteps; if nburnsteps is given, the burn-in chain will
            have nburnsteps and the production chain will have nsteps.
        prior: Prior object, optional
            Priors for input parameters.  If not given, a flat prior is placed
            on each parameter (and this is faster than explicitly providing a
            UniformPrior).
        verbose : boolean
            Whether to output status info.

        """
        zexp = self._pca.transform(np.atleast_2d(yexp))
        zerrsq = np.square(yerr*zexp)
        zweights = -.5 * self._pca.var[:self._pca.npc] / zerrsq

        def log_likelihood(theta):
            zmodel = self._predict_pc(theta[np.newaxis, :])
            log_prob = np.inner(np.square(zmodel-zexp), zweights)
            return log_prob, zmodel

        if prior is None:
            pos0 = np.random.rand(nwalkers, self._ndim)

            def log_post(theta):
                if np.any((theta < 0.) | (theta > 1.)):
                    return -np.inf, None
                else:
                    return log_likelihood(theta)

        else:
            pos0 = prior.rvs(nwalkers)

            def log_post(theta):
                log_prior = prior.logpdf(theta)
                if not np.isfinite(log_prior):
                    return -np.inf, None
                else:
                    log_prob, zmodel = log_likelihood(theta)
                    return log_prior + log_prob, zmodel

        sampler = emcee.EnsembleSampler(nwalkers, self._ndim, log_post)

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

        self._cal_sampler = sampler

    @property
    def cal_sampler(self):
        """
        Calibration MCMC sampler object.  Note that cal_sampler.(flat)chain
        should not be accessed directly; instead, use the MultiGP properties
        cal_(flat)chain, which properly transform back to "normal" space.  Use
        MultiGP.cal_samples to access the calibration samples.

        """
        try:
            return self._cal_sampler
        except AttributeError:
            raise RuntimeError('Calibration has not run yet.')

    @property
    def cal_chain(self):
        """
        Calibration MCMC chain, shape (nwalkers, nsteps, ndim).

        """
        chain = self._destandardize(self.cal_sampler.chain)

        return chain

    @property
    def cal_flatchain(self):
        """
        Flat calibration MCMC chain, shape (nwalkers*nsteps, ndim).

        """
        flatchain = self._destandardize(self.cal_sampler.flatchain)

        return flatchain

    @property
    def cal_samples(self):
        """
        Posterior calibration samples, shape (nwalkers*nsteps, ndim).

        """
        pc_samples = np.reshape(self.cal_sampler.blobs, (-1, self._pca.npc))
        samples = self._pca.inverse(pc_samples)

        return samples


def _format_number_list(*args, **kwargs):
    """
    Format a list of numbers into a nice string.

    """
    fmt = kwargs.pop('fmt', '{:.3g}')
    sep = kwargs.pop('sep', ' ')

    return sep.join(fmt.format(i) for i in args)


def _print_sampler_stats(sampler):
    """
    Output MCMC sampler statistics.

    """
    afrac = sampler.acceptance_fraction
    print('  acceptance fraction',
          _format_number_list(afrac.mean(), afrac.std(), sep=' ± '))
    window = min(50, int(sampler.chain.shape[1]/2))
    acor = sampler.get_autocorr_time(window)
    print('  autocorrelation times', _format_number_list(*acor))
