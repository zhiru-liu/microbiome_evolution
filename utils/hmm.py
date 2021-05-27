import numpy as np
from hmmlearn.base import _BaseHMM
from scipy.stats import poisson


class PoissonHMM(_BaseHMM):
    def __init__(self,
                 init_means=None, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stm", init_params="stm"):
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter, tol=tol, verbose=verbose,
                          params=params, init_params=init_params)
        self.init_means = init_means

    def _init(self, X, lengths=None):
        super(PoissonHMM, self)._init(X, lengths=lengths)
        _n_samples, n_features = X.shape
        if n_features != 1:
            raise ValueError("Only supporting 1d Poisson for our purpose. "
                             "Input data must have shape (n_samples, 1)")
        if self.init_means is not None:
            self.means_ = np.squeeze(np.array(self.init_means))
        else:
            raise ValueError("Must supply the initial means for Poisson")
        return

    def _check(self):
        super(PoissonHMM, self)._check()
        # checking the shape of means of Poisson
        if self.means_.shape != (self.n_components, ):
            raise ValueError("Means must have shape (n_components, ),"
                             "actual shape: {}".format(self.means_.shape))
        return

    def _generate_sample_from_state(self, state, random_state=None):
        return

    def _compute_log_likelihood(self, X):
        n_samples = X.shape[0]
        logp = np.zeros(shape=(n_samples, self.n_components))
        for i in range(self.n_components):
            logp[:, i] = np.squeeze(poisson.logpmf(X, self.means_[i]))
        return logp

    def _initialize_sufficient_statistics(self):
        stats = super(PoissonHMM, self)._initialize_sufficient_statistics()
        stats['sum_p'] = np.zeros(self.n_components)
        stats['sum_px'] = np.zeros(self.n_components)
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(PoissonHMM, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
        if 'm' in self.params:
            # TODO: 05.26.2021 figure out whether the next two lines should use posteriors instead of framelogprob!!
            stats['sum_p'] += np.transpose(np.sum(framelogprob, axis=0))
            stats['sum_px'] += np.squeeze(np.dot(np.transpose(framelogprob), X))
        return

    def _do_mstep(self, stats):
        super(PoissonHMM, self)._do_mstep(stats)
        if 'm' in self.params:
            self.means_ = np.divide(stats['sum_px'], stats['sum_p'])
        return
