import numpy as np
from hmmlearn.base import _BaseHMM
from scipy.stats import poisson, bernoulli


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
            stats['sum_p'] += np.transpose(np.sum(posteriors, axis=0))
            stats['sum_px'] += np.squeeze(np.dot(np.transpose(posteriors), X))
        return

    def _do_mstep(self, stats):
        super(PoissonHMM, self)._do_mstep(stats)
        if 'm' in self.params:
            self.means_ = np.divide(stats['sum_px'], stats['sum_p'])
        return


class ClosePairHMM(_BaseHMM):
    def __init__(self,
                 species_name=None, transfer_emissions=np.array([0.1]),
                 transfer_rate=1e-2, clonal_emission=1e-3,
                 transfer_length=5e2,
                 algorithm="viterbi", n_iter=10, tol=1e-2,
                 verbose=False, params="m"):
        if species_name is not None:
            self.transfer_emissions, self.transition_prior = self.get_empirical_emissions(species_name)
        elif transfer_emissions is not None:
            self.transfer_emissions = transfer_emissions
            self.transition_prior = np.ones(transfer_emissions.shape)
        else:
            raise ValueError("Must supply either the species name " +
                             "or the emission probability in transferred regions")
        n_components = 1 + len(self.transfer_emissions)
        # normalizing the transition prior
        self.transition_prior = self.transition_prior.astype(np.float32) / np.sum(self.transition_prior)
        self.transfer_rate = transfer_rate
        self.clonal_emission = clonal_emission
        self.exit_rate = 1. / transfer_length  # rate of leaving the transferred state

        _BaseHMM.__init__(self, n_components,
                          algorithm=algorithm,
                          n_iter=n_iter, tol=tol, verbose=verbose,
                          params=params)

    def get_empirical_emissions(self, species_name):
        # TODO: implement this!
        return None, None

    def _init(self, X, lengths=None):
        init = 1. / self.n_components
        self.startprob_ = np.zeros(self.n_components)
        self.startprob_[0] = 1  # always starts from clonal state

        # transmat is very sparse; no transitions between the recombined/transferred states
        self.transmat_ = np.zeros((self.n_components, self.n_components))
        # transitions from the recombined state
        self.transmat_[1:, 0] = self.exit_rate
        self.transmat_[np.diag_indices(self.n_components)] = 1 - self.exit_rate
        # transitions from the clonal state
        self.transmat_[0, 0] = 1 - self.transfer_rate
        self.transmat_[0, 1:] = self.transfer_rate * self.transition_prior

        _n_samples, n_features = X.shape
        if n_features != 1:
            raise ValueError("Only supports binned 1d genome as input data")
        return

    def _compute_log_likelihood(self, X):
        # each observation will be either "snp in bin / 1" or "no snp in bin/ 0"
        # so the emission simply follows bernoulli RV
        clonal_logp = bernoulli.logpmf(X, self.clonal_emission)
        transfer_logp = bernoulli.logpmf(X, self.transfer_emissions)
        # clonal_logp = poisson.logpmf(X, self.clonal_emission)
        # transfer_logp = poisson.logpmf(X, self.transfer_emissions)
        logp = np.hstack([clonal_logp, transfer_logp])
        return logp

    def _initialize_sufficient_statistics(self):
        # TODO may need to implement for inferring wall clock time
        return

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        # TODO may need to implement for inferring wall clock time
        print("Skipping")
        return

    def _do_mstep(self, stats):
        # TODO may need to implement for inferring wall clock time
        print("skipping m step")
        return
