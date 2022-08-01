import numpy as np
import os
from scipy import special
from hmmlearn import _hmmc
from .utils import log_normalize, log_mask_zero


class ClosePairHMM:
    def __init__(self,
                 species_name=None, block_size=1,
                 transfer_emissions=np.array([0.1]),
                 transfer_rate=1e-2, clonal_emission=1e-3,
                 transfer_length=5e2, transition_prior=None,
                 n_iter=5, min_clonal_emission=1e-6):

        # init emission probabilities
        if species_name is not None:
            self.transfer_emissions, self.transition_prior = self._get_empirical_emissions(
                species_name, block_size)
        else:
            self._init_emissions_manual(transfer_emissions, transition_prior)
        self.clonal_emission = clonal_emission
        self.init_clonal_emission = clonal_emission
        self.min_clonal_emissions = min_clonal_emission
        self.all_emissions = np.concatenate([[self.clonal_emission], self.transfer_emissions])

        # normalizing the transition prior
        n_components = 1 + len(self.transfer_emissions)
        self.n_components = n_components
        self.transition_prior = self.transition_prior.astype(np.float32) / np.sum(self.transition_prior)
        self.transfer_rate = transfer_rate
        self.init_transfer_rate = transfer_rate
        self.exit_rate = 1. / transfer_length  # rate of leaving the transferred state
        self._init_transitions()

        self.n_iter = n_iter

    def _get_empirical_emissions(self, species_name, block_size):
        path = os.path.join(os.path.dirname(__file__), 'dat', species_name + '.csv')
        if not os.path.exists(path):
            raise ValueError("No empirical data found for {}".format(species_name))
        dat = np.loadtxt(path)
        prob_has_snp = 1 - np.power(1 - dat[0, :], block_size)
        return prob_has_snp, dat[1, :]

    def _init_emissions_manual(self, transfer_emissions, transition_prior):
        if transfer_emissions is not None:
            if transition_prior is not None:
                if len(transition_prior) != len(transfer_emissions):
                    raise ValueError("Transition prior must have the same length as transfer emissions")
                if (transition_prior < 0).any():
                    raise ValueError("Transition prior must be all positive")
                self.transfer_emissions = transfer_emissions
                self.transition_prior = transition_prior
            else:
                print("No transition prior provided. Assuming uniform transition probability")
                self.transfer_emissions = transfer_emissions
                self.transition_prior = np.ones(transfer_emissions.shape)
        else:
            raise ValueError("Please provide either the species name for empirical emission rates "
                             "or relevant parameters directly")

    def _init_transitions(self):
        self.startprob_ = np.zeros(self.n_components)
        self.startprob_[0] = 1  # always starts from clonal state

        # transmat is very sparse; no transitions between the recombined/transferred states
        self.transmat_ = np.zeros((self.n_components, self.n_components))
        # transitions from the recombined state
        # TODO: modify this to allow state-wise exit rate
        self.transmat_[1:, 0] = self.exit_rate
        self.transmat_[np.diag_indices(self.n_components)] = 1 - self.exit_rate
        # transitions from the clonal state
        self.transmat_[0, 0] = 1 - self.transfer_rate
        self.transmat_[0, 1:] = self.transfer_rate * self.transition_prior

    def _update_clonal_emission(self, emission_rate):
        self.clonal_emission = emission_rate
        self.all_emissions[0] = emission_rate

    def _update_transfer_rate(self, transfer_rate):
        self.transfer_rate = transfer_rate
        self._init_transitions()

    def reinit_emission_and_transfer_rates(self):
        self._update_clonal_emission(self.init_clonal_emission)
        self._update_transfer_rate(self.init_transfer_rate)

    def _compute_log_likelihood(self, X):
        # each observation will be either "snp in bin / 1" or "no snp in bin/ 0"
        # so the emission simply follows bernoulli RV
        # logp = np.zeros((X.shape[0], self.transfer_emissions.shape[0] + 1))
        # logp[:, 0] = np.squeeze(bernoulli.logpmf(X, self.clonal_emission))
        # logp[:, 1:] = bernoulli.logpmf(X, self.transfer_emissions)
        # clonal_logp = poisson.logpmf(X, self.clonal_emission)
        # transfer_logp = poisson.logpmf(X, self.transfer_emissions)
        # logp = np.hstack([clonal_logp, transfer_logp])
        with np.errstate(divide='raise'):
            logp = np.log(1 - self.all_emissions + np.outer(X, 2 * self.all_emissions - 1))
        return logp

    def _check(self):
        self.startprob_ = np.asarray(self.startprob_)
        if len(self.startprob_) != self.n_components:
            raise ValueError("startprob_ must have length n_components")
        if not np.allclose(self.startprob_.sum(), 1.0):
            raise ValueError("startprob_ must sum to 1.0 (got {:.4f})"
                             .format(self.startprob_.sum()))

        self.transmat_ = np.asarray(self.transmat_)
        if self.transmat_.shape != (self.n_components, self.n_components):
            raise ValueError(
                "transmat_ must have shape (n_components, n_components)")
        if not np.allclose(self.transmat_.sum(axis=1), 1.0):
            raise ValueError("rows of transmat_ must sum to 1.0 (got {})"
                             .format(self.transmat_.sum(axis=1)))

    def _check_array(self, X):
        if len(X.shape)==1:
            raise ValueError("Please reshape sequence to be (length, 1) ")
        n_samples, n_features = X.shape
        if n_features != 1:
            raise ValueError("Only supports binned 1d genome as input data")
        return

    def fit(self, X):
        self._check()
        self._check_array(X)

        for iter in range(self.n_iter):
            framelogprob = self._compute_log_likelihood(X)
            if np.isnan(np.sum(framelogprob)):
                raise ValueError("Nan detected in log likelihood. Check X")
            logprob, fwdlattice = self._do_forward_pass(framelogprob)
            bwdlattice = self._do_backward_pass(framelogprob)
            log_xi_sum = self._do_forward_backward(fwdlattice, bwdlattice, framelogprob)

            # estimator of no transfer is prob{i->i} / prob{i->any}
            transfer_rate_est = 1 - np.exp(log_xi_sum[0, 0] - special.logsumexp(log_xi_sum[0, :]))
            self._update_transfer_rate(transfer_rate_est)

            # estimator of clonal divergence
            alpha_beta = fwdlattice[:, 0] + bwdlattice[:, 0]
            emission_est = special.logsumexp(alpha_beta[np.squeeze(X > 0)]) \
                           - special.logsumexp(alpha_beta)
            # set a lower limit for clonal emission to prevent zero probability
            emission_est = max(np.exp(emission_est), self.min_clonal_emissions)
            self._update_clonal_emission(emission_est)
        return self

    def decode(self, X):
        self._check()
        self._check_array(X)

        framelogprob = self._compute_log_likelihood(X)
        logprob, state_sequence = self._do_viterbi_pass(framelogprob)
        return logprob, state_sequence

    def _do_forward_pass(self, framelogprob):
        n_samples, n_components = framelogprob.shape
        # archived numpy version
        # fwdlattice = _routines._forward(n_samples, n_components,
        #                log_mask_zero(self.startprob_),
        #                log_mask_zero(self.transmat_),
        #                framelogprob)

        fwdlattice = np.zeros((n_samples, n_components))
        _hmmc._forward(n_samples, n_components,
                       log_mask_zero(self.startprob_),
                       log_mask_zero(self.transmat_),
                       framelogprob, fwdlattice)

        with np.errstate(under="ignore"):
            return special.logsumexp(fwdlattice[-1]), fwdlattice

    def _do_backward_pass(self, framelogprob):
        n_samples, n_components = framelogprob.shape
        bwdlattice = np.zeros((n_samples, n_components))
        _hmmc._backward(n_samples, n_components,
                        log_mask_zero(self.startprob_),
                        log_mask_zero(self.transmat_),
                        framelogprob, bwdlattice)
        # bwdlattice = _routines._backward(n_samples, n_components,
        #                 log_mask_zero(self.startprob_),
        #                 log_mask_zero(self.transmat_),
        #                 framelogprob)
        
        return bwdlattice

    def _do_viterbi_pass(self, framelogprob):
        n_samples, n_components = framelogprob.shape
        # logprob, state_sequence = _routines._viterbi(
        #     n_samples, n_components, log_mask_zero(self.startprob_),
        #     log_mask_zero(self.transmat_), framelogprob)
        state_sequence, logprob = _hmmc._viterbi(
                n_samples, n_components, log_mask_zero(self.startprob_),
                log_mask_zero(self.transmat_), framelogprob)
        return logprob, state_sequence

    def _do_forward_backward(self, fwdlattice, bwdlattice, framelogprob):
        n_samples, n_components = framelogprob.shape
        log_xi_sum = np.full((n_components, n_components), -np.inf)
        _hmmc._compute_log_xi_sum(n_samples, n_components, fwdlattice,
                                  log_mask_zero(self.transmat_),
                                  bwdlattice, framelogprob,
                                  log_xi_sum)
        return log_xi_sum

    def _compute_posteriors(self, fwdlattice, bwdlattice):
        # gamma is guaranteed to be correctly normalized by logprob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively.
        log_gamma = fwdlattice + bwdlattice
        log_normalize(log_gamma, axis=1)
        with np.errstate(under="ignore"):
            return np.exp(log_gamma)
