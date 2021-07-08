import numpy as np
from scipy import special

def _reshape_lattice_column(arr):
    n_components = len(arr)
    clonal = np.repeat(arr[0], n_components - 1)
    return np.vstack([clonal, arr[1:]])

def _reshape_transition_prob(arr, forward=True):
    # stack the diagonal and one of the row/column together for convenience
    transferred_to = np.diagonal(arr)[1:]
    if forward:
        clonal_to = arr[0, 1:]
        return np.vstack([clonal_to, transferred_to])
    else:
        to_clonal = arr[1:, 0]
        return np.vstack([to_clonal, transferred_to])

def _forward(n_samples, n_components, log_startprob,
             log_transmat, framelogprob):
    fwdlattice = np.zeros((n_samples, n_components))
    fwdlattice[0, :] = log_startprob + framelogprob[0, :]
    reshaped_a = _reshape_transition_prob(log_transmat)
    for t in xrange(1, n_samples):
        # compute fwdlattice for clonal state
        sumprob = special.logsumexp(fwdlattice[t - 1, :] + log_transmat[:, 0])
        fwdlattice[t, 0] = framelogprob[t, 0] + sumprob

        # compute fwdlattice for all other states
        # ignoring all zero transitions
        reshaped_alpha = _reshape_lattice_column(fwdlattice[t - 1, :])
        fwdlattice[t, 1:] = special.logsumexp(reshaped_a + reshaped_alpha, axis=0) + framelogprob[t, 1:]

    return fwdlattice

def _backward(n_samples, n_components, log_startprob,
             log_transmat, framelogprob):
    bwdlattice = np.zeros((n_samples, n_components))
    bwdlattice[-1, :] = 0
    reshaped_a = _reshape_transition_prob(log_transmat, forward=False)
    for t in xrange(n_samples - 2, -1, -1):
        bwdlattice[t, 0] = special.logsumexp(
            bwdlattice[t + 1, :] + framelogprob[t + 1, :] + log_transmat[0, :])

        reshaped_beta = _reshape_lattice_column(bwdlattice[t + 1, :])
        reshaped_logprob = _reshape_lattice_column(framelogprob[t + 1, :])
        bwdlattice[t, 1:] = special.logsumexp(reshaped_a + reshaped_beta + reshaped_logprob, axis=0)
    return bwdlattice

def _viterbi(n_samples, n_components, log_startprob,
             log_transmat,  framelogprob):
    # TODO
    state_sequence = np.empty(n_samples)
    logprob = 0
    return logprob, state_sequence
