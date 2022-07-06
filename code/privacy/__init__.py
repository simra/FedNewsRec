# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import json
from math import sqrt, exp, log
import privacy.analysis as privacy_analysis

def update_privacy_accountant(noise_multiplier, clip_norm, delta, num_clients, curr_iter, num_clients_curr_iter):
    
    K = 1  # from DP perspective each user is contributing one gradient
    B = num_clients_curr_iter  # batch size
    n = num_clients
    T = curr_iter + 1
    _delta = delta
    # _delta = dp_config.get('delta', min(1e-7, 1. / (n * log(n))))  # TODO should be precomputed in config
    
    global_sigma = noise_multiplier
    # TODO: this looks like LDP noise scale. Check that the GDP noise we apply is global_sigma*clip_norm
    noise_scale = global_sigma * clip_norm / B

    try:
        mu = K * B / n * sqrt(T * exp((1. / global_sigma) ** 2 - 1))
    except OverflowError:
        print(f"Error computing mu {global_sigma} {K} {B} {n} {T}")
        mu = -1

    orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] + list(range(5, 64)) + [128, 256, 512])
    q = B / n
    _sigma = global_sigma  # was: noise_scale but we should apply the noise multiplier.
    rdp = privacy_analysis.compute_rdp(q, _sigma, T, orders)

    rdp_epsilon, opt_order = privacy_analysis.get_privacy_spent(orders, rdp, _delta)

    """
    props = {
        'dp_global_K': K,  # gradients per user
        'dp_global_B': B,  # users per batch
        'dp_global_n': n,  # total users
        'dp_global_T': T,  # how many iterations
        'dp_sigma': _sigma,  # noise_multiplier. Should be combined global+local sigma.
        'dp_global_mu': mu,
        # 'dp_epsilon_fdp': fdp_epsilon,
        'dp_epsilon_rdp': rdp_epsilon,
        # 'dp_epsilon_exact': exact_eps,
        'dp_opt_order': opt_order,
        'dp_delta': _delta,
        'dp_noise_scale': noise_scale  # Note: not needed for accounting.
    }

    print(f'DP accounting: {json.dumps(props)}')
    """
    return rdp_epsilon
    