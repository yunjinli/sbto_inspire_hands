import numpy as np

def get_top_samples(
    costs: np.ndarray,
    samples: np.ndarray,
    N_top_samples: float
    ) -> np.ndarray:
    assert costs.ndim == 2, "Expected costs of shape (N, T)."
    assert samples.ndim == 3, "Expected samples of shape (N, T, D)."
    assert costs.shape[:2] == samples.shape[:2], "Mismatched iteration/sample dimensions."

    D = samples.shape[2]
    # Flatten across all iterations
    costs_flat = costs.reshape(-1)
    samples_flat = samples.reshape(-1, D)

    # Remove samples in double (in case keep elites for instance)
    costs_flat_unique, arg_unique = np.unique(costs_flat, return_index=True, sorted=True)
    samples_flat_unique = samples_flat[arg_unique]

    top_samples = samples_flat_unique[:N_top_samples]
    top_costs = costs_flat_unique[:N_top_samples]

    if top_costs.size > 0 and not np.all(np.isfinite(top_costs)):
        n_nonfinite_top = int(np.sum(~np.isfinite(top_costs)))
        n_finite_total = int(np.sum(np.isfinite(costs_flat)))
        print(
            f"[get_top_samples] Warning: {n_nonfinite_top}/{top_costs.size} "
            f"requested top samples have non-finite cost ({n_finite_total} "
            f"finite samples existed out of {costs_flat.size} total). This "
            "means every sample in the requested window diverged (NaN/Inf "
            "physics rollout) -- the returned 'best' trajectory/cost is not "
            "a real optimum, just the least-bad diverged sample available. "
            "Caller should fall back to a known-good solver state if possible."
        )

    return top_samples, top_costs