import numpy as np
import numpy.typing as npt
from typing import Tuple
from dataclasses import dataclass

from sbto.solvers.solver_base import SamplingBasedSolver, SolverState, ConfigSolver

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

@dataclass
class ConfigCEM(ConfigSolver):
    """
    elite_frac: Fraction of samples considered elite.
    alpha_mean: Smoothing coefficient for mean update.
    alpha_cov: Smoothing coefficient for covariance update.
    std_incr: Increase the diag of the cov matrix.
    """
    elite_frac: float = 0.05
    alpha_mean: float = 0.9
    alpha_cov: float = 0.1
    std_incr: float = 0.
    keep_frac: float = 0.
    min_std_collapsed: float = 0.
    _target_:str = "sbto.solvers.cem.CEM"
    
class CEM(SamplingBasedSolver):
    """
    Cross-Entropy Method (CEM) solver.
    """
    def __init__(self, D, cfg: ConfigCEM):
        super().__init__(D, cfg)
        self.N_elite = int(cfg.elite_frac * cfg.N_samples)
        self.N_keep = int(self.N_elite * cfg.keep_frac)
        # small diagonal regularization for covariance
        self.Id = np.diag(np.full(self.D, cfg.std_incr))
        self.reg_cov = cfg.std_incr > 0.
        
        self.first_it = True
        self.samples = np.zeros((cfg.N_samples, D))

    def get_samples(self) -> Array:
        """
        Get samples from distribution parametrized
        by the current state.
        """
        diag = np.diag(self.state.cov)
        self.collapsed_dim = diag < self.cfg.min_std_collapsed  # boolean mask
        self.dim_to_sample = ~self.collapsed_dim
        self.dim_to_sample[self.n_dim:] = False

        N = 0 if self.first_it else self.N_keep
        sampled = self.sampler.sample(
            mean=self.state.mean[self.dim_to_sample],
            cov=self.state.cov[self.dim_to_sample, :][:, self.dim_to_sample],
        )
        if not np.isfinite(sampled).all():
            n_sampled_dim = int(np.sum(self.dim_to_sample))
            raise FloatingPointError(
                "CEM.get_samples: drawn samples are non-finite (NaN/Inf). "
                "This means state.cov has become singular/non-positive-"
                f"definite (N_elite={self.N_elite} elite samples were used "
                f"to estimate a {n_sampled_dim}-dim covariance, with no "
                "regularization since cfg.std_incr=0.0). Increase "
                "elite_frac/N_samples, or set cfg.std_incr > 0, to avoid a "
                "degenerate covariance estimate. (Failing loudly here "
                "instead of letting NaN control knots silently propagate "
                "into the rollout, where they surface as a confusing "
                "downstream crash, e.g. PchipInterpolator 'y must contain "
                "only finite values'.)"
            )
        self.samples[N:, self.dim_to_sample] = sampled[N:]

        if np.any(self.collapsed_dim):
            self.samples[:, self.collapsed_dim] = self.state.mean[None, self.collapsed_dim]

        return self.samples
    
    def get_elites(self, samples: Array, costs: Array) -> Tuple[Array, IntArray]:
        """
        Returns (elites, elite_idx).

        Samples with non-finite cost (NaN/Inf, e.g. from a diverged physics
        rollout) are excluded before elite selection: np.argpartition's
        behavior is documented as undefined in the presence of NaN, and we
        never want a diverged sample to be treated as elite/best.
        If every sample this iteration is non-finite, returns (None, None).
        """
        finite_mask = np.isfinite(costs)
        n_finite = int(np.count_nonzero(finite_mask))
        if n_finite < costs.shape[0]:
            print(
                f"[CEM] Warning: {costs.shape[0] - n_finite}/{costs.shape[0]} "
                "sample costs are non-finite (NaN/Inf) this iteration; "
                "excluding them from elite selection."
            )
        if n_finite == 0:
            return None, None

        finite_idx = np.flatnonzero(finite_mask)
        n_elite = min(self.N_elite, n_finite)
        if n_elite >= n_finite:
            elites_idx = finite_idx
        else:
            finite_costs = costs[finite_idx]
            part = np.argpartition(finite_costs, n_elite)[:n_elite]
            elites_idx = finite_idx[part]
        elites_idx = elites_idx[np.argsort(costs[elites_idx])]

        elites = samples[elites_idx]
        return elites, elites_idx
    
    def update_distrib_param(self, state: SolverState, elites: Array) -> None:
        mean, cov = self.sampler.estimate_params(elites)
        if self.reg_cov:
            cov += self.Id

        # Update state params with exponential smoothing
        s = slice(0, self.n_dim)
        state.mean[s] += self.cfg.alpha_mean * (mean[s] - state.mean[s])
        state.cov[s, s] += self.cfg.alpha_cov * (cov[s, s] - state.cov[s, s])

    def update(self,
               samples: Array,
               costs: Array,
               ) -> None:
        """
        Update the solver state from elite samples.
        """
        elites, elites_idx = self.get_elites(samples, costs)
        if elites is None:
            # Every sample this iteration had non-finite cost (total
            # divergence). Keep the previous distribution and best-so-far
            # untouched rather than corrupting them with NaN/Inf.
            print(
                "[CEM] Warning: all samples had non-finite cost this "
                "iteration; skipping distribution/best update."
            )
            self.first_it = False
            return

        self.update_distrib_param(self.state, elites)
        if self.N_keep > 0:
            self.samples[:self.N_keep] = elites[:self.N_keep]

        arg_min = elites_idx[0]
        best = samples[arg_min]
        min_cost = costs[arg_min]
        self.update_min_cost_best(self.state, min_cost, best, best_id=arg_min)

        self.first_it = False