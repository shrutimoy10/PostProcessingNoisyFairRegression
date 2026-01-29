from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from scipy.optimize import nnls

import warnings

# import gurobipy
import cvxpy as cp
import numpy as np
import copy

# env = gurobipy.Env()
np.set_printoptions(linewidth=np.inf)

def iso_regression_linf(x):
  """Unweighted isotonic regression with L-infinity loss"""
  l_max = np.maximum.accumulate(x)
  r_min = np.minimum.accumulate(x[::-1])[::-1]
  return (l_max + r_min) / 2

def estimate_true_w(P, w_hat):
  w = cp.Variable(w_hat.shape[-1], name="w", nonneg=True)
  constraints = [cp.sum(w) == 1]
  obj = cp.Minimize(cp.norm(P @ w - w_hat, 2)**2 + (0.0 * cp.norm(w,2)))
  problem = cp.Problem(obj, constraints)
  
  problem.solve(enforce_dpp=True)
  
  corrected_w = w.value
  return corrected_w

def estimate_true_hist_by_group(Q, hist, w, w_hat):
  X = cp.Variable((hist.shape[1], hist.shape[0]), name="x", nonneg=True)
  constraints = []
  constraints.append(cp.sum(X, axis=0) == 1)
  # constraints.append(cp.sum(X, axis=1) == np.sum(hist.T, axis=1))
  constraints.append(X @ w == hist.T @ w_hat)
  obj = cp.Minimize(cp.norm(X @ Q.T - hist.T, 'fro')**2 + (0.0 * cp.norm(X,2)))
  problem = cp.Problem(obj, constraints)

  problem.solve(solver='CLARABEL', enforce_dpp=True)

  corrected_hist = X.value

  # print(f"sum : {np.sum(corrected_hist, axis=1)}")
  # assert 1 == 4
  return corrected_hist.T

class PrivateHDEFairPostProcessor:

  def fit(self,
          scores,
          groups,
          true_groups=None,
          alpha=0.0,
          bound=None,
          n_bins=10,
          eps=np.inf,
          noise=None, # Probability of protected attribute getting flipped
          noise_model=None,
          true_w=None,
          rng=None):

    if rng is None:
      rng = np.random.default_rng()
    self.rng_ = rng

    if not noise_model is None:
      self.P = noise_model

    if not noise is None:
      self.gamma = noise

    if not true_groups is None:
      self.true_groups = true_groups

    if not true_w is None:
      self.true_w = true_w

    self.alpha_ = alpha
    self.n_groups_ = int(1 + np.max(groups))

    if bound is None:
      warnings.warn(
          "Bound is not set, using min and max of scores, which violates differential privacy"
      )
      bound = (np.min(scores), np.max(scores))
    self.bound_ = bound


    # Creating the noise confusion matrix P
    # P_ii = 1 - gamma
    # P_ij = gamma / (self.n_groups_ - 1)
    P = (self.gamma / (self.n_groups_ - 1)) * np.ones((self.n_groups_ , self.n_groups_), dtype=float)
    np.fill_diagonal(P, 1 - self.gamma)
    _,s_p, _ = np.linalg.svd(P, full_matrices=False)
    # print(f"P:{P}")

    n = len(scores)
    # w = n / len(groups)
    self.n_bins_ = n_bins
    self.bin_width_ = (bound[1] - bound[0]) / n_bins

    """
     Output distributions of f, trained on noisy group information. 
          Pr(f(x) = t | A_hat = a), forall t, forall a
    """
    # Convert scores to bins (index)
    self.score_to_bin_ = lambda s: np.clip(
        np.floor((s - bound[0]) / self.bin_width_), 0, n_bins - 1).astype(int)
    bins = self.score_to_bin_(scores)

    # ========================TRUE====================================
    # Get histogram
    # hist_noisy = np.bincount(bins, minlength=n_bins).astype(float)
    # hist_noisy = np.astype(hist, dtype=float)
    # hist_noisy *= 1 / n
    """
    Each bin gives Pr(f(X) = t , G_hat = j).
    """
    hist_true = np.empty((self.n_groups_, n_bins), dtype=float)
    for a in range(self.n_groups_):
      mask = self.true_groups == a
      hist_true[a] = np.bincount(bins[mask], minlength=n_bins)
    hist_true *= 1 / n

    # Get group weight
    self.w_true = hist_true.sum(axis=1)
    # print(f"w_true : {self.w_true}")
    

    # Renormalize histogram
    hist_by_group_true = hist_true / self.w_true[:, None]
    cumsum_by_group_true = np.cumsum(hist_by_group_true,
                                axis=1)  # get partial sums ("cdf")
    for a in range(self.n_groups_):
      cumsum_by_group_true[a] = iso_regression_linf(
          cumsum_by_group_true[a])  # perform isotonic regression to get valid cdf
    cumsum_by_group_true = np.clip(cumsum_by_group_true, 0, 1)  # clip cdf to [0, 1]
    cumsum_by_group_true[:, -1] = 1  # set last value of "cdf" to 1

    self.hist_by_group_true = np.diff(cumsum_by_group_true, prepend=0, axis=1) # Pr(f(X) = t | G_hat = j), the noisy r_j's
    # print(f"hist_by_group_ column sums : {np.sum(self.hist_by_group_, axis=0)}")
    # print(f"Difference : {np.sum(np.sum(self.hist_by_group_true, axis=0) - np.sum(self.hist_by_group_, axis=0))}")
    # assert 1 == 3
    # ========================TRUE END================================


    # Get histogram
    # hist_noisy = np.bincount(bins, minlength=n_bins).astype(float)
    # hist_noisy = np.astype(hist, dtype=float)
    # hist_noisy *= 1 / n
    """
    Each bin gives Pr(f(X) = t , G_hat = j).
    """
    hist = np.empty((self.n_groups_, n_bins), dtype=float)
    for a in range(self.n_groups_):
      mask = groups == a
      hist[a] = np.bincount(bins[mask], minlength=n_bins)
    hist *= 1 / n
    # print(f"Diff in sums in joint distributions : {np.sum(np.abs(np.sum(hist_true, axis=0) - np.sum(hist,axis=0)))}")

    # Get group weight
    self.w_ = hist.sum(axis=1) #np.clip(hist.sum(axis=1), 1e-6, None) # Noisy weights

    

    # Renormalize histogram
    hist_by_group = hist / self.w_[:, None]
    # cumsum_by_group = np.cumsum(hist_by_group,
    #                             axis=1)  # get partial sums ("cdf")
    # for a in range(self.n_groups_):
    #   cumsum_by_group[a] = iso_regression_linf(
    #       cumsum_by_group[a])  # perform isotonic regression to get valid cdf
    # cumsum_by_group = np.clip(cumsum_by_group, 0, 1)  # clip cdf to [0, 1]
    # cumsum_by_group[:, -1] = 1  # set last value of "cdf" to 1

    self.hist_by_group_ = hist_by_group
    # self.hist_by_group_ = np.diff(cumsum_by_group, prepend=0, axis=1) # Pr(f(X) = t | G_hat = j), the noisy r_j's
    # print(f"hist_by_group_ column sums : {np.sum(self.hist_by_group_, axis=0)}")
    # print(f"Difference : {np.sum(np.sum(self.hist_by_group_true, axis=0) - np.sum(self.hist_by_group_, axis=0))}")
    # assert 1 == 3
    
    corrected_w = estimate_true_w(P, self.w_)
    diff = np.linalg.norm(corrected_w - self.w_true,1)
    # print(f"Self.w_ : {self.w_true}")
    # print(f"Corrected w : {corrected_w}")
    # print(f"P @ corrected_w - self.w_ : {np.linalg.norm(P @ corrected_w - self.w_)**2} ")
    # print(f"Max Difference of w : {np.sum(np.abs(corrected_w - self.w_true))}")
    
    
    #Estimating the Q matrix, where Q_ji = P_ji * (corrected_w[i] / self.w_[j])
    Q = np.diag(1 / self.w_) @ P @ np.diag(corrected_w)
    # _,s,_ = np.linalg.svd(Q, full_matrices=False)
    # print(f"Minimum Singular Value of Q : {np.min(s)}")


    corrected_hist = estimate_true_hist_by_group(Q, self.hist_by_group_, corrected_w, self.w_)
    # print(f"Column entries true : {self.hist_by_group_true[:,10]}, sum : {np.sum(self.hist_by_group_true[:,10])}")
    # print(f"Column entries noisy : {self.hist_by_group_[:,10]}, sum : {np.sum(self.hist_by_group_[:,10])}")
    # print(f"Column entries corrected : {corrected_hist[:,10]}, sum : {np.sum(corrected_hist[:,10])}")
    # print(f"Diff in sums of conditional probabilities : {np.sum(np.abs(np.sum(self.hist_by_group_true, axis=0) - np.sum(self.hist_by_group_, axis=0)))}")
    # print(f"Diff in joint distributions check : {np.linalg.norm((self.hist_by_group_.T @ self.w_) - (self.hist_by_group_true.T @ self.w_true))}")
    # print(f"Diff in joint distributions check (corrected) : {np.linalg.norm((corrected_hist.T @ corrected_w) - (self.hist_by_group_true.T @ self.w_true))}")
    # print(f"Diff in joint distributions check (corrected vs noisy) : {np.linalg.norm((corrected_hist.T @ corrected_w) - \
    #                                                             (self.hist_by_group_.T @ self.w_))}")

    # assert 1 == 3
    self.w_ = copy.deepcopy(corrected_w)
    self.hist_by_group_ = copy.deepcopy(corrected_hist)
    # print(f"Diff in sums of conditional probabilities (corrected) : {np.sum(np.abs(np.sum(self.hist_by_group_true, axis=0) - np.sum(self.hist_by_group_, axis=0)))}")
    
    # Estimating Pr(G_true = i ) using p_true = P^{-1} * self.w_
    # Augment row of 1's to P for probability constraint
    # Augment 1 to self.w_ for the same 

    # Get and solve fair post-processing LP
    problem = self.linprog_(self.hist_by_group_, alpha=alpha, w=self.w_)
    # problem.solve(solver='CLARABEL') # ..... if you do not have a Gurobi license
    # problem.solve(solver="GUROBI", env=env)  # ...if you have a Gurobi license
    try:
      problem.solve()
    except cp.SolverError:
      print("Solver Error : {cp.SolverError}")
      return
    except cp.ValueError:
      print("Value Error : {cp.ValueError}")
      return

    if not problem.status in  (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
      return 

    # Store value and target distributions
    self.score_ = problem.value / self.bin_width_**2
    self.q_by_group_ = problem.var_dict["q"].value

    # Store couplings and optimal transports
    self.gamma_by_group_ = np.clip(
        [problem.var_dict[f'gamma_{a}'].value for a in range(self.n_groups_)],
        0, 1)
    with np.errstate(invalid='ignore'):
      self.g_ = self.gamma_by_group_ / self.gamma_by_group_.sum(axis=-1,
                                                                keepdims=True)
    # Do nothing for unseen values
    for a in range(self.n_groups_):
      for i in range(n_bins):
        if np.isnan(self.g_[a][i][0]):
          self.g_[a][i] = 0
          self.g_[a][i][i] = 1

    return self, diff

  def linprog_(self, hist_by_group, alpha, w):

    alpha = cp.Parameter(value=alpha, name="alpha")
    n_bins = self.n_bins_ or hist_by_group.shape[1]
    n_groups = self.n_groups_ or hist_by_group.shape[0]

    # Variables are the probability mass of the couplings, the barycenter,
    # the output distributions, and slacks
    gamma_by_group = [
        cp.Variable((n_bins, n_bins), name=f"gamma_{a}", nonneg=True)
        for a in range(n_groups)
    ]
    barycenter = cp.Variable(n_bins, name="barycenter", nonneg=True)
    q = cp.Variable((n_groups, n_bins), name="q", nonneg=True)
    slack = cp.Variable((n_groups, n_bins), name="slack", nonneg=True)

    # Get l2 transportation costs
    costs = (np.arange(n_bins, dtype=float)[:, None] - np.arange(n_bins))**2
    M = cp.sum([
        cp.sum(cp.multiply(gamma_by_group[a], costs)) * w[a]
        for a in range(n_groups)
    ])

    # Adding Entropic regularization
    T = cp.sum([
          cp.sum(cp.entr(gamma_by_group[a]))
          for a in range(n_groups)
      ])

    cost = M - 0.0 * T
    #Regularization ends

    # Build constraints
    constraints = []

    # \sum_{s'} \gamma_{a, s, s'} = p_{a, s} for all a
    for a in range(self.n_groups_):
      constraints.append(cp.sum(gamma_by_group[a], axis=1) == hist_by_group[a])

    # \sum_s \gamma_{a, s, s'} = q_{a, s'}
    for a in range(self.n_groups_):
      constraints.append(cp.sum(gamma_by_group[a], axis=0) == q[a])

    # KS distance
    # | \sum_{s' <= s} (q_{a, s'} - barycenter_{s'}) | <= \xi_{a, s}
    for a in range(self.n_groups_):
      constraints.append(cp.abs(cp.cumsum(q[a] - barycenter)) <= slack[a])
    # \xi_{a, y} <= \alpha / 2
    constraints.append(slack <= alpha / 2)

    # # TV distance
    # # | q_{a, s} - barycenter_{s} | <= \xi_{a, s}
    # for a in range(self.n_groups_):
    #   constraints.append(cp.abs(q[a] - barycenter) <= slack[a])
    # # \sum_{s} \xi_{a, s} <= \alpha / 2
    # constraints.append(cp.sum(slack, axis=1) <= alpha / 2)

    return cp.Problem(cp.Minimize(cost), constraints)

  def predict(self, scores, groups):
    # Convert scores to bins (index)
    bins = self.score_to_bin_(scores)

    # Randomly reassign bins according to the optimal transports
    new_bins = np.empty_like(bins)
    for a in np.unique(groups):
      for i in np.unique(bins[groups == a]):
        mask = (bins == i) & (groups == a)
        new_bins[mask] = self.rng_.choice(self.n_bins_,
                                          size=np.sum(mask),
                                          p=self.g_[a][i])

    return new_bins * self.bin_width_ + self.bound_[0] + self.bin_width_ / 2


class WassersteinBarycenterFairPostProcessor:
  """
  Python reimplementation of https://github.com/lucaoneto/NIPS2020_Fairness
  """

  def fit(self, scores, groups, eps=None, rng=None):

    if rng is None:
      rng = np.random.default_rng()
    self.rng_ = rng

    self.n_groups_ = int(1 + np.max(groups))
    self.w_ = np.bincount(groups, minlength=self.n_groups_) / len(groups)

    if eps is None:
      eps = np.finfo(scores.dtype).eps
    self.eps_ = eps
    jitter = self.rng_.normal(scale=self.eps_, size=len(scores))
    scores = scores + jitter

    self.s0_by_group_ = []
    self.s1_by_group_ = []

    for a in range(self.n_groups_):
      mask = groups == a
      s = scores[mask]

      # Shuffle and split the scores in half
      s = self.rng_.permutation(s)
      s0, s1 = np.array_split(s, 2)

      # Sort and store the scores
      s0 = np.sort(s0)
      s1 = np.sort(s1)
      self.s0_by_group_.append(s0)
      self.s1_by_group_.append(s1)

    return self

  def predict(self, scores, groups):

    jitter = self.rng_.normal(scale=self.eps_, size=len(scores))
    scores = scores + jitter

    s1_sizes = np.array([len(s1) for s1 in self.s1_by_group_])
    new_scores = np.empty_like(scores)

    for a in np.unique(groups):
      mask = groups == a
      s = scores[mask]

      # Get percentile of scores in s0
      k = np.searchsorted(self.s0_by_group_[a], s) / len(self.s0_by_group_[a])

      # Get scores at the same percentile in s1 for all groups
      idx = np.clip(np.ceil(k * s1_sizes[:, None]).astype(int) - 1, 0, None)
      y = [self.s1_by_group_[b][idx[b]] for b in range(self.n_groups_)]

      # Take weighted average
      new_scores[mask] = np.tensordot(self.w_, y, axes=1)

    return new_scores
