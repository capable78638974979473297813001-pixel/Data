"""Models: ridge regression for expected goals + Poisson scoreline simulation,
logistic regression for win probability, and a blend of the two."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import TeamStats, matchup_diff, matchup_raw, stats_from_row

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

MAX_GOALS = 14
MIN_LAMBDA = 0.8


def regulation_goals(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Goals at the end of 60 minutes. An OT/SO game was tied after regulation, and the winner's
    final score includes the OT/shootout goal, so both teams' regulation total is the loser's score."""
    ot = (df["ending"] != "REG").to_numpy()
    lo = np.minimum(df["home_goals"], df["away_goals"]).to_numpy()
    return (np.where(ot, lo, df["home_goals"]).astype(float),
            np.where(ot, lo, df["away_goals"]).astype(float))


def _design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    hs, as_ = stats_from_row(df, "h"), stats_from_row(df, "a")
    xd = np.array([matchup_diff(h, a) for h, a in zip(hs, as_)])
    xr = np.array([matchup_raw(h, a) for h, a in zip(hs, as_)])
    return xd, xr


def outcome_probs(lh: np.ndarray, la: np.ndarray, phi: float = 1.0) -> dict[str, np.ndarray]:
    """Vectorised regulation outcome probabilities from independent Poisson goal rates,
    with the tie diagonal inflated by ``phi`` (hockey has more 60-minute ties than independent Poisson)."""
    k = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(k[None, :], np.asarray(lh)[:, None])
    pa = poisson.pmf(k[None, :], np.asarray(la)[:, None])
    cdf_a = np.cumsum(pa, axis=1)
    below = np.concatenate([np.zeros((len(pa), 1)), cdf_a[:, :-1]], axis=1)   # P(away < i)
    win = (ph * below).sum(1)
    tie = (ph * pa).sum(1)
    norm = 1.0 + (phi - 1.0) * tie
    tie_n = phi * tie / norm
    win_n = win / norm
    loss_n = 1.0 - win_n - tie_n
    return {"home_reg": win_n, "tie": tie_n, "away_reg": loss_n}


@dataclass
class Prediction:
    home: str
    away: str
    p_home: float                 # blended win probability (the headline number)
    p_logit: float                # logistic-regression-only
    p_sim: float                  # score-simulation-only
    exp_home_goals: float
    exp_away_goals: float
    p_home_reg: float             # home wins inside 60 minutes
    p_away_reg: float
    p_ot: float                   # tied after 60 -> OT / shootout
    p_home_ot: float              # home wins the OT/SO, given it goes there
    grid: np.ndarray = field(repr=False, default=None)   # regulation scoreline probabilities [home, away]

    @property
    def winner(self) -> str:
        return self.home if self.p_home >= 0.5 else self.away

    @property
    def confidence(self) -> float:
        return max(self.p_home, 1 - self.p_home)

    def top_scores(self, n: int = 5) -> list[tuple[int, int, float]]:
        idx = np.dstack(np.unravel_index(np.argsort(-self.grid, axis=None), self.grid.shape))[0][:n]
        return [(int(i), int(j), float(self.grid[i, j])) for i, j in idx]

    def total_over(self, line: float) -> float:
        i, j = np.indices(self.grid.shape)
        return float(self.grid[(i + j) > line].sum())

    def margin_prob(self, home_margin: int) -> float:
        """P(home wins regulation by at least ``home_margin``) (use negative for away-by-N)."""
        i, j = np.indices(self.grid.shape)
        return float(self.grid[(i - j) >= home_margin].sum())


class HockeyModel:
    def __init__(self, blend: float = 0.5) -> None:
        self.blend = blend   # weight on the logistic model; the rest goes to the score simulation

    # ---------------------------------------------------------------- fit
    def fit(self, train: pd.DataFrame) -> "HockeyModel":
        train = train.sort_values("date")
        xd, xr = _design(train)
        y = train["home_win"].to_numpy()
        hg, ag = regulation_goals(train)

        self.logit = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=TimeSeriesSplit(5),
                                 scoring="neg_log_loss", max_iter=2000),
        ).fit(xd, y)

        alphas = np.logspace(-1, 4, 16)
        self.reg_home = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)).fit(xr, hg)
        self.reg_away = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)).fit(xr, ag)

        # tie inflation: pick phi so predicted regulation-tie rate matches the observed one
        lh, la = self._lambdas(xr)
        observed_tie = float((train["ending"] != "REG").mean())
        lo, hi = 0.5, 3.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if outcome_probs(lh, la, mid)["tie"].mean() < observed_tie:
                lo = mid
            else:
                hi = mid
        self.phi = (lo + hi) / 2

        # who wins once it goes to OT/SO: tiny skill edge on top of home ice
        ot = (train["ending"] != "REG").to_numpy()
        self.ot_model = LogisticRegression(C=1.0).fit(xd[ot][:, [0]] / 100.0, y[ot])
        return self

    def _lambdas(self, xr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (np.clip(self.reg_home.predict(xr), MIN_LAMBDA, None),
                np.clip(self.reg_away.predict(xr), MIN_LAMBDA, None))

    # ---------------------------------------------------------- batch predict
    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        xd, xr = _design(df)
        return self._predict_matrices(xd, xr)

    def _predict_matrices(self, xd: np.ndarray, xr: np.ndarray) -> pd.DataFrame:
        lh, la = self._lambdas(xr)
        reg = outcome_probs(lh, la, self.phi)
        p_ot_home = self.ot_model.predict_proba(xd[:, [0]] / 100.0)[:, 1]
        p_sim = reg["home_reg"] + reg["tie"] * p_ot_home
        p_logit = self.logit.predict_proba(xd)[:, 1]
        return pd.DataFrame({
            "p_logit": p_logit, "p_sim": p_sim,
            "p_blend": self.blend * p_logit + (1 - self.blend) * p_sim,
            "exp_home_goals": lh, "exp_away_goals": la,
            "p_home_reg": reg["home_reg"], "p_away_reg": reg["away_reg"],
            "p_ot": reg["tie"], "p_home_ot": p_ot_home,
        })

    # ------------------------------------------------------ single matchup
    def predict_matchup(self, home: TeamStats, away: TeamStats, home_name: str = "HOME",
                        away_name: str = "AWAY") -> Prediction:
        p = self._predict_matrices(matchup_diff(home, away)[None, :], matchup_raw(home, away)[None, :]).iloc[0]
        k = np.arange(MAX_GOALS + 1)
        grid = np.outer(poisson.pmf(k, p.exp_home_goals), poisson.pmf(k, p.exp_away_goals))
        grid[k, k] *= self.phi
        grid /= grid.sum()
        return Prediction(home_name, away_name, float(p.p_blend), float(p.p_logit), float(p.p_sim),
                          float(p.exp_home_goals), float(p.exp_away_goals), float(p.p_home_reg),
                          float(p.p_away_reg), float(p.p_ot), float(p.p_home_ot), grid)
