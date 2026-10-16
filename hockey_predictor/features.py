"""Leakage-free pre-game team ratings and feature vectors.

Every game's features are computed from games played strictly *before* it, by walking the
schedule in date order and updating team state only after the features are recorded.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, fields

import numpy as np
import pandas as pd

ELO_START = 1500.0
ELO_K = 6.0
ELO_HOME_ADV = 33.0          # Elo points; fitted offline, the models re-learn home edge anyway
ELO_SEASON_REGRESS = 1 / 3   # pull toward 1500 each new season
WINDOW = 20                  # games in the rolling-form window
SHRINK = 6                   # pseudo-games of league average mixed into the rolling window
MIN_GAMES = 10               # games of history required before a game is used for train/eval


@dataclass
class TeamStats:
    """Everything the model needs to know about one team going into a game.

    All rolling numbers are per-game averages over the last ``WINDOW`` games (shrunk toward
    the league mean), so you can type in your own numbers for any team.
    """
    elo: float = ELO_START
    gf: float = 3.05      # goals for per game
    ga: float = 3.05      # goals against per game
    sf: float = 29.5      # shots on goal for per game
    sa: float = 29.5      # shots on goal against per game
    pts: float = 0.55     # standings points % (2 pts for W, 1 for OTL/SOL) over the window
    rest: float = 3.0     # days since last game (1 == back-to-back)

    def as_dict(self) -> dict:
        return asdict(self)


STAT_NAMES = [f.name for f in fields(TeamStats)]


def feature_names_diff() -> list[str]:
    return ["elo", "gf", "ga", "sf", "sa", "pts", "b2b", "rest"]


def matchup_diff(h: TeamStats, a: TeamStats) -> np.ndarray:
    """Symmetric 'home minus away' features for win probability (home edge = intercept)."""
    return np.array([
        h.elo - a.elo,
        h.gf - a.gf,
        h.ga - a.ga,
        h.sf - a.sf,
        h.sa - a.sa,
        h.pts - a.pts,
        float(h.rest <= 1) - float(a.rest <= 1),
        min(h.rest, 7) - min(a.rest, 7),
    ])


def matchup_raw(h: TeamStats, a: TeamStats) -> np.ndarray:
    """Both teams' raw numbers for the goals regressions (offense vs. opposing defense)."""
    return np.array([
        h.elo - a.elo,
        h.gf, h.ga, h.sf, h.sa,
        a.gf, a.ga, a.sf, a.sa,
        h.pts - a.pts,
        float(h.rest <= 1), float(a.rest <= 1),
        min(h.rest, 7), min(a.rest, 7),
    ])


class League:
    """Running team ratings; ``snapshot`` reads state, ``update`` folds in a finished game."""

    def __init__(self) -> None:
        self.elo: dict[str, float] = {}
        self.hist: dict[str, deque] = {}     # recent (gf, ga, sf, sa, pts) tuples
        self.last_date: dict[str, pd.Timestamp] = {}
        self.n_games: dict[str, int] = {}
        self.season: int | None = None
        # league-wide running averages (for shrinkage), seeded with sensible NHL values
        self._tot = np.array([3.05, 29.5, 0.55]) * 50
        self._n = 50

    # -- league means -------------------------------------------------------------
    @property
    def mean_goals(self) -> float: return self._tot[0] / self._n
    @property
    def mean_shots(self) -> float: return self._tot[1] / self._n
    @property
    def mean_pts(self) -> float: return self._tot[2] / self._n

    def _new_season(self, season: int) -> None:
        for t in self.elo:
            self.elo[t] = ELO_START + (self.elo[t] - ELO_START) * (1 - ELO_SEASON_REGRESS)
        self.season = season

    # -- read ---------------------------------------------------------------------
    def snapshot(self, team: str, date: pd.Timestamp | None = None) -> TeamStats:
        h = self.hist.get(team)
        g0, s0, p0 = self.mean_goals, self.mean_shots, self.mean_pts
        if h:
            arr = np.array(h)
            n = len(arr)
            gf = (arr[:, 0].sum() + SHRINK * g0) / (n + SHRINK)
            ga = (arr[:, 1].sum() + SHRINK * g0) / (n + SHRINK)
            sf = (arr[:, 2].sum() + SHRINK * s0) / (n + SHRINK)
            sa = (arr[:, 3].sum() + SHRINK * s0) / (n + SHRINK)
            pts = (arr[:, 4].sum() + SHRINK * p0) / (n + SHRINK)
        else:
            gf, ga, sf, sa, pts = g0, g0, s0, s0, p0
        rest = 3.0
        if date is not None and team in self.last_date:
            rest = float(max((date - self.last_date[team]).days, 1))
        return TeamStats(self.elo.get(team, ELO_START), gf, ga, sf, sa, pts, rest)

    # -- write --------------------------------------------------------------------
    def update(self, row) -> None:
        if self.season is None:
            self.season = row.season
        elif row.season != self.season:
            self._new_season(row.season)

        h, a = row.home, row.away
        hg, ag = row.home_goals, row.away_goals
        h_sog = row.home_sog if not pd.isna(row.home_sog) else self.mean_shots
        a_sog = row.away_sog if not pd.isna(row.away_sog) else self.mean_shots
        home_won = hg > ag
        ot = row.ending != "REG"
        h_pts = 1.0 if home_won else (0.5 if ot else 0.0)   # points% units (2 pts == 1.0)
        a_pts = 1.0 if not home_won else (0.5 if ot else 0.0)

        # Elo with margin-of-victory multiplier (FiveThirtyEight-style)
        eh, ea = self.elo.get(h, ELO_START), self.elo.get(a, ELO_START)
        dr = eh + ELO_HOME_ADV - ea
        exp_h = 1 / (1 + 10 ** (-dr / 400))
        margin = abs(hg - ag)
        winner_dr = dr if home_won else -dr
        mult = math.log(margin + 1) * (2.2 / (winner_dr * 0.001 + 2.2))
        delta = ELO_K * mult * ((1.0 if home_won else 0.0) - exp_h)
        self.elo[h], self.elo[a] = eh + delta, ea - delta

        self.hist.setdefault(h, deque(maxlen=WINDOW)).append((hg, ag, h_sog, a_sog, h_pts))
        self.hist.setdefault(a, deque(maxlen=WINDOW)).append((ag, hg, a_sog, h_sog, a_pts))
        self.last_date[h] = self.last_date[a] = row.date
        self.n_games[h] = self.n_games.get(h, 0) + 1
        self.n_games[a] = self.n_games.get(a, 0) + 1

        self._tot += np.array([(hg + ag) / 2, (h_sog + a_sog) / 2, (h_pts + a_pts) / 2])
        self._n += 1


def build_dataset(games: pd.DataFrame) -> tuple[pd.DataFrame, League]:
    """Walk the schedule in order; return one feature row per game and the final League state."""
    league = League()
    rows = []
    for row in games.itertuples(index=False):
        h = league.snapshot(row.home, row.date)
        a = league.snapshot(row.away, row.date)
        rec = {
            "game_id": row.game_id, "season": row.season, "date": row.date,
            "home": row.home, "away": row.away,
            "home_goals": row.home_goals, "away_goals": row.away_goals, "ending": row.ending,
            "home_win": int(row.home_goals > row.away_goals),
            "min_games": min(league.n_games.get(row.home, 0), league.n_games.get(row.away, 0)),
        }
        rec.update({f"h_{k}": v for k, v in h.as_dict().items()})
        rec.update({f"a_{k}": v for k, v in a.as_dict().items()})
        rows.append(rec)
        league.update(row)
    return pd.DataFrame(rows), league


def stats_from_row(df: pd.DataFrame, side: str) -> list[TeamStats]:
    cols = [f"{side}_{n}" for n in STAT_NAMES]
    return [TeamStats(*vals) for vals in df[cols].itertuples(index=False, name=None)]
