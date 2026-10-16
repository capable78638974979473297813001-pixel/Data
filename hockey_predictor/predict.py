"""High-level entry point: train on everything we have, then predict any matchup."""
from __future__ import annotations

import pandas as pd
import requests

from .data import API, canon, load_games
from .features import MIN_GAMES, STAT_NAMES, League, TeamStats, build_dataset
from .models import HockeyModel, Prediction


class Predictor:
    def __init__(self, model: HockeyModel, league: League, dataset: pd.DataFrame) -> None:
        self.model, self.league, self.dataset = model, league, dataset

    @classmethod
    def from_data(cls, blend: float = 0.5) -> "Predictor":
        games = load_games()
        ds, league = build_dataset(games)
        model = HockeyModel(blend=blend).fit(ds[ds["min_games"] >= MIN_GAMES])
        return cls(model, league, ds)

    @property
    def teams(self) -> list[str]:
        return sorted(self.league.elo)

    def team_stats(self, team: str | None, date: pd.Timestamp | None = None, **overrides: float) -> TeamStats:
        """Current stats for ``team`` (league-average if unknown / None), with any field overridden."""
        bad = set(overrides) - set(STAT_NAMES)
        if bad:
            raise ValueError(f"unknown stat(s) {sorted(bad)}; choose from {STAT_NAMES}")
        date = date if date is not None else pd.Timestamp.today().normalize()
        code = canon(team.upper()) if team else None
        base = self.league.snapshot(code, date) if code in self.league.elo else self.league.snapshot("__league__", None)
        d = base.as_dict()
        d.update({k: float(v) for k, v in overrides.items() if v is not None})
        return TeamStats(**d)

    def matchup(self, home: str, away: str, date: pd.Timestamp | None = None,
                home_overrides: dict | None = None, away_overrides: dict | None = None) -> Prediction:
        h = self.team_stats(home, date, **(home_overrides or {}))
        a = self.team_stats(away, date, **(away_overrides or {}))
        return self.model.predict_matchup(h, a, home.upper(), away.upper())

    def slate(self, date: pd.Timestamp) -> list[Prediction]:
        """Predict every NHL game scheduled on ``date`` (fetched live from the NHL API)."""
        r = requests.get(f"{API}/schedule/{date:%Y-%m-%d}", timeout=30)
        r.raise_for_status()
        out = []
        for day in r.json().get("gameWeek", []):
            if day["date"] != f"{date:%Y-%m-%d}":
                continue
            for g in day["games"]:
                if g.get("gameType") in (2, 3):
                    out.append(self.matchup(canon(g["homeTeam"]["abbrev"]), canon(g["awayTeam"]["abbrev"]), date))
        return out


def format_prediction(p: Prediction) -> str:
    lines = [
        f"{p.away} @ {p.home}",
        f"  Predicted winner : {p.winner}  ({p.confidence:.1%})",
        f"  Win probability  : {p.home} {p.p_home:.1%}   {p.away} {1 - p.p_home:.1%}",
        f"                     (logistic {p.p_logit:.1%} | score-sim {p.p_sim:.1%} for {p.home})",
        f"  Expected goals   : {p.home} {p.exp_home_goals:.2f}   {p.away} {p.exp_away_goals:.2f}"
        f"   (total {p.exp_home_goals + p.exp_away_goals:.2f})",
        f"  60-min result    : {p.home} win {p.p_home_reg:.1%} | {p.away} win {p.p_away_reg:.1%} | "
        f"OT/SO {p.p_ot:.1%} ({p.home} takes it {p.p_home_ot:.0%})",
        "  Likeliest 60-min scores (home-away): " + ", ".join(
            f"{h}-{a} ({pr:.1%})" for h, a, pr in p.top_scores(5)),
        f"  Total goals (60 min) over 5.5: {p.total_over(5.5):.1%} | over 6.5: {p.total_over(6.5):.1%}",
        f"  Wins by 2+ in 60 min: {p.home} {p.margin_prob(2):.1%} | {p.away} {_away_by(p, 2):.1%}",
    ]
    return "\n".join(lines)


def _away_by(p: Prediction, n: int) -> float:
    import numpy as np
    i, j = np.indices(p.grid.shape)
    return float(p.grid[(j - i) >= n].sum())
