import numpy as np
import pandas as pd

from hockey_predictor.features import TeamStats, build_dataset
from hockey_predictor.models import HockeyModel, outcome_probs, regulation_goals


def synthetic_games(n=1200, seed=0):
    """Fake league where a hidden 'strength' drives results, to check the pipeline learns it."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(16)]
    strength = dict(zip(teams, rng.normal(0, 0.35, 16)))
    rows, day = [], pd.Timestamp("2022-10-10")
    for g in range(n):
        if g % 8 == 0:
            day += pd.Timedelta(days=1)
        h, a = rng.choice(teams, 2, replace=False)
        lh = np.exp(np.log(3.0) + 0.1 + strength[h] - strength[a] * 0.5)
        la = np.exp(np.log(3.0) + strength[a] - strength[h] * 0.5)
        hg, ag = rng.poisson(lh), rng.poisson(la)
        ending = "REG"
        if hg == ag:
            ending = rng.choice(["OT", "SO"])
            if rng.random() < 0.5:
                hg += 1
            else:
                ag += 1
        rows.append(dict(game_id=g, season=20222023, date=day, home=h, away=a, home_goals=hg,
                         away_goals=ag, home_sog=30 + 3 * (hg - ag), away_sog=30 - 3 * (hg - ag), ending=ending))
    return pd.DataFrame(rows)


def test_features_do_not_peek_at_the_future():
    games = synthetic_games(400)
    ds1, _ = build_dataset(games)
    altered = games.copy()
    altered.loc[300:, ["home_goals", "away_goals"]] = [9, 0]     # rewrite the future
    ds2, _ = build_dataset(altered)
    cols = [c for c in ds1.columns if c.startswith(("h_", "a_"))]
    pd.testing.assert_frame_equal(ds1.loc[:300, cols], ds2.loc[:300, cols])


def test_outcome_probabilities_sum_to_one_and_ties_inflate():
    p = outcome_probs(np.array([3.0, 2.2]), np.array([2.5, 3.4]), phi=1.2)
    assert np.allclose(p["home_reg"] + p["tie"] + p["away_reg"], 1)
    assert (outcome_probs(np.array([3.0]), np.array([3.0]), 1.3)["tie"] > outcome_probs(np.array([3.0]), np.array([3.0]), 1.0)["tie"]).all()


def test_regulation_goals_strip_ot_goal():
    df = pd.DataFrame({"home_goals": [4, 3, 2], "away_goals": [2, 2, 3], "ending": ["REG", "OT", "SO"]})
    h, a = regulation_goals(df)
    assert list(h) == [4, 2, 2] and list(a) == [2, 2, 2]


def test_model_learns_strength_and_beats_coin_flip():
    ds, _ = build_dataset(synthetic_games(3000))
    ds = ds[ds["min_games"] >= 10]
    train, test = ds.iloc[:2200], ds.iloc[2200:]
    m = HockeyModel().fit(train)
    p = m.predict_frame(test)
    acc = ((p["p_blend"].to_numpy() >= 0.5) == (test["home_win"].to_numpy() == 1)).mean()
    assert acc > 0.55
    assert p["p_blend"].between(0, 1).all()


def test_matchup_stronger_team_favoured_and_grid_normalised():
    ds, _ = build_dataset(synthetic_games(3000))
    m = HockeyModel().fit(ds[ds["min_games"] >= 10])
    strong = TeamStats(elo=1580, gf=3.6, ga=2.5, sf=33, sa=27, pts=0.68)
    weak = TeamStats(elo=1420, gf=2.5, ga=3.6, sf=27, sa=33, pts=0.40)
    p = m.predict_matchup(strong, weak)
    assert p.p_home > 0.6 and abs(p.grid.sum() - 1) < 1e-9
    assert m.predict_matchup(weak, strong).p_home < p.p_home
