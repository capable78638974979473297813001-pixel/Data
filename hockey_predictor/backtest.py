"""Walk-forward backtest: for every test game the model has only ever seen earlier games."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_games
from .features import ELO_HOME_ADV, MIN_GAMES, build_dataset
from .models import HockeyModel

RESULTS = Path(__file__).resolve().parent.parent / "results"


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p >= 0.5) == (y == 1)))


def walk_forward(ds: pd.DataFrame, test_seasons: list[int], blend: float = 0.5) -> pd.DataFrame:
    """Retrain at the start of every calendar month of each test season (expanding window)."""
    ds = ds[ds["min_games"] >= MIN_GAMES].copy()
    ds["month"] = ds["date"].dt.to_period("M")
    out = []
    for season in test_seasons:
        test = ds[ds["season"] == season]
        for month, chunk in test.groupby("month"):
            train = ds[ds["date"] < chunk["date"].min()]
            model = HockeyModel(blend=blend).fit(train)
            preds = model.predict_frame(chunk)
            preds.index = chunk.index
            out.append(pd.concat([chunk, preds], axis=1).assign(
                base_home_rate=float(train["home_win"].mean())))
        print(f"  season {season}: {len(test)} games predicted", flush=True)
    res = pd.concat(out)
    res["p_elo"] = 1 / (1 + 10 ** (-(res["h_elo"] - res["a_elo"] + ELO_HOME_ADV) / 400))
    res["p_pts"] = np.clip(0.54 + 0.9 * (res["h_pts"] - res["a_pts"]), 0.05, 0.95)  # naive 'better record' rule
    return res


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    y = res["home_win"].to_numpy()
    rows = []
    for name, col in [
        ("Always pick home team", "base_home_rate"),
        ("Better points % (naive)", "p_pts"),
        ("Elo only", "p_elo"),
        ("Logistic regression", "p_logit"),
        ("Ridge goals + Poisson score sim", "p_sim"),
        ("Blend (headline model)", "p_blend"),
    ]:
        p = res[col].to_numpy()
        rows.append({"model": name, "accuracy": accuracy(y, p), "log_loss": log_loss(y, p), "brier": brier(y, p)})
    return pd.DataFrame(rows)


def confidence_table(res: pd.DataFrame, col: str = "p_blend") -> pd.DataFrame:
    y = res["home_win"].to_numpy()
    p = res[col].to_numpy()
    conf = np.maximum(p, 1 - p)
    correct = (p >= 0.5) == (y == 1)
    rows = []
    for lo in (0.50, 0.55, 0.60, 0.65, 0.70):
        m = conf >= lo
        if m.sum():
            rows.append({"model_confidence_at_least": f"{lo:.0%}", "games": int(m.sum()),
                         "share_of_games": m.mean(), "accuracy": correct[m].mean(),
                         "avg_stated_confidence": conf[m].mean()})
    return pd.DataFrame(rows)


def calibration_table(res: pd.DataFrame, col: str = "p_blend", bins: int = 8) -> pd.DataFrame:
    q = pd.qcut(res[col], bins, duplicates="drop")
    g = res.groupby(q, observed=True).agg(games=("home_win", "size"), predicted=(col, "mean"), actual=("home_win", "mean"))
    g.index = [f"{i.left:.2f}-{i.right:.2f}" for i in g.index]
    return g.rename_axis("predicted_home_win_prob_bin")


def run(test_start: int = 2022, blend: float = 0.5, write: bool = True) -> pd.DataFrame:
    games = load_games()
    ds, _ = build_dataset(games)
    last_full = int(ds["season"].max() // 10000)
    seasons = [int(f"{y}{y + 1}") for y in range(test_start, last_full + 1)]
    seasons = [s for s in seasons if (ds["season"] == s).sum() > 600]   # skip a barely-started season
    print(f"walk-forward over {len(seasons)} seasons: {seasons}")
    res = walk_forward(ds, seasons, blend)

    summ, conf, cal = summarize(res), confidence_table(res), calibration_table(res)
    se = np.sqrt(0.25 / len(res))   # ~standard error of an accuracy near 50-60%
    report = (
        f"# Walk-forward backtest\n\n"
        f"Seasons: {', '.join(f'{s // 10000}-{str(s % 10000)[2:]}' for s in seasons)} - **{len(res)} games**, "
        f"every prediction made using only earlier games (models retrained monthly).\n\n"
        f"Accuracy standard error is about +/-{se:.1%}, so differences smaller than ~1-2 points are noise.\n\n"
        f"## Models vs. baselines\n\n{summ.to_markdown(index=False, floatfmt='.4f')}\n\n"
        f"## Accuracy when the model is more confident\n\n{conf.to_markdown(index=False, floatfmt='.3f')}\n\n"
        f"## Calibration (headline model)\n\n{cal.to_markdown(floatfmt='.3f')}\n\n"
        f"## Per season (headline model)\n\n"
        f"{res.groupby('season').apply(lambda d: pd.Series({'games': len(d), 'accuracy': accuracy(d['home_win'].to_numpy(), d['p_blend'].to_numpy())}), include_groups=False).to_markdown(floatfmt='.3f')}\n"
    )
    print(report)
    if write:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "backtest.md").write_text(report)
        keep = ["game_id", "date", "home", "away", "home_goals", "away_goals", "home_win",
                "p_logit", "p_sim", "p_blend", "exp_home_goals", "exp_away_goals"]
        res[keep].to_csv(RESULTS / "backtest_predictions.csv", index=False)
    return res


if __name__ == "__main__":
    run()
