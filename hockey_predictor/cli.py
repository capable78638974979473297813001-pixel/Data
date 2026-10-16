"""Command line: ``python -m hockey_predictor <command>``."""
from __future__ import annotations

import argparse

import pandas as pd

from .features import STAT_NAMES


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="hockey_predictor", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("update", help="download new finished NHL games into data/games.csv")

    bt = sub.add_parser("backtest", help="walk-forward accuracy test on past seasons")
    bt.add_argument("--test-start", type=int, default=2022, help="first test season's start year")

    pr = sub.add_parser("predict", help="predict one game")
    pr.add_argument("--home", required=True, help="home team code (e.g. BOS) or any label for a custom team")
    pr.add_argument("--away", required=True)
    pr.add_argument("--date", help="game date YYYY-MM-DD (used for rest days); default today")
    for side in ("home", "away"):
        for stat in STAT_NAMES:
            pr.add_argument(f"--{side}-{stat}", type=float, help=f"override {side} {stat}")

    sl = sub.add_parser("slate", help="predict every game on a date (default today) from the live NHL schedule")
    sl.add_argument("--date")

    sub.add_parser("teams", help="list teams and their current Elo")

    args = ap.parse_args(argv)

    if args.cmd == "update":
        from .data import update
        df = update()
        print(f"{len(df)} games cached")
        return
    if args.cmd == "backtest":
        from .backtest import run
        run(test_start=args.test_start)
        return

    from .predict import Predictor, format_prediction
    pred = Predictor.from_data()

    if args.cmd == "teams":
        for t in sorted(pred.league.elo, key=pred.league.elo.get, reverse=True):
            s = pred.team_stats(t)
            print(f"{t}  elo {s.elo:6.0f}  GF {s.gf:.2f}  GA {s.ga:.2f}  SF {s.sf:.1f}  SA {s.sa:.1f}  pts% {s.pts:.3f}")
    elif args.cmd == "predict":
        date = pd.Timestamp(args.date) if args.date else None
        ov = {side: {s: getattr(args, f"{side}_{s}") for s in STAT_NAMES if getattr(args, f"{side}_{s}") is not None}
              for side in ("home", "away")}
        print(format_prediction(pred.matchup(args.home, args.away, date, ov["home"], ov["away"])))
    elif args.cmd == "slate":
        date = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()
        games = pred.slate(date)
        if not games:
            print(f"No games found on {date:%Y-%m-%d}")
        for g in games:
            print(format_prediction(g), end="\n\n")


if __name__ == "__main__":
    main()
