# NHL game predictor

Give it two teams (or your own stats) and it predicts the winner, the win probability, the expected
score, and the probability of every scoreline. It's trained on real NHL results from 2018-19 onward
(about 9,900 games) and tested by predicting games it had never seen.

## Quick start

```bash
pip install -r requirements.txt

python -m hockey_predictor predict --home COL --away BOS          # two NHL teams
python -m hockey_predictor slate                                  # every game today (live schedule)
python -m hockey_predictor teams                                  # current Elo / form for all teams

# plug in your own stats for either side (anything you leave out defaults to the team's current
# numbers, or league average for a made-up team)
python -m hockey_predictor predict --home A --away B \
    --home-elo 1560 --home-gf 3.5 --home-ga 2.6 --home-sf 32 --home-sa 27 --home-pts 0.66 --away-rest 1

python -m hockey_predictor update       # download newly finished games into data/games.csv
python -m hockey_predictor backtest     # re-run the accuracy test
pytest                                  # unit tests
```

Stats you can override per team: `elo`, `gf` / `ga` (goals for / against per game), `sf` / `sa`
(shots on goal for / against per game), `pts` (points %), `rest` (days since last game, 1 = back-to-back).
Rolling numbers are over the team's last 20 games.

## How it works

1. **Ratings.** Walking through games in date order, each game gets features computed only from games
   before it: Elo (with margin-of-victory), last-20-game goals/shots for and against, points %, rest days.
2. **Linear regression for goals.** Two ridge regressions predict each team's expected goals in 60 minutes.
3. **Every possible score.** Those expected goals drive a Poisson model over every scoreline (0-0 to 14-14),
   with the tie probability calibrated to hockey's real rate. Ties go to OT/shootout. This gives win,
   overtime, over/under and win-by-2 probabilities.
4. **Logistic regression** predicts the winner directly from the same ratings.
5. **Headline prediction** = 50/50 blend of 3 and 4.

## How accurate is it? (walk-forward test, 5,246 games, 2022-23 to 2025-26)

Every prediction used only games played before it; models were retrained monthly. Full tables in
[`results/backtest.md`](results/backtest.md).

| model | accuracy | log loss |
|:--|--:|--:|
| Always pick the home team | 53.8% | 0.690 |
| Team with better record | 56.3% | 0.689 |
| Elo only | 57.8% | 0.670 |
| **Blend (headline)** | **58.2%** | **0.665** |

Accuracy goes up as the model's confidence goes up, and the stated probabilities are close to reality:

| model says at least | share of games | actually right |
|:--|--:|--:|
| 55% | 70% | 62.1% |
| 60% | 45% | 66.0% |
| 65% | 25% | 69.5% |
| 70% | 12% | 76.3% |

**Honest limits.** Hockey is the most random of the major sports; one bounce decides a lot of games, so
~58% overall is in line with good public models, and no stats model gets close to "always right". The
number to trust is the probability, not the pick: when it says 70%+ it has been right about 3 times in 4.
The model doesn't know about injuries, goalies in the lineup, trades, or playoff series, which are the
main things that could still improve it. Accuracy varied by season (61%, 60%, 59%, 53%); 2025-26 was a
high-parity season (25% of games went past regulation). Differences under 1-2 points between models are noise.

Don't use this for anything where losing money matters: sportsbooks' lines already contain this
information and more.

## Layout

- `hockey_predictor/data.py` - NHL API download + cache (`data/games.csv`)
- `hockey_predictor/features.py` - Elo and rolling-form features (no look-ahead)
- `hockey_predictor/models.py` - ridge goals model, Poisson score simulation, logistic model, blend
- `hockey_predictor/backtest.py` - walk-forward test and report
- `hockey_predictor/predict.py`, `cli.py` - prediction API and command line
