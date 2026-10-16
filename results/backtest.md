# Walk-forward backtest

Seasons: 2022-23, 2023-24, 2024-25, 2025-26 - **5246 games**, every prediction made using only earlier games (models retrained monthly).

Accuracy standard error is about +/-0.7%, so differences smaller than ~1-2 points are noise.

## Models vs. baselines

| model                           |   accuracy |   log_loss |   brier |
|:--------------------------------|-----------:|-----------:|--------:|
| Always pick home team           |     0.5376 |     0.6904 |  0.2486 |
| Better points % (naive)         |     0.5631 |     0.6894 |  0.2476 |
| Elo only                        |     0.5783 |     0.6695 |  0.2385 |
| Logistic regression             |     0.5841 |     0.6651 |  0.2365 |
| Ridge goals + Poisson score sim |     0.5799 |     0.6652 |  0.2366 |
| Blend (headline model)          |     0.5822 |     0.6649 |  0.2364 |

## Accuracy when the model is more confident

| model_confidence_at_least   |   games |   share_of_games |   accuracy |   avg_stated_confidence |
|:----------------------------|--------:|-----------------:|-----------:|------------------------:|
| 50%                         |    5246 |            1.000 |      0.582 |                   0.602 |
| 55%                         |    3688 |            0.703 |      0.621 |                   0.635 |
| 60%                         |    2337 |            0.445 |      0.660 |                   0.671 |
| 65%                         |    1298 |            0.247 |      0.695 |                   0.709 |
| 70%                         |     633 |            0.121 |      0.763 |                   0.746 |

## Calibration (headline model)

| predicted_home_win_prob_bin   |   games |   predicted |   actual |
|:------------------------------|--------:|------------:|---------:|
| 0.14-0.39                     | 656.000 |       0.326 |    0.367 |
| 0.39-0.45                     | 656.000 |       0.420 |    0.434 |
| 0.45-0.49                     | 655.000 |       0.473 |    0.522 |
| 0.49-0.53                     | 656.000 |       0.513 |    0.483 |
| 0.53-0.57                     | 656.000 |       0.553 |    0.544 |
| 0.57-0.62                     | 655.000 |       0.595 |    0.573 |
| 0.62-0.67                     | 656.000 |       0.644 |    0.628 |
| 0.67-0.87                     | 656.000 |       0.730 |    0.748 |

## Per season (headline model)

|   season |    games |   accuracy |
|---------:|---------:|-----------:|
| 20222023 | 1310.000 |      0.611 |
| 20232024 | 1312.000 |      0.604 |
| 20242025 | 1312.000 |      0.587 |
| 20252026 | 1312.000 |      0.527 |
