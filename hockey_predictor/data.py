"""Download and cache NHL regular-season game results (scores, shots, OT/SO flag).

Source: the public NHL web API (api-web.nhle.com). One boxscore call per game.
Results are cached in ``data/games.csv`` and updates only fetch games we don't have yet.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://api-web.nhle.com/v1"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "games.csv"

# Current franchises. ARI (Coyotes) and UTA (Utah) share one franchise line for our purposes.
TEAM_CODES = [
    "ANA", "ARI", "BOS", "BUF", "CGY", "CAR", "CHI", "CBJ", "COL", "DAL", "DET",
    "EDM", "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR", "OTT", "PHI",
    "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
]
CANONICAL = {"ARI": "UTA"}  # keep Elo / form continuous across the relocation

COLUMNS = [
    "game_id", "season", "date", "home", "away",
    "home_goals", "away_goals", "home_sog", "away_sog", "ending",  # ending: REG / OT / SO
]


def canon(code: str) -> str:
    return CANONICAL.get(code, code)


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=32))
    return s


def season_id(start_year: int) -> int:
    return int(f"{start_year}{start_year + 1}")


def list_final_games(season: int, session: requests.Session) -> dict[int, str]:
    """Return {game_id: date} for every completed regular-season game of ``season`` (e.g. 20232024)."""
    games: dict[int, str] = {}
    for team in TEAM_CODES:
        r = session.get(f"{API}/club-schedule-season/{team}/{season}", timeout=30)
        if r.status_code == 404:
            continue  # team didn't exist / wasn't active that season
        r.raise_for_status()
        for g in r.json().get("games", []):
            if g.get("gameType") == 2 and g.get("gameState") in ("OFF", "FINAL"):
                games[g["id"]] = g["gameDate"]
    return games


def fetch_game(game_id: int, date: str, session: requests.Session) -> dict | None:
    r = session.get(f"{API}/gamecenter/{game_id}/boxscore", timeout=30)
    r.raise_for_status()
    b = r.json()
    home, away = b["homeTeam"], b["awayTeam"]
    if "score" not in home or "score" not in away:
        return None
    ending = (b.get("gameOutcome") or {}).get("lastPeriodType") or b.get("periodDescriptor", {}).get("periodType", "REG")
    return {
        "game_id": game_id,
        "season": b["season"],
        "date": date,
        "home": canon(home["abbrev"]),
        "away": canon(away["abbrev"]),
        "home_goals": home["score"],
        "away_goals": away["score"],
        "home_sog": home.get("sog"),
        "away_sog": away.get("sog"),
        "ending": ending,
    }


def load_games(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "game_id"]).reset_index(drop=True)


def update(first_year: int = 2018, last_year: int | None = None, path: Path = DATA_PATH,
           workers: int = 12) -> pd.DataFrame:
    """Fetch any completed games missing from the cache and rewrite the CSV."""
    last_year = last_year or (pd.Timestamp.today().year)
    session = _session()
    have = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
    known = set(have["game_id"].astype(int))

    todo: dict[int, str] = {}
    for y in range(first_year, last_year + 1):
        found = list_final_games(season_id(y), session)
        new = {g: d for g, d in found.items() if g not in known}
        print(f"season {y}-{y + 1}: {len(found)} final games, {len(new)} new", flush=True)
        todo.update(new)
    if not todo:
        return load_games(path) if path.exists() else have

    rows, done, t0 = [], 0, time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_game, g, d, session): g for g, d in todo.items()}
        for f in as_completed(futs):
            try:
                row = f.result()
            except Exception as e:  # keep going; a re-run picks up what failed
                print(f"  game {futs[f]} failed: {e}", file=sys.stderr)
                continue
            if row:
                rows.append(row)
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(todo)} fetched ({time.time() - t0:.0f}s)", flush=True)
                _write(have, rows, path)
    _write(have, rows, path)
    return load_games(path)


def _write(have: pd.DataFrame, rows: list[dict], path: Path) -> None:
    out = pd.concat([have, pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True)
    out = out.drop_duplicates("game_id").sort_values(["date", "game_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


if __name__ == "__main__":
    df = update()
    print(f"{len(df)} games cached in {DATA_PATH}")
