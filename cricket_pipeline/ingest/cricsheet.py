"""CricSheet ingester: download a zip of ball-by-ball JSON and load into DuckDB."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm

from .. import config
from ..db import connect


def download_zip(dataset: str, force: bool = False) -> Path:
    if dataset not in config.CRICSHEET_ZIPS:
        raise ValueError(
            f"Unknown CricSheet dataset '{dataset}'. "
            f"Options: {list(config.CRICSHEET_ZIPS)}"
        )
    url = config.CRICSHEET_ZIPS[dataset]
    dest = config.CACHE_DIR / f"{dataset}.zip"
    if dest.exists() and not force:
        return dest
    print(f"Downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dataset
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 15):
                f.write(chunk)
                bar.update(len(chunk))
    return dest


def _iter_match_jsons(zip_path: Path, limit: int | None = None) -> Iterable[dict]:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json") and n != "README.txt"]
        if limit:
            names = names[:limit]
        for name in names:
            with zf.open(name) as fh:
                try:
                    yield json.loads(fh.read().decode("utf-8"))
                except json.JSONDecodeError:
                    continue


def _match_row(m: dict) -> dict:
    info = m.get("info", {})
    dates = info.get("dates") or []
    outcome = info.get("outcome") or {}
    by = outcome.get("by") or {}
    teams = info.get("teams") or [None, None]
    toss = info.get("toss") or {}
    return {
        "match_id":           _match_id_from(info),
        "format":             info.get("match_type"),
        "competition":        (info.get("event") or {}).get("name"),
        "season":             str(info.get("season", "")),
        "start_date":         dates[0] if dates else None,
        "end_date":           dates[-1] if dates else None,
        "venue":              info.get("venue"),
        "city":               info.get("city"),
        "country":            (info.get("event") or {}).get("country"),
        "team_home":          teams[0] if len(teams) > 0 else None,
        "team_away":          teams[1] if len(teams) > 1 else None,
        "toss_winner":        toss.get("winner"),
        "toss_decision":      toss.get("decision"),
        "winner":             outcome.get("winner"),
        "win_margin_runs":    by.get("runs"),
        "win_margin_wickets": by.get("wickets"),
        "player_of_match":    (info.get("player_of_match") or [None])[0],
        "umpires":            ",".join((info.get("officials") or {}).get("umpires", []) or []),
    }


def _match_id_from(info: dict) -> str:
    """Deterministic match id. CricSheet uses filename id; we re-derive."""
    reg = info.get("registry", {}) or {}
    people = reg.get("people") or {}
    dates = info.get("dates") or [""]
    teams = info.get("teams") or []
    venue = info.get("venue") or ""
    raw = f"{dates[0]}|{'-'.join(teams)}|{venue}|{len(people)}"
    return str(abs(hash(raw)))


def _innings_rows(match_id: str, m: dict) -> list[dict]:
    rows = []
    for i, inn in enumerate(m.get("innings", []), start=1):
        runs = wkts = 0
        legal_balls = 0
        for ov in inn.get("overs", []):
            for d in ov.get("deliveries", []):
                r = d.get("runs", {})
                runs += int(r.get("total", 0) or 0)
                if d.get("wickets"):
                    wkts += len(d["wickets"])
                ex = d.get("extras", {}) or {}
                if not (ex.get("wides") or ex.get("noballs")):
                    legal_balls += 1
        rows.append({
            "match_id":     match_id,
            "innings_no":   i,
            "batting_team": inn.get("team"),
            "bowling_team": _other_team(inn.get("team"), m.get("info", {}).get("teams") or []),
            "total_runs":   runs,
            "total_wkts":   wkts,
            "total_overs":  legal_balls / 6.0,
            "target":       (inn.get("target") or {}).get("runs"),
        })
    return rows


def _other_team(team: str | None, teams: list[str]) -> str | None:
    if not team or not teams:
        return None
    return next((t for t in teams if t != team), None)


def _officials_rows(match_id: str, m: dict) -> list[dict]:
    officials = (m.get("info", {}) or {}).get("officials") or {}
    role_map = {
        "umpires":          "umpire",
        "tv_umpires":       "tv_umpire",
        "reserve_umpires":  "reserve_umpire",
        "match_referees":   "match_referee",
    }
    rows = []
    for key, role in role_map.items():
        for name in officials.get(key, []) or []:
            if name:
                rows.append({"match_id": match_id, "role": role, "name": name})
    return rows


def _ball_rows(match_id: str, m: dict) -> list[dict]:
    rows = []
    info_teams = m.get("info", {}).get("teams") or []
    for innings_no, inn in enumerate(m.get("innings", []), start=1):
        batting_team = inn.get("team")
        bowling_team = _other_team(batting_team, info_teams)
        legal_ball_no = 0
        for ov in inn.get("overs", []):
            over_no = ov.get("over")
            for ball_in_over, d in enumerate(ov.get("deliveries", []), start=1):
                runs = d.get("runs", {}) or {}
                extras = d.get("extras", {}) or {}
                ex_types = [k for k in ("wides", "noballs", "byes", "legbyes", "penalty") if k in extras]
                is_legal = not (extras.get("wides") or extras.get("noballs"))
                if is_legal:
                    legal_ball_no += 1
                wkts = d.get("wickets") or []
                w = wkts[0] if wkts else {}
                rows.append({
                    "match_id":      match_id,
                    "innings_no":    innings_no,
                    "over_no":       over_no,
                    "ball_in_over":  ball_in_over,
                    "legal_ball_no": legal_ball_no if is_legal else None,
                    "batting_team":  batting_team,
                    "bowling_team":  bowling_team,
                    "batter":        d.get("batter"),
                    "non_striker":   d.get("non_striker"),
                    "bowler":        d.get("bowler"),
                    "runs_batter":   int(runs.get("batter", 0) or 0),
                    "runs_extras":   int(runs.get("extras", 0) or 0),
                    "runs_total":    int(runs.get("total", 0) or 0),
                    "extras_type":   ",".join(ex_types) if ex_types else None,
                    "is_wicket":     bool(wkts),
                    "wicket_kind":   w.get("kind"),
                    "player_out":    w.get("player_out"),
                    "fielders":      ",".join(
                        f.get("name") for f in (w.get("fielders") or []) if f.get("name")
                    ) or None,
                })
    return rows


def load_zip_to_db(zip_path: Path, db_path: Path | str | None = None, limit: int | None = None) -> dict:
    con = connect(db_path)
    matches, innings, balls, officials = [], [], [], []
    count = 0
    for m in tqdm(_iter_match_jsons(zip_path, limit=limit), desc="parsing"):
        mid = _match_id_from(m.get("info", {}))
        matches.append(_match_row(m))
        innings.extend(_innings_rows(mid, m))
        balls.extend(_ball_rows(mid, m))
        officials.extend(_officials_rows(mid, m))
        count += 1
        if len(balls) > 200_000:
            _flush(con, matches, innings, balls, officials)
            matches, innings, balls, officials = [], [], [], []
    _flush(con, matches, innings, balls, officials)
    con.close()
    return {"matches_loaded": count}


def _flush(con, matches: list[dict], innings: list[dict], balls: list[dict], officials: list[dict] | None = None) -> None:
    """Bulk-insert each batch via a registered DataFrame.

    `executemany` with named parameters performs one bind+execute per row,
    which on a 100k+ row batch is dramatically slower than letting DuckDB
    ingest the batch in one go via a registered Pandas DataFrame.
    """
    import pandas as pd

    def _bulk(table: str, rows: list[dict], cols: list[str]) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows, columns=cols)
        con.register("_stage", df)
        try:
            con.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM _stage")
        finally:
            con.unregister("_stage")

    _bulk("matches", [{**m, "source": "cricsheet"} for m in matches], [
        "match_id", "format", "competition", "season",
        "start_date", "end_date", "venue", "city", "country",
        "team_home", "team_away", "toss_winner", "toss_decision",
        "winner", "win_margin_runs", "win_margin_wickets",
        "player_of_match", "umpires", "source",
    ])

    _bulk("innings", innings, [
        "match_id", "innings_no", "batting_team", "bowling_team",
        "total_runs", "total_wkts", "total_overs", "target",
    ])

    _bulk("balls", balls, [
        "match_id", "innings_no", "over_no", "ball_in_over",
        "legal_ball_no", "batting_team", "bowling_team",
        "batter", "non_striker", "bowler",
        "runs_batter", "runs_extras", "runs_total",
        "extras_type", "is_wicket", "wicket_kind",
        "player_out", "fielders",
    ])

    if officials:
        _bulk("match_officials", officials, ["match_id", "role", "name"])


def ingest(dataset: str = "t20s_json", limit: int | None = None, force: bool = False) -> dict:
    zip_path = download_zip(dataset, force=force)
    return load_zip_to_db(zip_path, limit=limit)
