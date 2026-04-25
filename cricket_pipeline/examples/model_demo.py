"""End-to-end model demo.

Run after you've ingested CricSheet T20Is and installed views:
    python -m cricket_pipeline.pipeline cricsheet --dataset t20s_json --limit 500
    python -m cricket_pipeline.pipeline views
    python -m cricket_pipeline.examples.model_demo
"""

from __future__ import annotations

import json

from cricket_pipeline.model import train as M
from cricket_pipeline.model.predict import predict_ball
from cricket_pipeline.model.simulate import simulate_innings


def main():
    print("\n=== Training (filter=IT20, limit=200000 rows) ===")
    M.train(format_filter="IT20", limit=200_000)

    print("\n=== Predicting one ball ===")
    state = {
        "format":             "IT20",
        "venue":              "Eden Gardens",
        "batter":             "V Kohli",
        "bowler":             "JJ Bumrah",
        "batter_hand":        "Right hand Bat",
        "bowler_type":        "Right arm Fast",
        "phase":              "middle",
        "innings_no":         1,
        "over_no":            12,
        "ball_in_over":       1,
        "runs_so_far":        95,
        "wickets_so_far":     2,
        "deliveries_so_far":  72,
        "legal_balls_left":   48,
        "current_run_rate":   7.92,
        "required_run_rate":  None,
        "batter_sr":          135.0,
        "batter_avg":         55.0,
        "batter_balls":       12000,
        "bowler_econ":        7.2,
        "bowler_avg":         22.0,
        "bowler_balls":       9000,
        "batter_form_sr":     150.0,
        "batter_form_avg":    50.0,
        "batter_form_runs":   400,
        "bowler_workload_30d": 18.0,
        "bowler_workload_7d":   6.0,
        "venue_avg_first_innings": 175.0,
        "venue_toss_winner_won_pct": 0.55,
        "temp_c":   28.0,
        "humidity": 65.0,
        "wind_kmh": 8.0,
    }
    print(json.dumps(predict_ball(state), indent=2))

    print("\n=== Simulating remaining innings (chasing 180) ===")
    sim_state = {**state, "target": 180}
    print(json.dumps(simulate_innings(sim_state, n_sim=2000, seed=0), indent=2))


if __name__ == "__main__":
    main()
