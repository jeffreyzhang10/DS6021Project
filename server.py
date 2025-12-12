from __future__ import annotations
from math import exp
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

# -----------------------
# Config
# -----------------------
TRACKING_CSV = "tracking_simplified_2023_all_weeks.csv"  # must contain both offense+defense rows

# Your fixed coefficients (log-odds)
COEF = {
    "position_WR": 0.8614567411084897,
    "position_TE": 0.48138759086548055,
    "position_RB": 0.21843175151292066,
    "dist_to_qb": -0.19653964181732517,
    "nearest_defender_dist": 0.10651670100892234,
    "vel_toward_qb": 0.06956957997670585,
}

# If you have the intercept, put it here (important for calibrated probabilities)
INTERCEPT = 0.0

TOPK = 12  # how many eligible players to return per play

app = Flask(__name__)

# -----------------------
# Load once (FAST at runtime)
# -----------------------
DF = pd.read_csv(TRACKING_CSV)

# normalize strings
DF["player_side"] = DF["player_side"].astype(str).str.upper().str.strip()
DF["position"] = DF["position"].astype(str).str.upper().str.strip()
DF["player_name"] = DF["player_name"].astype(str)

# build unique plays list once
UNIQ_PLAYS: List[Tuple[int, int]] = (
    DF[["game_id", "play_id"]].drop_duplicates().astype(int).to_records(index=False).tolist()
)


# -----------------------
# Helpers
# -----------------------
def sigmoid(z: float) -> float:
    if z >= 0:
        ez = exp(-z)
        return 1.0 / (1.0 + ez)
    ez = exp(z)
    return ez / (1.0 + ez)


def score_player(position: str, dist_to_qb: float, nearest_def: float | None, vel_toward_qb: int) -> float:
    z = INTERCEPT

    # one-hot positions (baseline = all other positions)
    if position == "WR":
        z += COEF["position_WR"]
    elif position == "TE":
        z += COEF["position_TE"]
    elif position == "RB":
        z += COEF["position_RB"]

    z += COEF["dist_to_qb"] * float(dist_to_qb)
    z += COEF["nearest_defender_dist"] * float(0.0 if nearest_def is None else nearest_def)
    z += COEF["vel_toward_qb"] * float(vel_toward_qb)

    return sigmoid(z)


def compute_features_for_play(df_play: pd.DataFrame) -> pd.DataFrame:
    """
    Input: all tracking rows for a single (game_id, play_id), offense+defense.
    Output: offense eligible rows (non-QB) with computed features needed by your model + plotting.
    """
    off = df_play[df_play["player_side"] == "OFFENSE"].copy()
    dfn = df_play[df_play["player_side"] == "DEFENSE"].copy()

    # QB location (mid-frame)
    qb = off[off["position"] == "QB"]
    if qb.empty:
        return pd.DataFrame()  # cannot compute without QB
    qb_row = qb.iloc[0]
    qb_x, qb_y = float(qb_row["x_mid"]), float(qb_row["y_mid"])

    # eligible offensive players (drop QB)
    off = off[off["position"] != "QB"].copy()
    if off.empty:
        return pd.DataFrame()

    # dist_to_qb
    dx = off["x_mid"].to_numpy() - qb_x
    dy = off["y_mid"].to_numpy() - qb_y
    off["dist_to_qb"] = np.sqrt(dx * dx + dy * dy)

    # vel_toward_qb (projection of velocity onto QB->player direction)
    dist = off["dist_to_qb"].to_numpy()
    ux = np.where(dist > 0, dx / dist, 0.0)
    uy = np.where(dist > 0, dy / dist, 0.0)

    theta = np.radians(off["dir_mid"].to_numpy())
    s = off["s_mid"].to_numpy()
    vx = s * np.cos(theta)
    vy = s * np.sin(theta)

    vel_proj_away = vx * ux + vy * uy
    off["vel_toward_qb"] = (vel_proj_away < 0).astype(int)

    # nearest_defender_dist
    if dfn.empty:
        off["nearest_defender_dist"] = np.nan
    else:
        def_xy = dfn[["x_mid", "y_mid"]].to_numpy()
        # compute min distance to any defender for each offensive player
        # (vectorized-ish per play)
        nearest = []
        for ox, oy in off[["x_mid", "y_mid"]].to_numpy():
            ddx = def_xy[:, 0] - ox
            ddy = def_xy[:, 1] - oy
            d = np.sqrt(ddx * ddx + ddy * ddy)
            nearest.append(float(np.min(d)))
        off["nearest_defender_dist"] = nearest

    # add QB coords for plotting
    off["qb_x"] = qb_x
    off["qb_y"] = qb_y

    return off


def rows_for_html(off_feat: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, r in off_feat.iterrows():
        nearest_def = None if pd.isna(r["nearest_defender_dist"]) else float(r["nearest_defender_dist"])
        p = score_player(
            position=str(r["position"]),
            dist_to_qb=float(r["dist_to_qb"]),
            nearest_def=nearest_def,
            vel_toward_qb=int(r["vel_toward_qb"]),
        )
        rows.append(
            {
                "nfl_id": int(r["nfl_id"]),
                "player_name": str(r["player_name"]),
                "position": str(r["position"]),
                "p_target": float(p),
                "dist_to_qb": float(r["dist_to_qb"]),
                "nearest_defender_dist": nearest_def,
                "vel_toward_qb": int(r["vel_toward_qb"]),
                "x_mid": float(r["x_mid"]),
                "y_mid": float(r["y_mid"]),
                "qb_x": float(r["qb_x"]),
                "qb_y": float(r["qb_y"]),
            }
        )
    rows.sort(key=lambda x: x["p_target"], reverse=True)
    return rows[:TOPK]


# -----------------------
# Routes
# -----------------------
@app.route("/")
def index():
    return send_from_directory(".", "log_reg.html")


@app.route("/api/predict_play")
def api_predict_play():
    """
    GET /api/predict_play?game_id=2023090700&play_id=101
    """
    game_id = int(request.args["game_id"])
    play_id = int(request.args["play_id"])

    df_play = DF[(DF["game_id"] == game_id) & (DF["play_id"] == play_id)]
    if df_play.empty:
        return jsonify({"error": "play not found"}), 404

    off_feat = compute_features_for_play(df_play)
    if off_feat.empty:
        return jsonify({"error": "QB missing or no eligible offense"}), 400

    rows = rows_for_html(off_feat)
    if not rows:
        return jsonify({"error": "no eligible rows"}), 400

    pred = rows[0]
    return jsonify(
        {
            "game_id": game_id,
            "play_id": play_id,
            "pred_nfl_id": pred["nfl_id"],
            "pred_player_name": pred["player_name"],
            "pred_position": pred["position"],
            "rows": rows,
        }
    )


@app.route("/api/random_play")
def api_random_play():
    """
    Convenience endpoint to grab a random play to display.
    GET /api/random_play?seed=7
    """
    seed = int(request.args.get("seed", "7"))
    rng = np.random.default_rng(seed)
    game_id, play_id = UNIQ_PLAYS[int(rng.integers(0, len(UNIQ_PLAYS)))]
    return jsonify({"game_id": int(game_id), "play_id": int(play_id)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
