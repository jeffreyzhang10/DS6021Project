"""
target_model.py

End-to-end pipeline to predict the targeted offensive player using
tracking-like data from input_2023_w01.csv.

Features include:
- Final-frame geometry
- Route shape (dx, dy, route_len)
- Distance to ball landing
- Defender separation & angle
- Number of defenders within 2 yards
- QB-relative orientation & distance (using player_position == 'QB')
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

# ---------------------------------------------------------------------
# Global paths (match your existing structure)
# ---------------------------------------------------------------------
BASE_DIR = ""#"./114239_nfl_competition_files_published_analytics_final"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SUPPLEMENTARY_PATH = os.path.join(TRAIN_DIR, "supplementary_data.csv")

# ---------------------------------------------------------------------
# Global feature lists
# ---------------------------------------------------------------------
# Numeric tracking / geometry features (unchanged from before)
NUMERIC_BASE_FEATURE_COLS = [
    "x_clean", "y_clean",
    "x_start", "y_start",
    "dx", "dy", "route_len",
    "dist_to_ball",
    "s_clean", "a_clean",
    "s_start", "a_start",
    #"absolute_yardline_number",
    "num_frames_output",
    "nearest_defender",
    "defender_angle_cos",
    "num_defenders_within_2",
    "dist_to_qb",
    "ori_diff_to_qb_deg",
    "cos_ori_to_qb",
]

# High-value categorical features (from supplementary + tracking)
CATEGORICAL_FEATURE_COLS = [
    # from supplementary_data.csv
    # "route_of_targeted_receiver",
    # "pass_length",
    # "possession_team",
    "team_coverage_type",
    "team_coverage_man_zone",
    # "yardline_side",
    # "down",
    # "pass_result",
    # NEW: from tracking/offense vectors
    "player_position",
    "alignment_role",
]


def build_offense_vectors(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Given a raw tracking-like dataframe with columns like input_2023_w01.csv,
    return one row per offensive player per play with engineered features.

    Features:
      - Final frame geometry and start geometry
      - dist_to_ball, dx, dy, route_len
      - Defender separation & angle
      - num_defenders_within_2
      - QB-relative distance & orientation
      - NEW: alignment_role (outside/slot/backfield/inline with side)
      - player_position (carried through for categorical features)
    """
    df = df_raw.copy()

    # --- Final frame per player ---
    df["player_final_frame"] = df.groupby(
        ["game_id", "play_id", "nfl_id"]
    )["frame_id"].transform("max")
    final = df[df["frame_id"] == df["player_final_frame"]].copy()

    # --- First frame per player ---
    first = (
        df.sort_values("frame_id")
          .groupby(["game_id", "play_id", "nfl_id"])
          .first()
          .reset_index()
    )

    start_cols = ["game_id", "play_id", "nfl_id", "x_clean", "y_clean", "s_clean", "a_clean"]
    first_small = first[start_cols].rename(columns={
        "x_clean": "x_start",
        "y_clean": "y_start",
        "s_clean": "s_start",
        "a_clean": "a_start",
    })

    final = final.merge(
        first_small,
        on=["game_id", "play_id", "nfl_id"],
        how="left",
        validate="one_to_one",
    )

    # --- Geometry to ball & route shape ---
    final["dist_to_ball"] = np.sqrt(
        (final["x_clean"] - final["ball_land_x"]) ** 2
        + (final["y_clean"] - final["ball_land_y"]) ** 2
    )
    final["dx"] = final["x_clean"] - final["x_start"]
    final["dy"] = final["y_clean"] - final["y_start"]
    final["route_len"] = np.sqrt(final["dx"] ** 2 + final["dy"] ** 2)

    # --- Offense and Defense splits ---
    off = final[final["player_side"] == "Offense"].copy()
    def_final = final[final["player_side"] == "Defense"].copy()

    # ------------------------------------------------------------------
    # Defender-based features (nearest defender, angle, defenders within 2y)
    # ------------------------------------------------------------------
    if not def_final.empty and not off.empty:
        pairs = off.merge(
            def_final[["game_id", "play_id", "nfl_id", "x_clean", "y_clean"]],
            on=["game_id", "play_id"],
            suffixes=("_off", "_def"),
        )

        # Vector from offensive player to defender (final frame)
        pairs["vec_od_x"] = pairs["x_def"] - pairs["x_off"]
        pairs["vec_od_y"] = pairs["y_def"] - pairs["y_off"]

        # Distance to each defender
        pairs["dist_off_def"] = np.sqrt(
            pairs["vec_od_x"] ** 2 + pairs["vec_od_y"] ** 2
        )

        # Defenders within 2 yards
        pairs["within_2"] = (pairs["dist_off_def"] <= 2.0).astype(int)
        close_counts = (
            pairs
            .groupby(["game_id", "play_id", "nfl_id_off"])["within_2"]
            .sum()
            .reset_index()
            .rename(columns={
                "nfl_id_off": "nfl_id",
                "within_2": "num_defenders_within_2",
            })
        )

        # Angle between route movement (dx, dy) and vector to defender
        pairs["route_norm"] = np.sqrt(pairs["dx"] ** 2 + pairs["dy"] ** 2)
        pairs["def_vec_norm"] = np.sqrt(
            pairs["vec_od_x"] ** 2 + pairs["vec_od_y"] ** 2
        )

        eps = 1e-6
        dot = pairs["dx"] * pairs["vec_od_x"] + pairs["dy"] * pairs["vec_od_y"]
        denom = pairs["route_norm"] * pairs["def_vec_norm"] + eps
        pairs["defender_angle_cos"] = dot / denom  # in [-1, 1]

        # Nearest defender per offensive player
        pairs = pairs.sort_values(
            ["game_id", "play_id", "nfl_id_off", "dist_off_def"]
        )
        nearest = (
            pairs
            .groupby(["game_id", "play_id", "nfl_id_off"])
            .first()
            .reset_index()
            .rename(columns={
                "nfl_id_off": "nfl_id",
                "dist_off_def": "nearest_defender",
            })
        )

        # Merge nearest defender distance & angle
        off = off.merge(
            nearest[[
                "game_id",
                "play_id",
                "nfl_id",
                "nearest_defender",
                "defender_angle_cos",
            ]],
            on=["game_id", "play_id", "nfl_id"],
            how="left",
        )

        # Merge defenders-within-2 count
        off = off.merge(
            close_counts,
            on=["game_id", "play_id", "nfl_id"],
            how="left",
        )

        off["num_defenders_within_2"] = (
            off["num_defenders_within_2"]
            .fillna(0)
            .astype(int)
        )
    else:
        off["nearest_defender"] = np.nan
        off["defender_angle_cos"] = np.nan
        off["num_defenders_within_2"] = 0

    # ------------------------------------------------------------------
    # QB-based features using player_position == 'QB'
    # ------------------------------------------------------------------
    if "player_position" in final.columns:
        qb_final = final[
            (final["player_side"] == "Offense")
            & (final["player_position"] == "QB")
        ][["game_id", "play_id", "x_clean", "y_clean", "o_clean"]].copy()

        qb_final = qb_final.rename(columns={
            "x_clean": "qb_x_final",
            "y_clean": "qb_y_final",
            "o_clean": "qb_o_final",
        })
    else:
        qb_final = pd.DataFrame(
            columns=["game_id", "play_id", "qb_x_final", "qb_y_final", "qb_o_final"]
        )

    if not qb_final.empty:
        off = off.merge(
            qb_final,
            on=["game_id", "play_id"],
            how="left",
        )

        # Vector from player to QB
        off["vec_p_to_qb_x"] = off["qb_x_final"] - off["x_clean"]
        off["vec_p_to_qb_y"] = off["qb_y_final"] - off["y_clean"]

        # Angle from player to QB (degrees)
        off["angle_to_qb_deg"] = np.degrees(
            np.arctan2(off["vec_p_to_qb_y"], off["vec_p_to_qb_x"])
        )

        # Normalize orientations to [0, 360)
        off["o_norm"] = off["o_clean"] % 360
        off["angle_to_qb_norm"] = off["angle_to_qb_deg"] % 360

        # Smallest signed angle difference in [-180, 180]
        off["ori_diff_to_qb_deg"] = (
            (off["o_norm"] - off["angle_to_qb_norm"] + 180) % 360 - 180
        )

        # Cosine of that difference: 1 = facing QB, -1 = facing away
        off["cos_ori_to_qb"] = np.cos(
            np.deg2rad(off["ori_diff_to_qb_deg"])
        )

        # Distance to QB
        off["dist_to_qb"] = np.sqrt(
            off["vec_p_to_qb_x"] ** 2 + off["vec_p_to_qb_y"] ** 2
        )
    else:
        off["ori_diff_to_qb_deg"] = np.nan
        off["cos_ori_to_qb"] = np.nan
        off["dist_to_qb"] = np.nan

    # ------------------------------------------------------------------
    # Ensure player_position is present for offense rows
    # ------------------------------------------------------------------
    if "player_position" not in off.columns and "player_position" in df.columns:
        off = off.merge(
            df[["game_id", "play_id", "nfl_id", "player_position"]].drop_duplicates(),
            on=["game_id", "play_id", "nfl_id"],
            how="left",
        )

    # ------------------------------------------------------------------
    # NEW: alignment_role based on starting x/y and player_position
    # ------------------------------------------------------------------

    # Estimate field center and sideline bands from the data
    y_min = off["y_start"].min()
    y_max = off["y_start"].max()
    mid_field_y = 0.5 * (y_min + y_max)

    sideline_band = 0.15 * (y_max - y_min)
    left_sideline_cut = y_min + sideline_band
    right_sideline_cut = y_max - sideline_band

    # Backfield heuristic: behind LOS / absolute yardline by ~2 yards
    if "absolute_yardline_number" in off.columns:
        off["is_backfield"] = (
            off["x_start"] < (off["absolute_yardline_number"] - 2.0)
        ).astype(int)
    else:
        off["is_backfield"] = 0  # fallback

    # Side of field: right vs left relative to mid_field_y
    off["side_right"] = (off["y_start"] > mid_field_y).astype(int)

    # ---------------- classify_alignment: full definition ----------------
    def classify_alignment(row):
        """
        Classify an offensive player's alignment role based on:
          - player_position
          - is_backfield (behind LOS)
          - side_right (right vs left of field center)
          - y_start (how close to sideline vs middle)
        Returns a string like:
          - 'backfield_rb_left', 'backfield_rb_right'
          - 'inline_te_left', 'inline_te_right'
          - 'outside_left', 'outside_right'
          - 'slot_left', 'slot_right'
          - 'backfield_qb', 'backfield_other_left', etc.
        """
        pos = str(row.get("player_position", "UNK") or "UNK")
        y0 = row["y_start"]

        # Backfield roles (RB, FB, QB occasionally)
        if row["is_backfield"]:
            if pos in ("RB", "FB"):
                return "backfield_rb_right" if row["side_right"] else "backfield_rb_left"
            elif pos == "QB":
                return "backfield_qb"
            else:
                return "backfield_other_right" if row["side_right"] else "backfield_other_left"

        # Inline TE: TE on or near LOS
        if pos == "TE":
            return "inline_te_right" if row["side_right"] else "inline_te_left"

        # WR-like: outside vs slot based on distance from sideline
        # near sideline: "outside"; else: "slot"
        if y0 <= left_sideline_cut:
            return "outside_left"
        elif y0 >= right_sideline_cut:
            return "outside_right"
        else:
            return "slot_right" if row["side_right"] else "slot_left"
    # ---------------- end classify_alignment ----------------------------

    off["alignment_role"] = off.apply(classify_alignment, axis=1)

    return off


# ---------------------------------------------------------------------
# NEW helper: merge supplementary play-level features
# ---------------------------------------------------------------------
def merge_supplementary(off: pd.DataFrame) -> pd.DataFrame:
    """
    Merge play-level supplementary data (formations, coverage, route, etc.)
    into the offense-level per-player dataframe.

    Expects SUPPLEMENTARY_PATH to contain at least:
      - game_id, play_id
      - route_of_targeted_receiver
      - pass_length
      - possession_team
      - team_coverage_type
      - team_coverage_man_zone
      - yardline_side
      - down
      - pass_result
    """
    if not os.path.exists(SUPPLEMENTARY_PATH):
        raise FileNotFoundError(
            f"Supplementary file not found at {SUPPLEMENTARY_PATH}"
        )

    supp = pd.read_csv(SUPPLEMENTARY_PATH)

    # Ensure join keys are strings
    for df_ in (off, supp):
        df_["game_id"] = df_["game_id"].astype(str)
        df_["play_id"] = df_["play_id"].astype(str)

    # Limit supplementary columns to only what we need (plus keys)
    cols_to_keep = ["game_id", "play_id"] + CATEGORICAL_FEATURE_COLS
    cols_to_keep = [c for c in cols_to_keep if c in supp.columns]

    supp_small = supp[cols_to_keep].copy()

    off_merged = off.merge(
        supp_small,
        on=["game_id", "play_id"],
        how="left",
        validate="many_to_one",  # many players -> one play row
    )

    return off_merged

# ---------------------------------------------------------------------
# CHANGED: build_feature_matrix
# ---------------------------------------------------------------------
def build_feature_matrix(off_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix X from offense-level dataframe.

    We do NOT preselect numeric / categorical columns here; we keep them
    in the DataFrame and let ColumnTransformer pick what it needs.

    We DO:
      - create the binary directional feature "play_dir_left"
    """
    X = off_df.copy()
    # encode play direction as a simple binary numeric feature
    X["play_dir_left"] = (off_df["play_direction"] == "left").astype(int)
    return X



# ---------------------------------------------------------------------
# CHANGED: train_and_evaluate – now uses ColumnTransformer with cat features
# ---------------------------------------------------------------------
def train_and_evaluate(df_raw: pd.DataFrame):
    """
    Build offense vectors, merge supplementary categorical features,
    train logistic regression model with numeric + categorical pipelines,
    and compute per-play metrics.
    """
    # 1) Build offense tracking features (unchanged logic)
    off = build_offense_vectors(df_raw)

    # 2) Merge supplementary play-level categorical info
    off = merge_supplementary(off)

    # 3) Label: targeted offensive player
    if "player_to_predict" not in off.columns:
        raise ValueError("Expected 'player_to_predict' column in data.")

    off["is_target"] = off["player_to_predict"].astype(int)

    # 4) Build feature matrix (adds play_dir_left)
    X = build_feature_matrix(off)
    y = off["is_target"].astype(int)

    # 5) Group by play for split (so we don't leak plays across splits)
    play_groups = (
        off[["game_id", "play_id"]]
        .astype(str)
        .agg("_".join, axis=1)
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=play_groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    meta_test = off.iloc[test_idx].copy()

    # 6) Define which columns are numeric vs categorical for the transformer
    numeric_features = NUMERIC_BASE_FEATURE_COLS + ["play_dir_left"]
    categorical_features = CATEGORICAL_FEATURE_COLS

    # Filter for columns that actually exist (in case of missing ones)
    numeric_features = [c for c in numeric_features if c in X_train.columns]
    categorical_features = [c for c in categorical_features if c in X_train.columns]

    # 7) Preprocessing pipelines
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",  # ignore any unused columns
    )

    # 8) Full model pipeline: preprocessing + logistic regression
    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=0.7,  # or even 0.3
        )),
    ])

    # 9) Fit model
    model.fit(X_train, y_train)

    # 10) Player-level probabilities (test only, for metrics)
    test_proba = model.predict_proba(X_test)[:, 1]
    meta_test["target_prob"] = test_proba

    # 11) Per-play evaluation (top-1 accuracy, mean prob, log-loss)
    meta_test["game_play"] = (
        meta_test["game_id"].astype(str) + "_" + meta_test["play_id"].astype(str)
    )

    play_groups = meta_test.groupby("game_play", group_keys=False)

    def per_play_stats(g: pd.DataFrame) -> pd.Series:
        # top-1 correctness (does argmax match true target?)
        best_idx = g["target_prob"].idxmax()
        top1_correct = int(g.loc[best_idx, "is_target"])

        # probability assigned to true target
        true_prob = g.loc[g["is_target"] == 1, "target_prob"].iloc[0]

        # positive log-loss: -log(p_true)
        pos_log_loss = -np.log(max(true_prob, 1e-15))

        return pd.Series({
            "top1_correct": top1_correct,
            "true_target_prob": true_prob,
            "pos_log_loss": pos_log_loss,
        })

    play_stats = play_groups.apply(per_play_stats)

    top1_accuracy = play_stats["top1_correct"].mean()
    mean_true_prob = play_stats["true_target_prob"].mean()
    mean_pos_log_loss = play_stats["pos_log_loss"].mean()

    print(f"Per-play top-1 accuracy: {top1_accuracy:.6f}")
    print(f"Mean true target probability: {mean_true_prob:.6f}")
    print(f"Mean positive log-loss: {mean_pos_log_loss:.6f}")

    return model, off, play_stats


# ---------------------------------------------------------------------
# Unchanged function: predict_targets_for_week (if you still use it)
# ---------------------------------------------------------------------
def predict_targets_for_week(model, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    [UNCHANGED or OPTIONAL: paste your previous implementation here,
    but you likely want to:
      - call build_offense_vectors(df_raw)
      - call merge_supplementary(off)
      - call build_feature_matrix(off)
      - then use model.predict_proba on X_new
    ]
    """
    raise NotImplementedError("Optionally update this based on your previous version.")


# ---------------------------------------------------------------------
# CHANGED: evaluate_week_with_model – also merges supplementary
# ---------------------------------------------------------------------
def evaluate_week_with_model(model, csv_path: str):
    """
    Load a week file, run the existing trained model on it,
    and if labels (player_to_predict) exist, compute per-play metrics.
    Otherwise, just return predictions and NaN metrics.

    Returns
    -------
    off : pd.DataFrame
        Per-offensive-player predictions for that week.
    metrics : dict
        Dictionary with keys:
            - top1_accuracy
            - mean_true_prob
            - mean_pos_log_loss
        Values are floats (or NaN if labels missing).
    """
    print(f"\n=== Evaluating file: {csv_path} ===")
    df_week = pd.read_csv(csv_path)

    off = build_offense_vectors(df_week)
    off = merge_supplementary(off)
    has_labels = "player_to_predict" in off.columns

    X_week = build_feature_matrix(off)

    # Same preprocessing as training: imputer + scaler are inside the model pipeline
    proba = model.predict_proba(X_week)[:, 1]
    off["target_prob"] = proba

    # Build a prediction flag per play
    off["game_play"] = (
        off["game_id"].astype(str) + "_" + off["play_id"].astype(str)
    )
    off["predicted_target"] = False
    for gp, idxs in off.groupby("game_play").groups.items():
        g = off.loc[idxs]
        best_idx = g["target_prob"].idxmax()
        off.loc[best_idx, "predicted_target"] = True

    metrics = {
        "top1_accuracy": np.nan,
        "mean_true_prob": np.nan,
        "mean_pos_log_loss": np.nan,
    }

    if has_labels:
        off["is_target"] = off["player_to_predict"].astype(int)

        meta_test = off.copy()
        meta_test["game_play"] = (
            meta_test["game_id"].astype(str)
            + "_"
            + meta_test["play_id"].astype(str)
        )
        play_groups = meta_test.groupby("game_play", group_keys=False)

        def per_play_stats(g: pd.DataFrame) -> pd.Series:
            best_idx = g["target_prob"].idxmax()
            top1_correct = int(g.loc[best_idx, "is_target"])
            true_prob = g.loc[g["is_target"] == 1, "target_prob"].iloc[0]
            pos_log_loss = -np.log(max(true_prob, 1e-15))
            return pd.Series({
                "top1_correct": top1_correct,
                "true_target_prob": true_prob,
                "pos_log_loss": pos_log_loss,
            })

        play_stats = play_groups.apply(per_play_stats)

        top1_accuracy = play_stats["top1_correct"].mean()
        mean_true_prob = play_stats["true_target_prob"].mean()
        mean_pos_log_loss = play_stats["pos_log_loss"].mean()

        print(f"Per-play top-1 accuracy: {top1_accuracy:.6f}")
        print(f"Mean true target probability: {mean_true_prob:.6f}")
        print(f"Mean positive log-loss: {mean_pos_log_loss:.6f}")

        metrics = {
            "top1_accuracy": float(top1_accuracy),
            "mean_true_prob": float(mean_true_prob),
            "mean_pos_log_loss": float(mean_pos_log_loss),
        }
    else:
        print("No 'player_to_predict' column found; returning predictions only.")

    return off, metrics

# ---------------------------------------------------------------------
# CHANGED: main – same flow, but uses global dirs and saves outputs
# ---------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # 1) Train on base week (e.g., week 1)
    #base_week = 1
    #base_csv = os.path.join(TRAIN_DIR, f"input_2023_w{base_week:02d}.csv")
    base_csv = "input_data_clean.csv"
    print(f"Training on {base_csv} ...")
    df_in = pd.read_csv(base_csv)

    # train_and_evaluate is assumed to already print its own metrics
    # and return play_stats, which we use to summarize training performance
    model, off_df, play_stats = train_and_evaluate(df_in)

    # Build a row of metrics for the training split
    train_metrics = {
        #"dataset": f"week_{base_week:02d}",
        "split_type": "train_split",
        "top1_accuracy": float(play_stats["top1_correct"].mean()),
        "mean_true_prob": float(play_stats["true_target_prob"].mean()),
        "mean_pos_log_loss": float(play_stats["pos_log_loss"].mean()),
    }

    # 2) Evaluate on multiple weeks and accumulate predictions + metrics
    #weeks_to_test = [1, 2, 3, 4]   # include 1 so you can compare train_split vs full week1
    all_predictions = []
    metrics_rows = [train_metrics]

    #for wk in weeks_to_test:
     #   csv_path = os.path.join(TRAIN_DIR, f"input_2023_w{wk:02d}.csv")
      #  try:
       #     week_preds, week_metrics = evaluate_week_with_model(model, csv_path)
        #    week_metrics["dataset"] = f"week_{wk:02d}"
         #   week_metrics["split_type"] = "eval_week"
         #   metrics_rows.append(week_metrics)
          #  all_predictions.append(week_preds)
        #except FileNotFoundError:
         #   print(f"File not found: {csv_path} (skipping)")

    all_predictions = evaluate_week_with_model(model, base_csv)

    # 3) Save combined predictions (unchanged behavior)
    if all_predictions:
        all_preds_df = pd.concat(all_predictions, ignore_index=True)

        cols_to_save = [
            "game_id", "play_id", "nfl_id", "player_name",
            "is_target",           # may be NaN if no labels for that week
            "predicted_target",
            "target_prob",
        ]
        cols_to_save = [c for c in cols_to_save if c in all_preds_df.columns]

        out_path = os.path.join(OUTPUTS_DIR, "predicted_targets_by_play.csv")
        all_preds_df[cols_to_save].to_csv(out_path, index=False)
        print(f"\nSaved combined predictions to: {out_path}")
    else:
        print("No predictions generated; nothing to save.")

    # 4) Build and print a metrics grid for easy comparison
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)

        # Order columns nicely
        col_order = [
            "dataset",
            "split_type",
            "top1_accuracy",
            "mean_true_prob",
            "mean_pos_log_loss",
        ]
        metrics_df = metrics_df[[c for c in col_order if c in metrics_df.columns]]

        print("\n=== Metrics Summary Grid ===")
        print(metrics_df.to_string(index=False, float_format="%.6f"))

        # Optionally save to CSV for tracking runs
        metrics_out = os.path.join(OUTPUTS_DIR, "metrics_summary.csv")
        metrics_df.to_csv(metrics_out, index=False)
        print(f"\nSaved metrics summary to: {metrics_out}")
