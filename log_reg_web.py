# log_reg_web.py

import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# Import your modeling code
from target_model import (
    TRAIN_DIR,
    OUTPUTS_DIR,
    train_and_evaluate,
    evaluate_week_with_model,
)

app = Flask(__name__)

# -------------------------------------------------------------------
# 1) Train model and build predictions + metrics at startup
# -------------------------------------------------------------------
WEEKS_TO_TRAIN = [1]           # base training week(s)
WEEKS_TO_EVAL = [1, 2, 3, 4]   # weeks to generate predictions from

print("=== Initializing logistic regression model and datasets ===")

# Ensure outputs dir exists (if you re-use it)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# --- Train on base week(s): here, just week 1 ---
base_week = WEEKS_TO_TRAIN[0]
base_csv = os.path.join(TRAIN_DIR, f"input_2023_w{base_week:02d}.csv")

if not os.path.exists(base_csv):
    raise FileNotFoundError(f"Training file not found: {base_csv}")

df_train = pd.read_csv(base_csv)
model, off_train, train_play_stats = train_and_evaluate(df_train)

train_metrics_row = {
    "dataset": f"week_{base_week:02d}",
    "split_type": "train_split",
    "top1_accuracy": float(train_play_stats["top1_correct"].mean()),
    "mean_true_prob": float(train_play_stats["true_target_prob"].mean()),
    "mean_pos_log_loss": float(train_play_stats["pos_log_loss"].mean()),
}

# --- Evaluate across multiple weeks to get "output" and "summary" datasets ---
all_predictions = []
metrics_rows = [train_metrics_row]

for wk in WEEKS_TO_EVAL:
    csv_path = os.path.join(TRAIN_DIR, f"input_2023_w{wk:02d}.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping missing file: {csv_path}")
        continue

    week_preds, week_metrics = evaluate_week_with_model(model, csv_path)
    week_metrics["dataset"] = f"week_{wk:02d}"
    week_metrics["split_type"] = "eval_week"

    all_predictions.append(week_preds)
    metrics_rows.append(week_metrics)

if all_predictions:
    predictions_df = pd.concat(all_predictions, ignore_index=True)
else:
    predictions_df = pd.DataFrame()


# Build metrics summary ("summary" dataset)
metrics_df = pd.DataFrame(metrics_rows)

# -------------------------------------------------------------------
# 2) Build a set of example rows for the UI
# -------------------------------------------------------------------

def make_examples(pred_df: pd.DataFrame, n_examples: int = 20):
    """
    Create a small set of interesting examples from the prediction dataset.
    Each example is one offensive player in some play, with:
      - game_id, play_id, nfl_id, player_name, player_position
      - target_prob, is_target, predicted_target
    """
    if pred_df.empty:
        return []

    # Keep only reasonable columns
    cols = [
        "game_id", "play_id", "nfl_id", "player_name", "player_position",
        "target_prob", "is_target", "predicted_target",
        #keep route + coverage context
        "dx", "dy", "route_len",
        "nearest_defender",
        "dist_to_qb",
        "num_defenders_within_2",
        "play_direction",
    ]
    cols = [c for c in cols if c in pred_df.columns]

    df_small = pred_df[cols].copy()

    # For variety, sample both true targets and non-targets
    targets = df_small[df_small.get("is_target", 0) == 1]
    non_targets = df_small[df_small.get("is_target", 0) == 0]

    examples = []

    def add_examples(sub_df, label, max_n):
        sub_df = sub_df.sample(min(len(sub_df), max_n), random_state=42) if len(sub_df) > 0 else sub_df
        for idx, row in sub_df.iterrows():
            examples.append({
                "id": f"{label}_{idx}",
                "game_id": str(row.get("game_id", "")),
                "play_id": str(row.get("play_id", "")),
                "nfl_id": str(row.get("nfl_id", "")),
                "player_name": row.get("player_name", "Unknown"),
                "player_position": row.get("player_position", "UNK"),
                "target_prob": float(row.get("target_prob", np.nan)),
                "is_target": int(row.get("is_target", 0)) if not pd.isna(row.get("is_target", np.nan)) else None,
                "predicted_target": bool(row.get("predicted_target", False)),

                #NEW: route / vector / defender context
                "dx": float(row.get("dx", np.nan)),
                "dy": float(row.get("dy", np.nan)),
                "route_len": float(row.get("route_len", np.nan)),
                "nearest_defender": float(row.get("nearest_defender", np.nan)),
                "dist_to_qb": float(row.get("dist_to_qb", np.nan)),
                "play_direction": (row.get("play_direction", None))
            })



    # e.g., up to half from targets, half from non-targets
    half = n_examples // 2
    add_examples(targets, "T", half)
    add_examples(non_targets, "N", n_examples - len(examples))

    return examples

EXAMPLES = make_examples(predictions_df, n_examples=20)

print(f"Prepared {len(EXAMPLES)} example rows for UI.")
print("=== Initialization complete ===")

# -------------------------------------------------------------------
# 3) Helper: get top players for a given play
# -------------------------------------------------------------------

def get_top_players_for_play(game_id: str, play_id: str, top_k: int = 5):
    if predictions_df.empty:
        return []

    mask = (
        predictions_df["game_id"].astype(str).eq(str(game_id)) &
        predictions_df["play_id"].astype(str).eq(str(play_id))
    )
    sub = predictions_df[mask].copy()
    if sub.empty:
        return []

    # Keep only useful columns (including route and coverage fields)
    cols = [
        "nfl_id", "player_name", "player_position",
        "target_prob", "is_target", "predicted_target",
        "dx", "dy", "route_len", "nearest_defender", "dist_to_qb"
    ]
    cols = [c for c in cols if c in sub.columns]
    sub = sub[cols]

    # sort by target probability (descending)
    sub = sub.sort_values("target_prob", ascending=False)

    # Take only the top K rows
    sub = sub.head(top_k)

    # Convert to dicts for HTML
    results = []
    for _, row in sub.iterrows():
        results.append({
            "nfl_id": str(row.get("nfl_id", "")),
            "player_name": row.get("player_name", ""),
            "player_position": row.get("player_position", ""),
            "target_prob": float(row.get("target_prob", float('nan'))),
            "is_target": int(row.get("is_target", 0)) if not pd.isna(row.get("is_target", np.nan)) else None,
            "predicted_target": bool(row.get("predicted_target", False)),
            "dx": float(row.get("dx", float('nan'))),
            "dy": float(row.get("dy", float('nan'))),
            "route_len": float(row.get("route_len", float('nan'))),
            "nearest_defender": float(row.get("nearest_defender", float('nan'))),
            "dist_to_qb": float(row.get("dist_to_qb", float('nan'))),
        })

    return results


# -------------------------------------------------------------------
# 4) Route: main Logistic Regression exploration page
# -------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def log_reg_page():
    """
    Render the HTML UI for exploring the target receiver logistic regression.
    - Shows metrics summary (metrics_df)
    - Shows a dropdown of example rows (EXAMPLES)
    - When an example is selected, shows:
        - That player's predicted probability
        - The top-k players for that play
    """
    selected_example_id = None
    selected_example = None
    top_players = []

    if request.method == "POST":
        selected_example_id = request.form.get("example_id", "")

        # Find the corresponding example object
        for ex in EXAMPLES:
            if ex["id"] == selected_example_id:
                selected_example = ex
                break

        if selected_example is not None:
            game_id = selected_example["game_id"]
            play_id = selected_example["play_id"]
            top_players = get_top_players_for_play(game_id, play_id, top_k=5)

    # Convert metrics_df to list of dicts for easy use in Jinja
    metrics_rows = metrics_df.to_dict(orient="records") if not metrics_df.empty else []

    return render_template(
        "log_reg.html",
        metrics_rows=metrics_rows,
        examples=EXAMPLES,
        selected_example=selected_example,
        top_players=top_players,
    )

# -------------------------------------------------------------------
# 5) Run standalone
# -------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)
