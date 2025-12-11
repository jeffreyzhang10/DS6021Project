import pandas as pd
import numpy as np
from data_loader import load_weekly_data

input_data,output_data = load_weekly_data()
# Clean tracking data
# Clean orientation and direction
input_data["o_clean"] = (-(input_data["o"] - 90)) % 360
input_data["dir_clean"] = (-(input_data["dir"] - 90)) % 360
# Set x on same scale
input_data["x_clean"] = np.where(
      input_data["play_direction"] == "left",
      120 - input_data["x"],
      input_data[
          "x"
      ], 
  )
# y, s, a alreday clean
input_data["y_clean"] = input_data["y"]
input_data["s_clean"] = input_data["s"]
input_data["a_clean"] = input_data["a"]
# Clean orientation based on play direction
input_data["o_clean"] = np.where(
    input_data["play_direction"] == "left", 180 - input_data["o_clean"], input_data["o_clean"]
)
# Clean orientation, direction, vx, and vy
input_data["o_clean"] = (input_data["o_clean"] + 360) % 360 
input_data["dir_clean"] = (input_data["dir_clean"] + 360) % 360
input_data["dir_radians"] = np.radians(input_data["dir_clean"])
input_data["v_x"] = input_data["s_clean"] * np.cos(input_data["dir_radians"])
input_data["v_y"] = input_data["s_clean"] * np.sin(input_data["dir_radians"])
# Assuming your data should be sorted by time/sequence within a play:
input_data = input_data.sort_values(by=['nfl_id', 'game_id', 'play_id', 'frame_id'])
# Group by the specified columns and apply the shift within each group
input_data["prev_x"] = input_data.groupby(['nfl_id', 'game_id', 'play_id'])["x_clean"].shift(1)
input_data["prev_y"] = input_data.groupby(['nfl_id', 'game_id', 'play_id'])["y_clean"].shift(1)
# Remove the NaNs introduced by the shift operation (only the first entry of each group)
input_data = input_data.dropna(subset=["prev_x", "prev_y"])
df =input_data.drop(columns=['absolute_yardline_number', 'player_name', 'player_height', 'player_weight', 'player_birth_date', 'player_position', 'wk',
                             'x', 'y', 's', 'a', 'o', 'dir', 'play_direction', 'num_frames_output'])

df_sample = df.sample(n=100000, random_state=42)


df_sample.to_csv("input_data_clean.csv", index=False)






