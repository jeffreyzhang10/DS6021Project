import pandas as pd
# Build function to read in data based on input/output for a given week
def read_data(prefix, wk):
    # Set "file" equal to the inputted file path and week
    file = f"{prefix}_2023_w{wk}.csv"
    return pd.read_csv(file) # Return the file read
# Load all input data for a given week
def load_weekly_data(weeks=range(1, 19)): # For weeks 1-18
    # Create empty data frames
    input_data = pd.DataFrame()

    # Iterate through weeks
    for wk in weeks:
        # Set week as a string (e.g. 1 becomes "01")
        wk_str = f"{wk:02d}"

        # Read the data 
        input_df = read_data("train/input", wk_str)

        # Concatenate the data for each week
        input_data = pd.concat([input_data, input_df], ignore_index=True)

    return input_data
def load_supplementary():
    return pd.read_csv("supplementary_data.csv", low_memory=False)