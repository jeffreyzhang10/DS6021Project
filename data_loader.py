import pandas as pd

def read_data(prefix, wk):
    file = f"{prefix}_2023_w{wk}.csv"
    return pd.read_csv(file)

def load_weekly_data(weeks=range(1, 19)):
    input_data = pd.DataFrame()
    output_data = pd.DataFrame()

    for wk in weeks:
        wk_str = f"{wk:02d}"

        input_df = read_data("input", wk_str)
        output_df = read_data("output", wk_str)

        input_df["wk"] = wk
        output_df["wk"] = wk

        input_data = pd.concat([input_data, input_df], ignore_index=True)
        output_data = pd.concat([output_data, output_df], ignore_index=True)

    return input_data, output_data

def load_supplementary():
    return pd.read_csv("supplementary_data.csv", low_memory=False)