import os
import pandas as pd
import numpy as np
import datetime

# Constants
DAY_ZERO = datetime.datetime(2020, 1, 22)
OUTPUT_DIR = "evaluation_full"
US_DEATH_URL = "https://raw.githubusercontent.com/scc-usc/ReCOVER-COVID-19/master/results/forecasts/us_deaths.csv"
US_DEATH_FORECASTS_DIR = r"formatted-forecasts/US-COVID/state-death"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Utility Functions
def datetime_to_str(date):
    return date.strftime("%Y-%m-%d")

def fetch_inc_truth(url):
    # Fetch observed cumulative data
    cum_truth = pd.read_csv(url, index_col="id")
    
    # Convert all columns to numeric, coercing errors to NaN
    for col in cum_truth.columns:
        cum_truth[col] = pd.to_numeric(cum_truth[col], errors="coerce")
    
    # Handle NaN values by filling with 0 (optional based on data expectations)
    cum_truth.fillna(0, inplace=True)

    # Calculate incident data (week-over-week differences)
    inc_truth = cum_truth.diff(axis=1).fillna(0)

    # Save for debugging
    inc_truth.to_csv(f"{OUTPUT_DIR}/inc_truth.csv")
    return inc_truth

def fetch_sample_forecast_data(inc_truth):
    # Simulate forecast data aligned with inc_truth dates
    forecast_data = inc_truth.copy()
    for col in forecast_data.columns:
        forecast_data[col] = forecast_data[col] + np.random.randint(-20, 20, forecast_data.shape[0])
    forecast_data.to_csv(f"{OUTPUT_DIR}/forecast_data.csv")  # Save for debugging
    return forecast_data

def calculate_metrics(inc_truth, forecast_data):
    # Calculate MAE
    mae = np.abs(forecast_data - inc_truth)
    mae.to_csv(f"{OUTPUT_DIR}/mae.csv")  # Save MAE for debugging

    # Calculate MAPE
    mape = np.abs(forecast_data - inc_truth) / inc_truth.replace(0, np.nan)
    mape.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinity values
    mape.fillna(0, inplace=True)  # Replace NaN with 0 for simplicity
    mape.to_csv(f"{OUTPUT_DIR}/mape.csv")  # Save MAPE for debugging

    return mae, mape

def generate_average_eval(mae, mape):
    # Combine MAE and MAPE into an average evaluation metric
    average_eval = (mae + mape) / 2
    average_eval.to_csv(f"{OUTPUT_DIR}/average_eval.csv")  # Save average evaluation
    return average_eval

def run():
    # Fetch and process data
    print("Fetching incident truth data...")
    inc_truth = fetch_inc_truth(US_DEATH_URL)

    print("Generating sample forecast data...")
    forecast_data = fetch_sample_forecast_data(inc_truth)

    print("Calculating metrics...")
    mae, mape = calculate_metrics(inc_truth, forecast_data)

    print("Generating average evaluation...")
    generate_average_eval(mae, mape)

    print(f"CSV files generated in the directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    run()