import os
import pandas as pd
import numpy as np
import datetime
import threading


DAY_ZERO = datetime.datetime(2020,1,22)
FORECASTS_NAMES = r"scripts\evaluate-script\forecasts_filenames.txt"
MODEL_NAMES = r"scripts\evaluate-script\models.txt"
US_DEATH_URL = "https://raw.githubusercontent.com/scc-usc/ReCOVER-COVID-19/master/results/forecasts/us_deaths.csv"
US_DEATH_FORECASTS_DIR = r"formatted-forecasts\US-COVID\state-death"
US_CASE_URL = "https://raw.githubusercontent.com/scc-usc/ReCOVER-COVID-19/master/results/forecasts/us_data.csv"
US_CASE_FORECASTS_DIR = r"formatted-forecasts\US-COVID\state-case"

def datetime_to_str(date):
    return date.strftime("%Y-%m-%d")

def str_to_datetime(date_str):
    return datetime.datetime.strptime(date_str,"%Y-%m-%d")

def get_inc_truth(url):
    # Fetch observed data.
    cum_truth = pd.read_csv(url, index_col="id")

    # Calculate incident data.
    inc_truth = cum_truth.drop(columns=["Country"])
    inc_truth = inc_truth.diff(axis=1)

    # Format week intervals.
    date_col1 = list(inc_truth.columns)
    date_col1.pop()
    date_col2 = list(inc_truth.columns)
    date_col2.pop(0)

    end_date = date_col2

    # Assign new column names.
    inc_truth = inc_truth.drop(columns=["2020-01-25"])
    inc_truth.columns = date_col2

    # Add region names.
    inc_truth.insert(0, "State", cum_truth["Country"])
    return inc_truth

def get_model_reports_mapping(forecasts_dir):
    mapping = {}
    # List all models (directories) in the forecasts directory
    models = [d for d in os.listdir(forecasts_dir) if os.path.isdir(os.path.join(forecasts_dir, d))]
    for model in models:
        model_dir = os.path.join(forecasts_dir, model)
        # List all forecast files for the model
        reports = [f for f in os.listdir(model_dir) if f.endswith('.csv')]
        mapping[model] = reports
    return mapping


def get_evaluation_df(foreast_type, metric, inc_truth, regions, models):
    wk_intervals = list(inc_truth.columns)[22:]
    model_evals = {}

    for region in regions:
        model_evals[region] = []
        for i in range(0, 4):
            path = "../../evaluation/US-COVID/{0}_eval/{1}_{2}_weeks_ahead_{3}.csv".format(foreast_type, metric, i+1, region)
            if os.path.exists(path):
                df = pd.read_csv(path, index_col=0);
                model_evals[region].append(pd.DataFrame(df, columns=wk_intervals))
            else:
                empty_array = np.empty((len(models), len(wk_intervals)))
                empty_array[:] = np.nan
                model_evals[region].append(pd.DataFrame(empty_array, columns=wk_intervals, index=models))

    return model_evals

def evaluate(inc_truth, model_name, metric, reports, regions, model_evals, forecasts_dir):
    for report in reports:
        path = os.path.join(forecasts_dir, model_name, report)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        # Fetch report data.
        print(f"Evaluating report: {report} for model: {model_name}")
        # print("Evaluating " + report)
        pred = pd.read_csv(path, index_col=0)
        
        # print("Forecast Data (pred.head()):")
        # print(pred.head())
        
        pred = pred.drop(columns=[pred.columns[1]])

        # Assign each column name to be week intervals.
        cols = list(pred.columns)
        for i in range(1, len(cols)):
            epi_day = int(cols[i])
            end_date = datetime_to_str(DAY_ZERO + datetime.timedelta(days=epi_day))
            cols[i] = end_date
        pred.columns = cols

        if metric == "mae":
            # Calculate MAE for each state
            pred_num = pred.drop(columns=["State"])
            pred_num = pred_num[sorted(pred_num.columns)]
            observed_wks = 4
            for i in range(0, 4):
                if i >= len(pred_num.columns) or pred_num.columns[i] > inc_truth.columns[-1]:
                    observed_wks -= 1
            pred_num = pred_num.drop(columns=pred_num.columns[observed_wks:])  # Only look at first 4 observed weeks
            
            # Calculate MAE
            mae_df = np.abs((pred_num - inc_truth[pred_num.columns]))
            mae_df.insert(0, "State", regions[:-1])

            # Calculate the mean MAE as the overall error
            overall_mae = mae_df.mean(numeric_only=True)
            overall_mae['State'] = "states"
            
            # Create DataFrame for overall and concat instead of append
            overall_mae_df = pd.DataFrame([overall_mae])
            mae_df = pd.concat([mae_df, overall_mae_df], ignore_index=True)
            
            # Update model_evals
            for i in range(0, observed_wks):
                interval = mae_df.columns[i+1]
                if interval in model_evals["states"][i].columns:
                    for region in regions:
                        value = mae_df[interval][mae_df["State"] == region].tolist()
                        if value:
                            model_evals[region][i].loc[model_name, interval] = value[0]
                        else:
                            model_evals[region][i].loc[model_name, interval] = np.nan

        elif metric == "mape":
            pred_num = pred.drop(columns=["State"])
            pred_num = pred_num[sorted(pred_num.columns)]
            observed_wks = 4
            for i in range(0, 4):
                if i >= len(pred_num.columns) or pred_num.columns[i] > inc_truth.columns[-1]:
                    observed_wks -= 1
            pred_num = pred_num.drop(columns=pred_num.columns[observed_wks:])
            
            # Calculate MAPE
            mape_df = np.abs((pred_num - inc_truth[pred_num.columns])) / inc_truth[pred_num.columns]
            mape_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            mape_df = mape_df.fillna(0)  # Changed from fillna(0) to make it consistent
            mape_df.insert(0, "State", regions[:-1])

            # Calculate the mean MAPE as the overall error
            overall_mape = mape_df.mean(numeric_only=True)
            overall_mape['State'] = "states"
            
            # Create DataFrame for overall and concat instead of append
            overall_mape_df = pd.DataFrame([overall_mape])
            mape_df = pd.concat([mape_df, overall_mape_df], ignore_index=True)
            
            # Update model_evals
            for i in range(0, observed_wks):
                interval = mape_df.columns[i+1]
                if interval in model_evals["states"][i].columns:
                    for region in regions:
                        value = mape_df[interval][mape_df["State"] == region].tolist()
                        if value:
                            model_evals[region][i].loc[model_name, interval] = value[0]
                        else:
                            model_evals[region][i].loc[model_name, interval] = np.nan

        elif metric == "forecast_value":
            # Existing forecast_value code remains the same
            pred_num = pred.drop(columns=["State"])
            pred_num = pred_num[sorted(pred_num.columns)]
            observed_wks = 4
            for i in range(0, 4):
                if i >= len(pred_num.columns) or pred_num.columns[i] > inc_truth.columns[-1]:
                    observed_wks -= 1
            pred_num = pred_num.drop(columns=pred_num.columns[observed_wks:])

            # Prepare DataFrame with predictions
            forecast_df = pred_num.copy()
            forecast_df.insert(0, "State", pred["State"])

            # Add an overall average for "states"
            overall_forecast = forecast_df.mean(numeric_only=True)
            overall_forecast['State'] = "states"
            
            overall_forecast_df = pd.DataFrame([overall_forecast])
            forecast_df = pd.concat([forecast_df, overall_forecast_df], ignore_index=True)

            # Update model_evals
            for i in range(0, observed_wks):
                interval = forecast_df.columns[i+1]
                if interval in model_evals["states"][i].columns:
                    for region in regions:
                        value = forecast_df[interval][forecast_df["State"] == region].tolist()
                        if value:
                            model_evals[region][i].loc[model_name, interval] = value[0]
                        else:
                            model_evals[region][i].loc[model_name, interval] = np.nan

def generate_average_evals(regions, model_evals):
    average_evals = {}
    for region in regions:
        week_ahead_4 = model_evals[region][3]
        week_ahead_3 = model_evals[region][2]
        week_ahead_2 = model_evals[region][1]
        week_ahead_1 = model_evals[region][0]

        # Make sure the forecast made in the same forecast report are named under the same column.
        week_ahead_4 = week_ahead_4[week_ahead_4.columns[3:]]
        week_ahead_3 = week_ahead_3[week_ahead_3.columns[2:-1]]
        week_ahead_2 = week_ahead_2[week_ahead_2.columns[1:-2]]
        week_ahead_1 = week_ahead_1[week_ahead_1.columns[:-3]]

        week_ahead_3.columns = week_ahead_4.columns
        week_ahead_2.columns = week_ahead_4.columns
        week_ahead_1.columns = week_ahead_4.columns

        average = (week_ahead_4 + week_ahead_3 + week_ahead_2 + week_ahead_1) / 4
        average_evals[region] = average
    return average_evals

def run():
    model_reports_mapping = get_model_reports_mapping(US_DEATH_FORECASTS_DIR)
    # print("Model Reports Mapping:")
    # print(model_reports_mapping)
    
    #Death eval - Forecast_val
    output_dir = os.path.join('evaluation', 'US-COVID', 'state_death_eval')
    os.makedirs(output_dir, exist_ok=True)
    inc_truth = get_inc_truth(US_DEATH_URL)
    state_col = list(inc_truth["State"])
    state_col.append("states")
    model_evals = get_evaluation_df("state_death", "forecast_value", inc_truth, state_col, model_reports_mapping.keys())

    for model in model_reports_mapping:
        reports = model_reports_mapping[model]
        evaluate(inc_truth, model, "forecast_value", reports, state_col, model_evals, US_DEATH_FORECASTS_DIR)

    for state in model_evals:
        for i in range(len(model_evals[state])):
            # print(f"Model Evaluations for state: {state}, week ahead: {i+1}")
            # print(model_evals[state][i].head())
            model_evals[state][i].to_csv(output_dir + "/forecast_value_{0}_weeks_ahead_{1}.csv".format(i+1, state))

    average_evals = generate_average_evals(state_col, model_evals)
    for state in average_evals:
        average_evals[state].to_csv(output_dir + "/forecast_value_avg_{0}.csv".format(state))
        
    # Death eval - MAE    
    model_evals = get_evaluation_df("state_death", "mae", inc_truth, state_col, model_reports_mapping.keys())
    
    for model in model_reports_mapping:
        reports = model_reports_mapping[model]
        evaluate(inc_truth, model, "mae", reports, state_col, model_evals, US_DEATH_FORECASTS_DIR)

    for state in model_evals:
        for i in range(len(model_evals[state])):
            # print(f"Model Evaluations for state: {state}, week ahead: {i+1}")
            # print(model_evals[state][i].head())
            model_evals[state][i].to_csv(output_dir + "/mae_{0}_weeks_ahead_{1}.csv".format(i+1, state))

    average_evals = generate_average_evals(state_col, model_evals)
    for state in average_evals:
        average_evals[state].to_csv(output_dir + "/mae_avg_{0}.csv".format(state))

    # Death eval - MAPE
    model_evals = get_evaluation_df("state_death", "mape", inc_truth, state_col, model_reports_mapping.keys())
    for model in model_reports_mapping:
        reports = model_reports_mapping[model]
        evaluate(inc_truth, model, "mape", reports, state_col, model_evals, US_DEATH_FORECASTS_DIR)

    for state in model_evals:
        for i in range(len(model_evals[state])):
            model_evals[state][i].to_csv(output_dir + "/mape_{0}_weeks_ahead_{1}.csv".format(i+1, state))

    average_evals = generate_average_evals(state_col, model_evals)
    for state in average_evals:
        average_evals[state].to_csv(output_dir + "/mape_avg_{0}.csv".format(state))
    
    #Case eval - Forecast_val
    output_dir = os.path.join('evaluation', 'US-COVID', 'state_case_eval')
    os.makedirs(output_dir, exist_ok=True)
    inc_truth = get_inc_truth(US_CASE_URL)
    state_col = list(inc_truth["State"])
    state_col.append("states")    

    model_evals = get_evaluation_df("state_case", "forecast_value", inc_truth, state_col, model_reports_mapping.keys())
    
    for model in model_reports_mapping:
        reports = model_reports_mapping[model]
        evaluate(inc_truth, model, "forecast_value", reports, state_col, model_evals, US_CASE_FORECASTS_DIR)
    
    for state in model_evals:
        for i in range(len(model_evals[state])):
            # print(f"Model Evaluations for state: {state}, week ahead: {i+1}")
            # print(model_evals[state][i].head())
            model_evals[state][i].to_csv(output_dir + "/forecast_value_{0}_weeks_ahead_{1}.csv".format(i+1, state))
    
    average_evals = generate_average_evals(state_col, model_evals)
    for state in average_evals:
        average_evals[state].to_csv(output_dir + "/forecast_value_avg_{0}.csv".format(state))
        
    # Case eval - MAE
    output_dir = os.path.join('evaluation', 'US-COVID', 'state_case_eval')
    os.makedirs(output_dir, exist_ok=True)
    inc_truth = get_inc_truth(US_CASE_URL)
    state_col = list(inc_truth["State"])
    state_col.append("states")

    model_evals = get_evaluation_df("state_case", "mae", inc_truth, state_col, model_reports_mapping.keys())
    for model in model_reports_mapping:
        reports = model_reports_mapping[model]
        evaluate(inc_truth, model, "mae", reports, state_col, model_evals, US_CASE_FORECASTS_DIR)

    for state in model_evals:
        for i in range(len(model_evals[state])):
            model_evals[state][i].to_csv(output_dir + "/mae_{0}_weeks_ahead_{1}.csv".format(i+1, state))

    average_evals = generate_average_evals(state_col, model_evals)
    for state in average_evals:
        average_evals[state].to_csv(output_dir + "/mae_avg_{0}.csv".format(state))

    # Case eval - MAPE
    model_evals = get_evaluation_df("state_case", "mape", inc_truth, state_col, model_reports_mapping.keys())
    for model in model_reports_mapping:
        reports = model_reports_mapping[model]
        evaluate(inc_truth, model, "mape", reports, state_col, model_evals, US_CASE_FORECASTS_DIR)

    for state in model_evals:
        for i in range(len(model_evals[state])):
            model_evals[state][i].to_csv(output_dir + "/mape_{0}_weeks_ahead_{1}.csv".format(i+1, state))

    average_evals = generate_average_evals(state_col, model_evals)
    for state in average_evals:
        average_evals[state].to_csv(output_dir + "/mape_avg_{0}.csv".format(state))


if __name__ == "__main__":
    run()