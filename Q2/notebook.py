# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars==1.35.2",
#     "pulp[highs]==3.3.0",
# ]
# ///

import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import marimo as mo

    import itertools
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import warnings
    from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
    return (
        ARIMA,
        ConvergenceWarning,
        ValueWarning,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pd,
        warnings,
    )


@app.cell
def _(pd):
    ## Data Cleaning Functions

    def rename_columns(df):
        return df.rename(
            columns={
                "Date of write up": "date",
                "Job Description": "lens_type",
            })


    def fix_dates(df):
        df["date"] = pd.to_datetime(
            df["date"], format="mixed", dayfirst=True, errors="coerce"
        )
        return df
    return fix_dates, rename_columns


@app.cell
def _(pd):
    df_lab_data = pd.read_excel("./Lens Data.xlsx", sheet_name="Lab Data")
    df_lab_data
    return (df_lab_data,)


@app.cell
def _(df_lab_data, fix_dates, rename_columns):
    df_lab_clean = (
        df_lab_data.pipe(rename_columns)
        .pipe(fix_dates)
        .dropna(subset=["date", "lens_type"])
        .assign(lens_type=lambda x: x["lens_type"].str.lower())
        .assign(lens_type=lambda x: x["lens_type"].replace({"repairs": "repair"}))
        .drop(["Lab/Invoice #", "FRAME INFO", "Contacted"], axis=1)
    )

    df_lab_clean  # [["date", "lens_type"]]
    return (df_lab_clean,)


@app.cell
def _(df_lab_clean):
    top_30_lens = (
        df_lab_clean["lens_type"]
        .str.lower()
        .value_counts()
        .reset_index()
        .sort_values(by=["count", "lens_type"], ascending=[False, False])
        .head(30)
    )

    top_30_lens
    return (top_30_lens,)


@app.cell
def _(pd):
    def filter_by_lens_list(df, lens_list):
        """Filters the dataframe to keep only the top 30 lenses in the df"""
        return df[df["lens_type"].isin(lens_list)].copy()


    def aggregate_quarterly(df):
        """
        Groups transactions by Quarter and Lens Type.
        """
        return (
            df.groupby(["lens_type", pd.Grouper(key="date", freq="QE")])
            .size()
            .reset_index(name="demand")
        )
    return aggregate_quarterly, filter_by_lens_list


@app.cell
def _(aggregate_quarterly, df_lab_clean, filter_by_lens_list, top_30_lens):
    unique_lens = top_30_lens["lens_type"].tolist()

    (time_series_data := df_lab_clean
        .pipe(filter_by_lens_list, lens_list=unique_lens)
        .pipe(aggregate_quarterly)
        .pivot(index="date", columns="lens_type", values="demand")
        .fillna(0)
    )

    (time_series_data := time_series_data.asfreq("QE-DEC"))
    return (time_series_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Arima Implementation
    """)
    return


@app.cell
def _(ARIMA, ConvergenceWarning, ValueWarning, warnings):
    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    warnings.simplefilter("ignore", category=ValueWarning)

    def fit_best_arima(series):
        """
        Simple Grid search p,d,q in {0,1} to find best order
        of autoregressive, differences, and moving components
        then forecast using best aic
        """

        best_aic = float("inf")
        best_model = None

        # Simplified grid search over precomputed p,d,q
        orders = [(0,0,0), (1,0,0), (0,1,0), (0,0,1), 
                  (1,1,0), (1,0,1), (0,1,1), (1,1,1)]

        for order in orders:
            try:
                model = ARIMA(series, order=order, missing='drop')
                results = model.fit()
                # print(f"Order {order}: AIC={results.aic}")
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_model = results
            except Exception as e:
                print(f"Failed to fit order {order}: {e}")
                continue

        # print(f"Best Model: {best_model} \tBest Order: {best_aic}")

        if best_model:
            return best_model

        print(f"WARNING: No suitable ARIMA model found for series. Defaulting to mean model (0,0,0).")
        # Fallback: Return a simple mean model if everything failed
        return ARIMA(series, order=(0,0,0)).fit()
    return (fit_best_arima,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Arima Rolling Evaluation
    """)
    return


@app.cell
def _(fit_best_arima, mean_absolute_error, mean_squared_error, np):
    from tqdm import tqdm

    def rolling_arima_eval(series, min_train=8):
        """
        Performs rolling origin evaluation using your existing predict_arima_next function.
        Extracts only the 1st step of the 4-step forecast for validation.

        min_train: minimum quarters needed for training. 8 was chosen since paper mentioned
            using 2 years of historical data
        """
        actuals = []
        forecasts = []

        for i in range(min_train, len(series)):
            train = series.iloc[:i]
            actual = series.iloc[i]

            arima_model = fit_best_arima(train)
            pred_val = arima_model.forecast(steps=1).iloc[0]

            forecasts.append(pred_val)
            actuals.append(actual)

        return {
            'MAE': mean_absolute_error(actuals, forecasts),
            'RMSE': np.sqrt(mean_squared_error(actuals, forecasts))
        }
    return rolling_arima_eval, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prophet Implementation
    """)
    return


@app.cell
def _(mean_absolute_error, mean_squared_error, np, warnings):
    from prophet import Prophet
    import logging

    # # Suppress Prophet logging to keep output clean
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=FutureWarning, module="prophet")
    def predict_prophet_next(series):
        """
        Fits Prophet with paper-specific priors and forecasts 1 step ahead.
        """
        df = series.reset_index()
        df.columns = ['ds', 'y']

        # Paper specs: changepoint_prior_scale=0.01, seasonality_prior_scale=10
        m = Prophet(
            changepoint_prior_scale=0.01, 
            seasonality_prior_scale=10.0,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        m.fit(df)

        # Forecast next quarter
        future = m.make_future_dataframe(periods=1, freq='Q')
        forecast = m.predict(future)

        return forecast['yhat'].iloc[-1]

    def rolling_prophet_eval(series, min_train=8):
        actuals = []
        forecasts = []

        for i in range(min_train, len(series)):
            train = series.iloc[:i]
            actual = series.iloc[i]

            pred_val = predict_prophet_next(train)

            forecasts.append(pred_val)
            actuals.append(actual)

        return {
            'MAE': mean_absolute_error(actuals, forecasts),
            'RMSE': np.sqrt(mean_squared_error(actuals, forecasts))
        }
    return (rolling_prophet_eval,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forecasting Training Loop
    """)
    return


@app.cell
def _(pd, rolling_arima_eval, rolling_prophet_eval, time_series_data, tqdm):
    results = []

    # Single loop for both models
    for col in tqdm(time_series_data.columns, desc="Processing Lenses"):
        series = time_series_data[col]
        total_demand = series.sum()

        # Fit ARIMA model
        arima_res = rolling_arima_eval(series)

        # Fit Prophet
        prophet_res = rolling_prophet_eval(series)

        # Ref: Table I shows 'Difference' columns 
        diff_mae = prophet_res['MAE'] - arima_res['MAE']
        diff_rmse = prophet_res['RMSE'] - arima_res['RMSE']

        # Obtain results for Table 1.
        results.append({
            "Combination": col,
            "Total Demand": total_demand,

            # Prophet Cols
            "Prophet MAE": prophet_res['MAE'],
            "Prophet RMSE": prophet_res['RMSE'],

            # ARIMA Cols
            "ARIMA MAE": arima_res['MAE'],
            "ARIMA RMSE": arima_res['RMSE'],

            # Difference Cols
            "Diff MAE": diff_mae,
            "Diff RMSE": diff_rmse
        })

    # Create final DataFrame
    comparison_df = pd.DataFrame(results).round(3)
    comparison_df = comparison_df.sort_values(by="Total Demand", ascending=False)

    comparison_df.head()
    return (comparison_df,)


@app.cell
def _(comparison_df):
    target_lenses = [
        "prog, trans",
        "sv, bb",
        "sv, trans, bb",
        "repair",
        "sv, poly",
        "sv, clear"
    ]

    # 2. Filter the dataframe to only show these rows
    (selected_rows := comparison_df[comparison_df['Combination'].isin(target_lenses)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linear Programming Model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cost Data Cleaning
    Before creating our linear programming mode, we will clean the 'Cost Data' sheet from the excel.
    """)
    return


@app.cell
def _(pd):
    cost_df = pd.read_excel("Lens Data.xlsx", sheet_name="Cost data")
    cost_df = (
        cost_df
        .pipe(lambda df: df.rename(columns={df.columns[0]: "lens_type"}))
        .rename(columns={
            "ordering_cost_TTD": "direct_cost",
            "middleman_fee": "middleman_cost",
            "holding_cost_per_unit": "holding_cost",
            "Description": "description",
            "Overall Demand": "demand"
        })
        .assign(lens_type=lambda df: (
            df["lens_type"]
            .astype(str)
            .str.lstrip("0123456789 \t")
            .str.strip() 
            )
        )
        .set_index("lens_type")
    )

    cost_df
    return (cost_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forecasting for next 4 Quarters
    """)
    return


@app.cell
def _(cost_df, fit_best_arima, pd, time_series_data):
    # %%
    # Function to predict next 4 quarters using your optimized ARIMA
    def predict_next_4_quarters(series):
        model_fit = fit_best_arima(series)
        forecast = model_fit.forecast(steps=4)

        return forecast.values
    # 1. Generate forecasts for all lenses
    future_forecasts = {}

    print("Forecasting next 4 quarters for inventory planning...")
    for cols in time_series_data.columns:
        arima_forecasts = predict_next_4_quarters(time_series_data[cols])
        future_forecasts[cols] = arima_forecasts

    # 2. Create the Demand DataFrame (Rows=Quarters, Cols=Lenses)
    demand_df = pd.DataFrame(future_forecasts)

    # # Set a clean index for the 4 quarters (e.g., Q1, Q2, Q3, Q4)
    demand_df.index = [f"Q{i+1}" for i in range(len(demand_df))]
    demand_df.columns = demand_df.columns.str.strip()
    demand_df.round(3)
    # # 3. CRITICAL: Align Cost and Demand Data
    # # We only want lenses that exist in BOTH forecasts and cost data
    common_lenses = demand_df.columns.intersection(cost_df.index)
    common_lenses

    # demand_df = demand_df[common_lenses]
    (cost_df_clean := cost_df.loc[common_lenses])

    # print(f"\nAligned Data: {len(common_lenses)} lenses ready for optimization.")
    # demand_df.head()
    return


@app.cell
def _():
    import pulp
    from pulp import LpMaximize, LpProblem, LpVariable, lpSum

    def optimize_inventory(demand, costs, budget):
        quarters = demand.index
        lenses = demand.columns
        model = LpProblem(f"Optical_Inventory_{budget}", LpMaximize)
    
        # Variables
        q = LpVariable.dicts("Order", (quarters, lenses), lowBound=0, cat='Integer')
        f = LpVariable.dicts("Fulfill", (quarters, lenses), lowBound=0, cat='Integer')
        s = LpVariable.dicts("Stock", (quarters, lenses), lowBound=0, cat='Integer')

        # Objective Terms
        objective_terms = []
        for t in quarters:
            for k in lenses:
                if k not in costs.index: continue
            
                D_tk = float(demand.loc[t, k])
                Cc = float(costs.loc[k, 'direct_cost'])
                Cm = float(costs.loc[k, 'middleman_cost'])
                Ch = float(costs.loc[k, 'holding_cost'])
            
                rev = 0.10 * Cc * f[t][k]
                penalty = Cm * (D_tk - f[t][k])
                hold = Ch * s[t][k]
            
                objective_terms.append(rev - penalty - hold)
    
        model += lpSum(objective_terms)

        # Constraints
        for t_idx, t in enumerate(quarters):
            # Budget
            daily_spend = [q[t][k] * float(costs.loc[k, 'direct_cost']) for k in lenses]
            model += lpSum(daily_spend) <= budget
        
            for k in lenses:
                safe_k = str(k).replace(" ", "_").replace(",", "_").replace(".", "_")
                D_tk = float(demand.loc[t, k])
                prev = 0 if t_idx == 0 else s[quarters[t_idx-1]][k]
            
                model += s[t][k] == prev + q[t][k] - f[t][k], f"Bal_{t}_{safe_k}"
                model += f[t][k] <= D_tk, f"MaxDem_{t}_{safe_k}"
                model += f[t][k] <= prev + q[t][k], f"MaxStock_{t}_{safe_k}"

        model.solve(pulp.PULP_CBC_CMD(msg=0))
    
        # --- 3. EXTRACT METRICS FOR TABLE II ---
        # We calculate these manually from the solver results
        total_direct_cost = 0  # "Budget Utilized"
        total_middleman_cost = 0
        total_middleman_units = 0
        total_holding_cost = 0
    
        for t in quarters:
            for k in lenses:
                if k not in costs.index: continue
            
                # Get values
                q_val = q[t][k].varValue
                f_val = f[t][k].varValue
                s_val = s[t][k].varValue
                D_val = float(demand.loc[t, k])
            
                # Costs
                Cc = float(costs.loc[k, 'direct_cost'])
                Cm = float(costs.loc[k, 'middleman_cost'])
                Ch = float(costs.loc[k, 'holding_cost'])
            
                # Accumulate Metrics
                total_direct_cost += q_val * Cc
            
                # Unfulfilled demand = Middleman usage
                # Note: D_val - f_val might be 0.000001 due to floats, so we max(0, ...)
                unfulfilled = max(0, D_val - f_val)
                total_middleman_units += unfulfilled
                total_middleman_cost += unfulfilled * Cm
            
                total_holding_cost += s_val * Ch

        # Total Cost defined in Table III as: Direct + Middleman + Holding
        total_cost_overall = total_direct_cost + total_middleman_cost + total_holding_cost

        return {
            "profit": pulp.value(model.objective),
            "total_cost_overall": total_cost_overall,
            "total_middleman_cost": total_middleman_cost,
            "total_middleman_units": total_middleman_units,
            "total_direct_cost": total_direct_cost,
            "status": pulp.LpStatus[model.status]
        }
    return (optimize_inventory,)


@app.cell
def _(cost_df, fit_best_arima, optimize_inventory, pd, time_series_data):

    cutoff_date = "2023-09-30"
    clean_history = time_series_data.loc[:cutoff_date]

    print(f"Original Training Data: {time_series_data.shape}")
    print(f"Fixed Training Data:    {clean_history.shape}")

    # --- STEP 2: RE-GENERATE FORECASTS ---
    # Uses your fit_best_arima function on the CLEAN data
    future_forecasts_clean = {}

    print("Regenerating forecasts...")
    for colss in clean_history.columns:
        # Train on clean_history, not time_series_data
        model = fit_best_arima(clean_history[colss])
        future_forecasts_clean[colss] = model.forecast(steps=4).values

    # Create new, correct Demand DataFrame
    clean_ts_df = pd.DataFrame(future_forecasts_clean)
    clean_ts_df.columns = clean_ts_df.columns.str.strip()
    clean_ts_df.index = ["Q1", "Q2", "Q3", "Q4"]


    clean_ts_df.round(3)
    # Verify the fix (Should be ~7.18 now, NOT 0)
    # # print(f"New Forecast for 'prog, trans': {clean_ts_df['prog, trans'].values}")

    # # --- STEP 3: RUN OPTIMIZATION WITH FIXED DEMAND ---
    # # Now pass this new dataframe to your optimization function
    # print("\nRunning Optimization with Fixed Demand...")

    # # # Ensure strict integer handling function is defined (from previous step)
    res_30k = optimize_inventory(clean_ts_df, cost_df, 30000)
    res_50k = optimize_inventory(clean_ts_df, cost_df, 50000)

    res_30k



    return (clean_ts_df,)


@app.cell
def _(clean_ts_df, cost_df, optimize_inventory, pd):
    budgets = [10000, 20000, 30000, 40000, 50000]
    model_results = []

    print("Running Optimization with Descriptions...")
    for b in budgets:
        res = optimize_inventory(clean_ts_df, cost_df, b)
        model_results.append(res)

    pd.DataFrame(model_results).round(3)
    return


@app.cell
def _(clean_ts_df):
    clean_ts_df.round(3)
    return


if __name__ == "__main__":
    app.run()
