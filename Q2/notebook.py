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
    return (pd,)


@app.cell
def _(pd):
    df_lab_data = pd.read_excel("./Lens Data.xlsx", sheet_name="Lab Data")
    df_lab_data
    return (df_lab_data,)


@app.cell
def _(pd):
    ## Data Cleaning Functions


    def rename_columns(df):
        return df.rename(
            columns={
                "Date of write up": "date",
                "Job Description": "lens_type",
            }
        )


    def fix_dates(df):
        df["date"] = pd.to_datetime(
            df["date"], format="mixed", dayfirst=True, errors="coerce"
        )
        return df
    return fix_dates, rename_columns


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
            # .rename(columns={'date': 'ds'})
        )
    return aggregate_quarterly, filter_by_lens_list


@app.cell
def _(aggregate_quarterly, df_lab_clean, filter_by_lens_list, top_30_lens):
    unique_lens = top_30_lens["lens_type"].tolist()

    df_timeseries = df_lab_clean.pipe(
        filter_by_lens_list, lens_list=unique_lens
    ).pipe(aggregate_quarterly)

    df_timeseries
    return (df_timeseries,)


@app.cell
def _(df_timeseries):
    time_series_data = df_timeseries.pivot(
        index="date", columns="lens_type", values="demand"
    ).fillna(0)
    time_series_data
    return (time_series_data,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Arima Implementation
    """)
    return


@app.cell
def _():
    import numpy as np
    import itertools
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import warnings


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
                model = ARIMA(series, order=order)
                results = model.fit()
                # print(f"Order {order}: AIC={results.aic}")
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_model = results
            except Exception as e:
                print(f"Failed to fit order {order}: {e}")
                continue

        print(f"Best Model: {best_model} \tBest Order: {best_aic}")

        if best_model:
            return best_model

        print(f"WARNING: No suitable ARIMA model found for series. Defaulting to mean model (0,0,0).")
        # Fallback: Return a simple mean model if everything failed
        return ARIMA(series, order=(0,0,0)).fit()
    return fit_best_arima, mean_absolute_error, mean_squared_error, np


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
def _(mean_absolute_error, mean_squared_error, np):
    from prophet import Prophet
    import logging

    # Suppress Prophet logging to keep output clean
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

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


@app.cell(hide_code=True)
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
    )

    cost_df.head()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forecasting for next 4 Quarters
    """)
    return


@app.cell
def _(time_series_data):
    print("Inspecting 'prog, trans' data:")
    if "prog, trans" in time_series_data.columns:
        pt_series = time_series_data["prog, trans"]
        print(pt_series)
        print(f"Total Demand: {pt_series.sum()}")
    else:
        print("'prog, trans' not found in time_series_data columns")
    return


@app.cell(hide_code=True)
def _(fit_best_arima, pd, time_series_data):
    # %%
    # Function to predict next 4 quarters using your optimized ARIMA
    def predict_next_4_quarters(series):
        model_fit = fit_best_arima(series)
        forecast = model_fit.forecast(steps=16)
    
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
    demand_df.round(3)
    # # 3. CRITICAL: Align Cost and Demand Data
    # # We only want lenses that exist in BOTH forecasts and cost data
    # common_lenses = demand_df.columns.intersection(cost_df.index)

    # demand_df = demand_df[common_lenses]
    # cost_df_clean = cost_df.loc[common_lenses]

    # print(f"\nAligned Data: {len(common_lenses)} lenses ready for optimization.")
    # demand_df.head()
    return


@app.cell
def _(pd):
    import pulp
    from pulp import LpMaximize, LpProblem, LpVariable, lpSum

    def optimize_inventory(demand_df, cost_df, quarterly_budget=30000):
        """
        Optimizes lens inventory orders to maximize profit.
        Ref: Section III-D, Eq (1) - (4)
    
        Args:
            demand_df (pd.DataFrame): Forecasted demand (Index=Quarters, Cols=LensTypes).
            cost_df (pd.DataFrame): Costs per lens (Index=LensType, Cols=['direct_cost', 'middleman_cost', 'holding_cost']).
            quarterly_budget (float): Budget constraint per quarter (B_t).
        
        Returns:
            dict: Contains solution status and dataframes for Orders (q), Fulfilled (f), and Stock (s).
        """
        # Sets
        quarters = demand_df.index
        lenses = demand_df.columns
    
        # Initialize Model
        model = LpProblem("Optical_Lens_Inventory_Optimization", LpMaximize)
    
        # --- Decision Variables ---
        # q: Quantity ordered directly [cite: 138]
        q = LpVariable.dicts("Order", (quarters, lenses), lowBound=0, cat='Integer')
    
        # f: Quantity fulfilled from direct stock [cite: 139]
        f = LpVariable.dicts("Fulfill", (quarters, lenses), lowBound=0, cat='Integer')
    
        # s: Inventory level at end of quarter [cite: 136]
        s = LpVariable.dicts("Stock", (quarters, lenses), lowBound=0, cat='Integer')

        # Parameters
        profit_margin = 0.10  # pi = 0.10 [cite: 140]
    
        # --- Objective Function --- 
        # Maximize: Profit from Direct Sales - Middleman Costs for Unfulfilled - Holding Costs
        # Ref: Equation (1) 
        objective_terms = []
    
        for t in quarters:
            for k in lenses:
                D_tk = demand_df.loc[t, k] # Forecasted Demand
            
                # Costs
                Cc = cost_df.loc[k, 'direct_cost']
                Cm = cost_df.loc[k, 'middleman_cost']
                Ch = cost_df.loc[k, 'holding_cost']
            
                # Profit Term: pi * Cc * f_tk
                profit_term = profit_margin * Cc * f[t][k]
            
                # Middleman Cost Term: Cm * (D_tk - f_tk)
                # Since D_tk is constant, we minimize Cm * (D_tk - f_tk)
                # equivalent to Maximizing +Cm * f_tk in the objective (and subtracting the constant Cm*D_tk constant later if needed)
                middleman_cost = Cm * (D_tk - f[t][k])
            
                # Holding Cost Term: Ch * s_tk
                holding_term = Ch * s[t][k]
            
                objective_terms.append(profit_term - middleman_cost - holding_term)
            
        model += lpSum(objective_terms), "Total_Profit"

        # --- Constraints ---
    
        for t_idx, t in enumerate(quarters):
        
            # 2. Budget Constraint [cite: 151]
            # Sum of (Order Qty * Direct Cost) <= Budget
            daily_spend = [q[t][k] * cost_df.loc[k, 'direct_cost'] for k in lenses]
            model += lpSum(daily_spend) <= quarterly_budget, f"Budget_{t}"
        
            for k in lenses:
                D_tk = demand_df.loc[t, k]
            
                # 1. Inventory Balance [cite: 148]
                # s_t = s_{t-1} + q_t - f_t
                if t_idx == 0:
                    prev_stock = 0 # Assume 0 start [cite: 149]
                else:
                    prev_t = quarters[t_idx - 1]
                    prev_stock = s[prev_t][k]
            
                model += s[t][k] == prev_stock + q[t][k] - f[t][k], f"Inv_Bal_{t}_{k}"
            
                # 3. Fulfillment Bounds [cite: 153, 154]
                # Cannot fulfill more than Demand
                model += f[t][k] <= D_tk, f"Max_Dem_{t}_{k}"
            
                # Cannot fulfill more than Available (Prev Stock + New Order)
                model += f[t][k] <= prev_stock + q[t][k], f"Max_Avail_{t}_{k}"

        # --- Solve ---
        # Using default solver (CBC)
        model.solve()
    
        # --- Extract Results ---
        print(f"Status: {pulp.LpStatus[model.status]}")
    
        # Helper to pull variable values into DF
        def extract_var(var_dict):
            data = {k: [var_dict[t][k].varValue for t in quarters] for k in lenses}
            return pd.DataFrame(data, index=quarters)

        return {
            "status": pulp.LpStatus[model.status],
            "objective": pulp.value(model.objective),
            "orders": extract_var(q),
            "fulfilled": extract_var(f),
            "stock": extract_var(s)
        }
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
