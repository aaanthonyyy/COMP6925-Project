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
    import marimo as mo
    import pandas as pd
    import numpy as np
    import warnings
    import pulp

    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_absolute_error, mean_squared_error
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
        pulp,
        warnings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Preprocessing
    """)
    return


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
    return aggregate_quarterly, filter_by_lens_list, fix_dates, rename_columns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lab Data
    """)
    return


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

    df_lab_clean.head()
    return (df_lab_clean,)


@app.cell
def _(aggregate_quarterly, df_lab_clean, filter_by_lens_list):
    top_30_lens = (
        df_lab_clean["lens_type"]
        .str.lower()
        .value_counts()
        .reset_index()
        .sort_values(by=["count", "lens_type"], ascending=[False, False])
        .head(30)
    )

    unique_lens = top_30_lens["lens_type"].tolist()

    time_series_data = (
        df_lab_clean
        .pipe(filter_by_lens_list, lens_list=unique_lens)
        .pipe(aggregate_quarterly)
        .assign(lens_type=lambda x: x["lens_type"].str.strip())
        .pivot(index="date", columns="lens_type", values="demand")
        .fillna(0)
        .asfreq("QE-DEC")
    )

    time_series_data.head()
    return (time_series_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cost Data
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

    cost_df.head()
    return (cost_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Forecasting
    """)
    return


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
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_model = results
            except Exception as e:
                print(f"Failed to fit order {order}: {e}")
                continue


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
            pred_val = arima_model.forecast(steps=1).iloc[0].round()

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

    # Filter the results to match Table 1. from paper.
    (selected_rows := comparison_df[comparison_df['Combination'].isin(target_lenses)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forecasting Additional Quarters
    """)
    return


@app.cell
def _(fit_best_arima, pd, time_series_data):
    # 1. Ensure input has a frequency (Crucial for auto-date generation)
    if time_series_data.index.freq is None:
        time_series_data.index.freq = pd.infer_freq(time_series_data.index)


    forecast_steps = 4
    future_forecasts = {
        col: fit_best_arima(time_series_data[col]).forecast(steps=forecast_steps)
        for col in time_series_data.columns
    }


    arima_forecast = (
        pd.DataFrame(future_forecasts)
        .round(0)
        .clip(lower=0)
        .astype(int)
    )


    timeseries_arima = (
        pd.concat([time_series_data, arima_forecast])
        .fillna(0)
        .astype(int)
    )

    timeseries_arima.tail(forecast_steps)
    return (arima_forecast,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linear Programming Model
    """)
    return


@app.cell
def _(pd, pulp):
    def optimize_inventory(demand_df, cost_df, budget_per_quarter):
        """
        Optimizes lens inventory procurement.
        Robust version: Handles bad data (NaNs) and failed solver states safely.
        """

        # Fill NaNs with 0 to prevent solver crashes
        demand_df = demand_df.fillna(0)
        cost_df = cost_df.fillna(0)

        # Check orientation
        common_cols = len(demand_df.columns.intersection(cost_df.index))
        common_index = len(demand_df.index.intersection(cost_df.index))

        if common_cols > common_index:
            print("Note: Transposing demand_df to match (Products x Quarters) format...")
            demand_df = demand_df.transpose()


        # Setup Parameters
        quarter_cols = demand_df.columns.tolist()
        quarter_keys = [str(q) for q in quarter_cols]
        time_map = dict(zip(quarter_keys, quarter_cols))

        products = demand_df.index.tolist()

        # Filter valid products
        valid_products = [p for p in products if p in cost_df.index]
        dropped = len(products) - len(valid_products)
        if dropped > 0:
            print(f"Warning: Dropped {dropped} items from demand that were missing in cost table.")
        products = valid_products

        profit_margin = 0.10
        prob = pulp.LpProblem("Optical_Lens_Inventory_Optimization", pulp.LpMaximize)


        # Define Variables
        keys = [(q, p) for q in quarter_keys for p in products]

        q_vars = pulp.LpVariable.dicts("Order_Qty", keys, lowBound=0, cat='Integer')
        f_vars = pulp.LpVariable.dicts("Fulfilled_Direct", keys, lowBound=0, cat='Integer')
        s_vars = pulp.LpVariable.dicts("Stock_End", keys, lowBound=0, cat='Integer')

    
        # Objective Function
        objective_terms = []

        for t_str in quarter_keys:
            t_original = time_map[t_str]
            for k in products:
                D_tk = float(demand_df.at[k, t_original])
                C_c = float(cost_df.at[k, 'direct_cost'])
                C_m = float(cost_df.at[k, 'middleman_cost'])
                C_h = float(cost_df.at[k, 'holding_cost'])

                f_tk = f_vars[(t_str, k)]
                s_tk = s_vars[(t_str, k)]

                # Profit Terms
                direct_profit = profit_margin * C_c * f_tk
                middleman_cost = C_m * (D_tk - f_tk)
                holding_cost = C_h * s_tk

                objective_terms.append(direct_profit - middleman_cost - holding_cost)

        prob += pulp.lpSum(objective_terms), "Total_Profit"

        # Constraints
        for t_idx, t_str in enumerate(quarter_keys):
            t_original = time_map[t_str]

            # Budget Constraint
            quarterly_spend = [q_vars[(t_str, k)] * float(cost_df.at[k, 'direct_cost']) for k in products]
            prob += pulp.lpSum(quarterly_spend) <= budget_per_quarter, f"Budget_{t_str}"

            for k in products:
                # Previous Inventory
                if t_idx == 0:
                    s_prev = 0
                else:
                    prev_q_str = quarter_keys[t_idx - 1]
                    s_prev = s_vars[(prev_q_str, k)]

                # Inventory Balance
                prob += s_vars[(t_str, k)] == s_prev + q_vars[(t_str, k)] - f_vars[(t_str, k)], f"Inv_Bal_{t_str}_{k}"

                # Fulfillment Constraints
                D_tk = float(demand_df.at[k, t_original])
                prob += f_vars[(t_str, k)] <= D_tk, f"Max_Demand_{t_str}_{k}"
                prob += f_vars[(t_str, k)] <= s_prev + q_vars[(t_str, k)], f"Max_Stock_{t_str}_{k}"

        # Solve Model
        # Use a generous time limit to prevent "No Solution" errors on complex data
        solver = pulp.PULP_CBC_CMD(msg=True, gapRel=0.1, timeLimit=30)
        prob.solve(solver)

        # SAFETY CHECK
        status_str = pulp.LpStatus[prob.status]
        if prob.status != pulp.LpStatusOptimal:
            print(f"\nCRITICAL WARNING: Solver failed with status '{status_str}'. Results may be empty.")

        results = []

        def safe_val(var):
            val = pulp.value(var)
            return val if val is not None else 0.0

        for t_str in quarter_keys:
            t_original = time_map[t_str]
            for k in products:
            
                # Safe extraction
                val_q = safe_val(q_vars[(t_str, k)])
                val_f = safe_val(f_vars[(t_str, k)])
                val_s = safe_val(s_vars[(t_str, k)])
                val_d = float(demand_df.at[k, t_original])
                val_m = max(0, val_d - val_f)

                # Costs
                c_direct = float(cost_df.at[k, 'direct_cost'])
                c_middle = float(cost_df.at[k, 'middleman_cost'])
                c_holding = float(cost_df.at[k, 'holding_cost'])

                # Financials
                spend_direct = val_q * c_direct
                spend_middle = val_m * c_middle
                spend_holding = val_s * c_holding
                total_item_cost = spend_direct + spend_middle + spend_holding

                results.append({
                    'Quarter': t_original,
                    'Product': k,
                    'Demand': val_d,
                    'Direct_Order': val_q,
                    'Fulfilled_Direct': val_f,
                    'Middleman_Units': val_m,
                    'End_Stock': val_s,

                    # Financial Columns
                    'Budget_Utilized_($)': spend_direct,
                    'Middleman_Cost_($)': spend_middle,
                    'Holding_Cost_($)': spend_holding,
                    'Total_Cost_($)': total_item_cost
                })

        return {
            'status': status_str,
            'total_profit': pulp.value(prob.objective) if pulp.value(prob.objective) else 0.0,
            'results_df': pd.DataFrame(results).set_index("Quarter")
        }
    return (optimize_inventory,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Allocation
    Solving for Table 2.
    """)
    return


@app.cell
def _(arima_forecast, cost_df, optimize_inventory, pd):
    summary_data = []

    print("Running Optimization across scenarios...")
    budgets = [10000, 20000, 30000, 40000, 50000]

    for b in budgets:
        # 1. Run the optimization
        # (Ensure you are using the function that returns 'results_df')
        output = optimize_inventory(arima_forecast, cost_df, b)
    
        # 2. Get the detailed breakdown dataframe
        df_res = output['results_df']
    
        # 3. Calculate the aggregate totals
        total_profit = output['total_profit']
    
        # Summing up the specific columns we created earlier
        budget_utilized = df_res['Budget_Utilized_($)'].sum()
        middleman_cost = df_res['Middleman_Cost_($)'].sum()
        holding_cost = df_res['Holding_Cost_($)'].sum()
        middleman_units = df_res['Middleman_Units'].sum()
    
        # Total Cost = Direct Spend + Middleman Fees + Holding Costs
        total_cost = budget_utilized + middleman_cost + holding_cost
    
        # 4. Calculate Profit per $10k Invested
        # Formula: Profit / (Money Spent / 10,000)
        if budget_utilized > 0:
            profit_per_10k = total_profit / (budget_utilized / 10000)
        else:
            profit_per_10k = 0
        
        # 5. Append to summary list
        summary_data.append({
            "Budget Scenario": b,
            "Total Profit ($)": total_profit,
            "Total Cost ($)": total_cost,
            "Middleman Cost ($)": middleman_cost,
            "Middleman Units": middleman_units,
            "Budget Utilized ($)": budget_utilized,
            "Profit/$10k": profit_per_10k,
            "Status": output['status']
        })

    # 6. Create and Print the Final DataFrame
    comparison_table = pd.DataFrame(summary_data).round(2)
    comparison_table
    return


@app.cell
def _(combined_df, cost_df, optimize_inventory_new):
    solution_new = optimize_inventory_new(combined_df, cost_df, 10000)
    solution_new['results_df'].set_index('Quarter').sum()
    return


if __name__ == "__main__":
    app.run()
