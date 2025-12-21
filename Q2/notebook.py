# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars==1.35.2",
#     "pulp[highs]==3.3.0",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import warnings
    import pulp
    import re

    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
    return (
        ARIMA,
        ConvergenceWarning,
        ValueWarning,
        mean_absolute_error,
        mean_squared_error,
        np,
        pd,
        pulp,
        warnings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part A: Replicating trends from paper
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Preprocessing
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
            }
        )


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
        .drop(["Lab/Invoice #", "FRAME INFO", "Contacted"], axis=1)
    )

    df_lab_clean["lens_type"] = (
        df_lab_clean["lens_type"]
        .str.lower()
        # .str.strip()
        # .str.replace(r"\s*,\s*", ", ", regex=True)
    )

    df_lab_clean.head()
    return (df_lab_clean,)


@app.cell
def _(aggregate_quarterly, df_lab_clean, filter_by_lens_list):
    top_30_lens = (
        df_lab_clean["lens_type"]
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
        .pivot(index="date", columns="lens_type", values="demand")
        .fillna(0)
        .asfreq("QE-DEC")
        # .tail(18)
    )

    time_series_data
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
        cost_df.pipe(lambda df: df.rename(columns={df.columns[0]: "lens_type"}))
        .rename(
            columns={
                "ordering_cost_TTD": "direct_cost",
                "middleman_fee": "middleman_cost",
                "holding_cost_per_unit": "holding_cost",
                "Description": "description",
                "Overall Demand": "demand",
            }
        )
        .assign(
            lens_type=lambda df: (
                df["lens_type"].astype(str).str.lstrip("0123456789 \t").str.strip()
            )
        )
        .set_index("lens_type")
    )

    cost_df.head()
    return (cost_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forecasting
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Arima Implementation
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
        orders = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ]

        for order in orders:
            try:
                model = ARIMA(series, order=order, missing="drop")
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_model = results
            except Exception as e:
                print(f"Failed to fit order {order}: {e}")
                continue

        if best_model:
            return best_model

        print(
            f"WARNING: No suitable ARIMA model found for series. Defaulting to mean model (0,0,0)."
        )

        # Fallback: Return a simple mean model if everything failed
        # return ARIMA(series, order=(0, 0, 0)).fit()
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
            "MAE": mean_absolute_error(actuals, forecasts),
            "RMSE": np.sqrt(mean_squared_error(actuals, forecasts)),
        }
    return rolling_arima_eval, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prophet Implementation
    """)
    return


@app.cell
def _(mean_absolute_error, mean_squared_error, np, warnings):
    from prophet import Prophet
    import logging

    # # Suppress Prophet logging to keep output clean
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=FutureWarning, module="prophet")


    def predict_prophet_next(series, steps=1):
        """
        Fits Prophet with paper-specific priors and forecasts 1 step ahead.
        """
        df = series.reset_index()
        df.columns = ["ds", "y"]

        # Paper specs: changepoint_prior_scale=0.01, seasonality_prior_scale=10
        m = Prophet(
            changepoint_prior_scale=0.01,
            seasonality_prior_scale=10.0,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        m.fit(df)

        # Forecast next quarter
        future = m.make_future_dataframe(periods=steps, freq="Q")
        forecast = m.predict(future)

        return forecast["yhat"].iloc[-1]


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
            "MAE": mean_absolute_error(actuals, forecasts),
            "RMSE": np.sqrt(mean_squared_error(actuals, forecasts)),
        }
    return predict_prophet_next, rolling_prophet_eval


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Forecasting Training Loop
    """)
    return


@app.cell
def _():
    # results = []
    # target_lenses = [
    #     "prog, trans",
    #     "sv, bb",
    #     "sv, trans, bb",
    #     "repair",
    #     "sv, poly",
    #     "sv, clear",
    # ]

    # # To match the demand from the paper, the last 18 quarters are used
    # timeseries_18 = (
    #     time_series_data
    #     .tail(18)
    #     .copy()
    # )


    # lenses_to_process = [l for l in target_lenses if l in timeseries_18.columns]

    # # Single loop for both models
    # for col in tqdm(lenses_to_process, desc="Processing Lenses"):
    #     series = timeseries_18[col]
    #     total_demand = series.sum()

    #     # Fit ARIMA model
    #     arima_res = rolling_arima_eval(series)

    #     # Fit Prophet
    #     prophet_res = rolling_prophet_eval(series)

    #     # Ref: Table I shows 'Difference' columns
    #     diff_mae = prophet_res["MAE"] - arima_res["MAE"]
    #     diff_rmse = prophet_res["RMSE"] - arima_res["RMSE"]

    #     # Obtain results for Table 1.
    #     results.append(
    #         {
    #             "Combination": col,
    #             "Total Demand": total_demand,
    #             # Prophet Cols
    #             "Prophet MAE": prophet_res["MAE"],
    #             "Prophet RMSE": prophet_res["RMSE"],
    #             # ARIMA Cols
    #             "ARIMA MAE": arima_res["MAE"],
    #             "ARIMA RMSE": arima_res["RMSE"],
    #             # Difference Cols
    #             "Diff MAE": diff_mae,
    #             "Diff RMSE": diff_rmse,
    #         }
    #     )

    # # Create final DataFrame
    # comparison_df = pd.DataFrame(results).round(3)
    # comparison_df = comparison_df.sort_values(by="Total Demand", ascending=False)

    # comparison_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Forecasting Additional Quarters
    """)
    return


@app.cell
def _(fit_best_arima, pd, predict_prophet_next):
    def generate_forecast_scenario(history_df, steps=4, method="arima", custom_func=None):
        """
        Generates a full inventory scenario (History + Forecast).
        Uses helper functions for model-specific prediction logic.
        """
        # 1. Ensure frequency exists
        # if history_df.index.freq is None:
        history_df.index.freq = pd.infer_freq(history_df.index)

        future_vals = {}
        print(
            f"Generating {method.upper()} forecasts for {len(history_df.columns)} products..."
        )

        for col in history_df.columns:
            series = history_df[col]

            if method == "arima":
                # Use ARIMA helper
                model = fit_best_arima(series)
                forecast_values = model.forecast(steps=steps).values

            elif method == "prophet":
                # Use Prophet helper (Refactored call)
                forecast_values = predict_prophet_next(series, steps=steps)

            elif method == "sarima" and custom_func:
                # use SARIMA grid search helper
                model = custom_func(series)
                forecast_values = model.forecast(steps=steps).values

            else:
                raise ValueError(f"Method {method} not supported or custom_func missing.")

            future_vals[col] = forecast_values

        # --- SHARED CLEANING & MERGING ---
        # 1. Generate future Date Index
        last_date = history_df.index[-1]
        freq = history_df.index.freq
        future_dates = pd.date_range(
            start=last_date, periods=steps + 1, freq=freq
        )[1:]

        # 2. Create & Clean Forecast DataFrame
        forecast_df = (
            pd.DataFrame(future_vals, index=future_dates)
            .clip(lower=0)
            .round(4)
            .astype(int)
        )

        return forecast_df
    return (generate_forecast_scenario,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Linear Programming Model
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
            print(
                "Note: Transposing demand_df to match (Products x Quarters) format..."
            )
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
            print(
                f"Warning: Dropped {dropped} items from demand that were missing in cost table."
            )
        products = valid_products

        profit_margin = 0.10
        prob = pulp.LpProblem(
            "Optical_Lens_Inventory_Optimization", pulp.LpMaximize
        )

        # Define Variables
        keys = [(q, p) for q in quarter_keys for p in products]

        q_vars = pulp.LpVariable.dicts(
            "Order_Qty", keys, lowBound=0, cat="Integer"
        )
        f_vars = pulp.LpVariable.dicts(
            "Fulfilled_Direct", keys, lowBound=0, cat="Integer"
        )
        s_vars = pulp.LpVariable.dicts(
            "Stock_End", keys, lowBound=0, cat="Integer"
        )
        m_vars = pulp.LpVariable.dicts(
            "Middleman_Stock", keys, lowBound=0, cat="Integer"
        )

        # Objective Function
        objective_terms = []

        # t_str - time period
        for t_str in quarter_keys:
            t_original = time_map[t_str]
            # k - combination index
            for k in products:
                D_tk = float(demand_df.at[k, t_original])
                C_c = float(cost_df.at[k, "direct_cost"])
                C_m = float(cost_df.at[k, "middleman_cost"])
                C_h = float(cost_df.at[k, "holding_cost"])

                f_tk = f_vars[(t_str, k)]
                s_tk = s_vars[(t_str, k)]
                m_tk = m_vars[(t_str, k)]

                prob += m_tk >= D_tk - f_tk

                # Profit Terms
                direct_profit = profit_margin * C_c * f_tk
                middleman_cost = C_m * m_tk 
                holding_cost = C_h * s_tk

                objective_terms.append(
                    direct_profit - middleman_cost - holding_cost
                )

        prob += pulp.lpSum(objective_terms), "Total_Profit"

        # Constraints
        for t_idx, t_str in enumerate(quarter_keys):
            t_original = time_map[t_str]

            # Budget Constraint
            quarterly_spend = [
                q_vars[(t_str, k)] * float(cost_df.at[k, "direct_cost"])
                for k in products
            ]
            prob += (
                pulp.lpSum(quarterly_spend) <= budget_per_quarter,
                f"Budget_{t_str}",
            )

            for k in products:
                # Previous Inventory
                if t_idx == 0:
                    s_prev = 0
                else:
                    prev_q_str = quarter_keys[t_idx - 1]
                    s_prev = s_vars[(prev_q_str, k)]

                # Inventory Balance
                prob += (
                    s_vars[(t_str, k)]
                    == s_prev + q_vars[(t_str, k)] - f_vars[(t_str, k)],
                    f"Inv_Bal_{t_str}_{k}",
                )

                # Fulfillment Constraints
                D_tk = float(demand_df.at[k, t_original])
                prob += f_vars[(t_str, k)] <= D_tk, f"Max_Demand_{t_str}_{k}"
                prob += (
                    f_vars[(t_str, k)] <= s_prev + q_vars[(t_str, k)],
                    f"Max_Stock_{t_str}_{k}",
                )

        # Solve Model
        # Use a generous time limit to prevent "No Solution" errors on complex data
        solver = pulp.PULP_CBC_CMD(msg=True, gapRel=0.05, timeLimit=30)
        prob.solve(solver)

        # SAFETY CHECK
        status_str = pulp.LpStatus[prob.status]
        if prob.status != pulp.LpStatusOptimal:
            print(
                f"\nCRITICAL WARNING: Solver failed with status '{status_str}'. Results may be empty."
            )

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
                c_direct = float(cost_df.at[k, "direct_cost"])
                c_middle = float(cost_df.at[k, "middleman_cost"])
                c_holding = float(cost_df.at[k, "holding_cost"])

                # Financials
                spend_direct = val_q * c_direct
                spend_middle = val_m * c_middle
                spend_holding = val_s * c_holding
                total_item_cost = spend_direct + spend_middle + spend_holding

                results.append(
                    {
                        "Quarter": t_original,
                        "Product": k,
                        "Demand": val_d,
                        "Direct_Order": val_q,
                        "Fulfilled_Direct": val_f,
                        "Middleman_Units": val_m,
                        "End_Stock": val_s,
                        # Financial Columns
                        "Budget_Utilized_($)": spend_direct,
                        "Middleman_Cost_($)": spend_middle,
                        "Holding_Cost_($)": spend_holding,
                        "Total_Cost_($)": total_item_cost,
                    }
                )

        return {
            "status": status_str,
            "total_profit": pulp.value(prob.objective)
            if pulp.value(prob.objective)
            else 0.0,
            "results_df": pd.DataFrame(results).set_index("Quarter"),
        }
    return (optimize_inventory,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Allocation
    Solving for Table 2.
    """)
    return


@app.cell
def _(optimize_inventory, pd):
    def evaluate_budget_scenarios(demand_df, cost_df, budgets):
        """
        Runs the inventory optimization model across multiple budget levels.
        Returns a summary DataFrame comparing financial performance.
        """
        summary_data = []

        # 1. Ensure we are only running on the Top 30 products to match paper scale
        # (Assuming cost_df index represents the valid Top 30 list)
        valid_products = [p for p in demand_df.columns if p in cost_df.index]
        demand_df_filtered = demand_df[valid_products].copy()

        print(
            f"Evaluating {len(budgets)} budget scenarios for {len(valid_products)} products..."
        )

        for b in budgets:
            # Run Optimization
            # (Using the robust 'new' function from before)
            output = optimize_inventory(demand_df_filtered, cost_df, b)

            # Get Data
            df_res = output["results_df"]
            total_profit = output["total_profit"]

            # Calculate Aggregates
            budget_utilized = df_res["Budget_Utilized_($)"].sum()
            middleman_cost = df_res["Middleman_Cost_($)"].sum()
            holding_cost = df_res["Holding_Cost_($)"].sum()
            middleman_units = df_res["Middleman_Units"].sum()
            total_cost = budget_utilized + middleman_cost + holding_cost

            # Calculate ROI Metric
            profit_per_10k = total_profit / (b / 10000)

            summary_data.append(
                {
                    "Budget Scenario": b,
                    "Total Profit ($)": total_profit,
                    "Total Cost ($)": total_cost,
                    "Middleman Cost ($)": middleman_cost,
                    "Middleman Units": middleman_units,
                    "Budget Utilized ($)": budget_utilized,
                    "Profit/$10k": profit_per_10k,
                    "Status": output["status"],
                }
            )

        return pd.DataFrame(summary_data).round(2)
    return (evaluate_budget_scenarios,)


@app.cell
def _():
    forecast_steps = 1
    budgets = [10000, 20000, 30000, 40000, 50000]
    return (budgets,)


@app.cell
def _(
    budgets,
    cost_df,
    evaluate_budget_scenarios,
    generate_forecast_scenario,
    pd,
    time_series_data,
):
    prophet_forecast = generate_forecast_scenario(
        time_series_data, steps=4, method="prophet"
    )
    # --- PROPHET EVALUATION ---
    print("\n--- PROPHET RESULTS ---")
    df_prophet_results = evaluate_budget_scenarios(
        pd.concat([time_series_data.tail(18), prophet_forecast]), cost_df, budgets
    )
    df_prophet_results
    return


@app.cell
def _(
    budgets,
    cost_df,
    evaluate_budget_scenarios,
    generate_forecast_scenario,
    pd,
    time_series_data,
):
    arima_forecast = generate_forecast_scenario(
        time_series_data, steps=12, method="arima"
    )

    # --- ARIMA EVALUATION ---
    print("--- ARIMA RESULTS ---")
    df_arima_results = evaluate_budget_scenarios(
        pd.concat([time_series_data.tail(18), arima_forecast]), cost_df, budgets
    )
    df_arima_results
    return arima_forecast, df_arima_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part B
    Compare the performance using the SARIMA forecasting model.
    """)
    return


@app.cell
def _(ARIMA):
    def fit_best_sarima(series):
        """
        Grid searches for the best SARIMA model based on AIC.
        Fixed Seasonality (s=4) for Quarterly data.
        """
        best_aic = float("inf")
        best_model = None

        # Simplified Grid: (p,d,q) x (P,D,Q,s)
        orders = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ]
        seasonal_orders = [
            (0, 1, 1, 4),  # Classic seasonal moving average
            (1, 1, 0, 4),  # Seasonal autoregressive
            (0, 1, 0, 4),  # Pure seasonal differencing
        ]

        for order in orders:
            for seasonal_order in seasonal_orders:
                try:
                    # Enforce stationarity/invertibility to avoid bad models
                    model = ARIMA(
                        series,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    results = model.fit()

                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_model = results
                except:
                    continue

        # Fallback if grid search fails completely
        if best_model is None:
            return ARIMA(
                series, order=(0, 0, 0), seasonal_order=(0, 0, 0, 4)
            ).fit()

        return best_model
    return (fit_best_sarima,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Table II. Forecast Accuracy Comparison for Selected Product Combinations
    (Prophet, ARIMA, SARIMA)
    """)
    return


@app.cell
def _(
    pd,
    rolling_arima_eval,
    rolling_prophet_eval,
    rolling_sarima_eval,
    time_series_data,
    tqdm,
):
    results = []
    target_lenses = [
        "prog, trans",
        "sv, bb",
        "sv, trans, bb",
        "repair",
        "sv, poly",
        "sv, clear",
    ]

    # To match the demand from the paper, the last 18 quarters are used
    timeseries_18 = (
        time_series_data
        .tail(18)
        .copy()
    )


    lenses_to_process = [l for l in target_lenses if l in time_series_data.columns]

    # Single loop for both models
    for col in tqdm(lenses_to_process, desc="Processing Lenses"):
        series = timeseries_18[col]
        total_demand = series.sum()

        # 1. Fit ARIMA
        arima_res = rolling_arima_eval(series)

        # 2. Fit Prophet
        prophet_res = rolling_prophet_eval(series)

        # 3. Fit SARIMA (New)
        # (Ensure rolling_sarima_eval function is defined from previous step)
        sarima_res = rolling_sarima_eval(series)

        # --- Calculate Comparisons ---

        # Prophet vs ARIMA (Table I style: Negative means Prophet is better)
        diff_mae_prophet = prophet_res["MAE"] - arima_res["MAE"]
        diff_rmse_prophet = prophet_res["RMSE"] - arima_res["RMSE"]

        # SARIMA vs ARIMA (Positive means SARIMA is better/lower error)
        # We calculate how much error SARIMA *reduced* compared to ARIMA
        impv_mae_sarima = arima_res["MAE"] - sarima_res["MAE"]

        # Store Results
        results.append({
            "Combination": col,
            "Total Demand": total_demand,

            # Prophet Metrics
            "Prophet MAE": prophet_res["MAE"],
            "Prophet RMSE": prophet_res["RMSE"],

            # ARIMA Metrics
            "ARIMA MAE": arima_res["MAE"],
            "ARIMA RMSE": arima_res["RMSE"],

            # SARIMA Metrics (New)
            "SARIMA MAE": sarima_res["MAE"],
            "SARIMA RMSE": sarima_res["RMSE"],

            # Comparisons
            "Diff MAE (Prophet-ARIMA)": diff_mae_prophet,
            "Improvement (SARIMA vs ARIMA)": impv_mae_sarima
        })

    # Create final DataFrame
    comparison_df = pd.DataFrame(results).round(3)
    comparison_df = comparison_df.sort_values(by="Total Demand", ascending=False)

    # Reorder columns for a logical report view
    cols = [
        "Combination", "Total Demand", 
        "ARIMA MAE", "Prophet MAE", "SARIMA MAE", 
        "Diff MAE (Prophet-ARIMA)", "Improvement (SARIMA vs ARIMA)"
    ]
    comparison_df = comparison_df[cols]
    comparison_df
    return (timeseries_18,)


@app.cell
def _(
    arima_forecast,
    budgets,
    cost_df,
    evaluate_budget_scenarios,
    fit_best_sarima,
    generate_forecast_scenario,
    pd,
    time_series_data,
):
    sarima_forecast = generate_forecast_scenario(
        time_series_data, steps=12, method="sarima", custom_func=fit_best_sarima
    )

    # --- SARIMA EVALUATION ---
    print("--- SARIMA RESULTS ---")
    df_sarima_results = evaluate_budget_scenarios(
        pd.concat([time_series_data.tail(18), arima_forecast]), cost_df, budgets
    )
    df_sarima_results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part C
    """)
    return


@app.cell
def _(
    arima_forecast,
    cost_df,
    df_arima_results,
    np,
    optimize_inventory,
    pd,
    timeseries_18,
):
    import scipy.stats as stats
    import matplotlib.pyplot as plt

    def get_prescription_weights(sigma, mean=0, start=-6, end=6, step=0.25):
        """
        Generates normalized probabilities for lens powers based on a Normal Distribution.
        """
        # Generate discrete power steps (Diopters)
        powers = np.arange(start, end + step, step)

        # Calculate PDF (Probability Density Function)
        # We treat the discrete step as the center of a bin
        probs = stats.norm.pdf(powers, loc=mean, scale=sigma)

        # Normalize so they sum to 1 (100% of inventory)
        probs = probs / probs.sum()

        return pd.Series(probs, index=powers)

    def simulate_prescription_impact(aggregate_results, cost_df, sigma_plan, sigma_actual):
        """
        Simulates the financial impact of distributing aggregate orders into specific powers.

        aggregate_results: The DataFrame output from your LP model
        sigma_plan: The sigma used to buy inventory (Assumption)
        sigma_actual: The sigma of actual customer demand (Reality)
        """

        # 1. Get Weights
        weights_plan = get_prescription_weights(sigma_plan)
        weights_actual = get_prescription_weights(sigma_actual)

        total_real_profit = 0
        total_mismatch_cost = 0

        # 2. Iterate through every aggregate decision
        # We assume 'aggregate_results' has columns: ['Product', 'Direct_Order', 'Demand']
        for _, row in aggregate_results.iterrows():
            prod = row['Product']
            if prod not in cost_df.index: continue

            qty_ordered = row['Direct_Order']
            qty_demand_total = row['Demand']

            # Costs
            c_direct = cost_df.at[prod, 'direct_cost']
            c_middle = cost_df.at[prod, 'middleman_cost']
            c_holding = cost_df.at[prod, 'holding_cost']
            profit_margin = 0.10

            # 3. Distribute to Bins (The "Micro" Simulation)
            # Inventory is bought based on the PLAN distribution
            stock_bins = (qty_ordered * weights_plan).round(0)

            # Customers arrive based on the ACTUAL distribution
            demand_bins = (qty_demand_total * weights_actual).round(0)

            # 4. Calculate Fulfillment per Bin
            # For each power (e.g. +1.00), we sell min(demand, stock)
            sales_bins = np.minimum(stock_bins, demand_bins)

            # Unfulfilled demand goes to Middleman
            middleman_bins = demand_bins - sales_bins

            # Unsold stock incurs Holding Cost
            leftover_bins = stock_bins - sales_bins

            # 5. Calculate Financials
            revenue = (sales_bins * c_direct * profit_margin).sum()
            cost_middleman = (middleman_bins * c_middle).sum()
            cost_holding = (leftover_bins * c_holding).sum()

            # Total Profit for this product
            # Note: We paid for direct stock upfront! (qty_ordered * c_direct) is sunk cost?
            # The paper calculates profit as: Margin*Sold - MiddlemanCost - HoldingCost
            # It assumes the "Cost of Goods Sold" is covered, but we must subtract ordering cost?
            # Following paper formula: Z = (Margin * Cost * Sold) - (MiddlemanCost) - (HoldingCost)
            # Note: This formula assumes we only profit on what we sell.

            item_profit = revenue - cost_middleman - cost_holding
            total_real_profit += item_profit

        return total_real_profit

    # --- EXECUTE ANALYSIS ---

    # 1. Use your optimized results from Part A (e.g. for $30k budget)
    # Ensure you use the 'results_df' from the forecast optimization
    base_results = df_arima_results.loc[df_arima_results['Budget Scenario'] == 30000]
    # We need the DETAILED dataframe, not the summary. 
    # Re-run one optimization to get the detail if needed:
    best_model_run = optimize_inventory(pd.concat([timeseries_18, arima_forecast]), cost_df, 30000)
    detailed_plan = best_model_run['results_df']

    # 2. Define Scenarios
    # Baseline: We assumed 1.25, and reality was 1.25 (Perfect Match)
    profit_baseline = simulate_prescription_impact(detailed_plan, cost_df, sigma_plan=1.25, sigma_actual=1.25)

    # New Scenario: We assumed 1.25, but reality changed to 0.625 (Mismatch)
    # This simulates "Using the wrong distribution for the PrescriptionDistributor"
    profit_new_reality = simulate_prescription_impact(detailed_plan, cost_df, sigma_plan=1.25, sigma_actual=0.625)

    # Optimized New: We assumed 0.625, and reality was 0.625 (Optimized)
    # This simulates "If we had updated our model"
    profit_optimized = simulate_prescription_impact(detailed_plan, cost_df, sigma_plan=0.625, sigma_actual=0.625)

    # 3. Print Results
    print(f"--- PART (C) PRESCRIPTION DISTRIBUTION IMPACT ---")
    print(f"Original Profit (Aggregate Model): ${best_model_run['total_profit']:,.2f}")
    print("-" * 50)
    print(f"Scenario 1: Distribution Match (Wide/Wide)   : ${profit_baseline:,.2f}")
    print(f"Scenario 2: Distribution Mismatch (Wide/Narrow): ${profit_new_reality:,.2f}")
    print(f"Scenario 3: Updated Model (Narrow/Narrow)    : ${profit_optimized:,.2f}")
    print("-" * 50)
    print(f"Loss due to Mismatch: ${profit_baseline - profit_new_reality:,.2f}")

    # 4. Visualize the Distributions
    x = np.arange(-6, 6.25, 0.25)
    y1 = get_prescription_weights(1.25, start=-6, end=6).values
    y2 = get_prescription_weights(0.625, start=-6, end=6).values

    plt.figure(figsize=(10, 5))
    plt.plot(x, y1, label='Original (Sigma=1.25)', color='blue', linewidth=2)
    plt.plot(x, y2, label='New (Sigma=0.625)', color='red', linestyle='--', linewidth=2)
    plt.fill_between(x, y1, alpha=0.1, color='blue')
    plt.fill_between(x, y2, alpha=0.1, color='red')
    plt.title("Prescription Demand Distribution Change")
    plt.xlabel("Diopters (Power)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
