import pandas as pd
import numpy as np
import json
import inspect
from factor_backtest import analyze_factor
import os
from datetime import datetime
from config import Config


def make_panel(df, fields):
    panel_data = {}
    for f in fields:
        if f in df.columns:
            panel_data[f] = df.pivot(index='time', columns='code', values=f)
    return panel_data

def load_data():
    data = {}
    print("Loading and integrating fundamental and market data...")
    
    # Get the data paths to load
    components = [
        ('price', pd.read_csv(Config.PATHS['price_data'])),
        ('valuation', pd.read_csv(Config.PATHS['valuation_data'])),
        ('balance', pd.read_csv(Config.PATHS['balance_data'])),
        ('cash_flow', pd.read_csv(Config.PATHS['cash_flow_data'])),
        ('income', pd.read_csv(Config.PATHS['income_data'])),
        ('indicator', pd.read_csv(Config.PATHS['indicator_data']))
    ]

    # Merge data and convert to panel structure
    for name, df in components:
        try:
            if name == 'price':
                data.update(make_panel(df, ['open', 'close', 'high', 'low', 'volume', 'money', 'pre_close', 'paused']))
            else:
                # Remove data columns other than code and time
                fields = [col for col in df.columns if col not in ['code', 'time']]
                data.update(make_panel(df, fields))
        except FileNotFoundError:
            print(f"{name} data not found: {Config.PATHS[f'{name}_data']}")

    return data


def run_pipeline(target_factor_id=None):
    '''
    target_factor_id: Specify the factor ID for backtesting, e.g., 'factor_5'. If None, backtest all factors.
    '''
    # Result save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_path = os.path.join(Config.RES_DIR, f'factor_results_{timestamp}.csv')
    error_log_path = os.path.join(Config.RES_DIR, f'error_log_{timestamp}.txt') 

    print("Loading data and building Panel...")
    data = load_data()
    print(f"Data loading complete. Number of available fields: {len(data.keys())}")
    
    # 2. Read jsonl file and execute functions
    jsonl_file_path = Config.PATHS['factor_jsonl']
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            factor_info = [json.loads(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Cannot find {jsonl_file_path}")
        return

    # Factor backtesting
    for i, info in enumerate(factor_info):
        factor_id = f"factor_{i}"

        # If a specific factor ID is specified, skip unmatched factors
        if target_factor_id is not None and factor_id != target_factor_id:
            continue

        code_str = info['factor_python']

        local_env = {}
        try:
            exec(code_str, globals(), local_env)
            func_name = [k for k, v in local_env.items() if inspect.isfunction(v)][0]
            factor_func = local_env[func_name]
            
            sig = inspect.signature(factor_func)
            required_args = list(sig.parameters.keys())
            
            func_inputs = {k: data[k] for k in required_args if k in data}
            missing = set(required_args) - set(func_inputs.keys())
            if missing:
                print(f"[{factor_id}] Missing fields {missing}, skipping")
                continue
                
            print(f"[{factor_id}] Calculating factor value...")
            factor_value = factor_func(**func_inputs)
            
            # Analyze the factor
            print(f"[{factor_id}] Running analyze_factor...")
            far = analyze_factor(
                factor_value,
                quantiles=Config.BACKTEST['quantiles'],
                periods=Config.BACKTEST['periods'],
                industry=Config.BACKTEST['industry_level']
            )
            
            # Print daily IC
            print(f"[{factor_id}] Daily IC value: {far.ic_summary.loc[1, 'IC_Mean']}") # type: ignore
            
            # If testing a single factor, plot quantile and cumulative returns
            if target_factor_id is not None:
                far.plot_quantile_returns(period=1, save_path=os.path.join(Config.RES_DIR, f'{factor_id}_quantile_returns.png'))
                far.plot_cumulative_returns(period=1, save_path=os.path.join(Config.RES_DIR, f'{factor_id}_cumulative_returns.png'))
            
            # Save all backtest results (12 indicators for 1, 5, 22 periods)
            res_row = {'Factor_ID': factor_id,
                    'Factor_Formula': info.get('factor_formula', '')  # Factor expression
            }

            for p in [1, 5, 22]:
                if p in far.ic_summary.index: # type: ignore
                    res_row[f'IC_{p}'] = far.ic_summary.loc[p, 'IC_Mean'] # type: ignore
                    res_row[f'IR_{p}'] = far.ic_summary.loc[p, 'IR'] # type: ignore
                    res_row[f'Rank_IC_{p}'] = far.rank_ic_summary.loc[p, 'IC_Mean'] # type: ignore
                    res_row[f'Rank_IR_{p}'] = far.rank_ic_summary.loc[p, 'IR'] # type: ignore
            
            res_df = pd.DataFrame([res_row])
            
            # Append to CSV in real-time, write header if creating for the first time
            file_exists = os.path.exists(res_path)
            res_df.to_csv(res_path, mode='a', index=False, header=not file_exists)

        except Exception as e:
            error_msg = f"[{factor_id}] Execution error: {e}"
            print(error_msg)
            # Write error message to log file
            with open(error_log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(error_msg + '\n')

        # If testing a single factor, break the loop after execution, stop iterating remaining factors
        if target_factor_id is not None and factor_id == target_factor_id:
            break

if __name__ == '__main__':
    # Test all factors
    run_pipeline()

    # Test a single factor
    # run_pipeline(target_factor_id='factor_43')