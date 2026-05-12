import os
import sys

# ensure current directory is in sys.path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_backtest import run_pipeline

if __name__ == '__main__':
    print("=== Start Backtesting ===") 
    # 从指定的因子开始断点续跑
    # run_pipeline(start_factor_id='factor_306')

    # test all factors 
    run_pipeline()

    # test a single factor, specify the target_factor_id (e.g., 'factor_43')
    # run_pipeline(target_factor_id='factor_37')
    
    print("=== Complete Backtesting! ===")   