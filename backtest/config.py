import os
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

class Config:
    # JQData account configuration (read from .env file)
    JQ_USERNAME = os.getenv('JQ_USERNAME')
    JQ_PASSWORD = os.getenv('JQ_PASSWORD')

    # directory paths
    DATA_DIR = 'data'
    RES_DIR = 'output'

    # data paths
    PATHS = {
        'price_data': os.path.join(DATA_DIR, 'hs300_price_data.csv'),
        'industry_map': os.path.join(DATA_DIR, 'industry.json'),
        'valuation_data': os.path.join(DATA_DIR, 'valuation.csv'),
        'balance_data': os.path.join(DATA_DIR, 'balance.csv'),
        'cash_flow_data': os.path.join(DATA_DIR, 'cash_flow.csv'),
        'income_data': os.path.join(DATA_DIR, 'income.csv'),
        'indicator_data': os.path.join(DATA_DIR, 'indicator.csv'),
        'factor_jsonl': os.path.join(DATA_DIR, 'samples.jsonl'),
    }

    # backtest parameters
    BACKTEST = {
        'index_code': '000300.XSHG',
        'quantiles': 10,
        'periods': (1, 5, 22),
        'industry_level': 'jq_l1'
    }

os.makedirs(Config.DATA_DIR, exist_ok=True)
os.makedirs(Config.RES_DIR, exist_ok=True)