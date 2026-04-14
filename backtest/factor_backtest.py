import pandas as pd
import numpy as np
import json
import scipy.stats as st
import matplotlib.pyplot as plt
import warnings
from config import Config

# Filter out common runtime warnings
warnings.filterwarnings('ignore')

class FactorAnalysisResult:
    def __init__(self, ic_df, rank_ic_df, quantile_rets_dict):
        self.ic = ic_df
        self.rank_ic = rank_ic_df
        self.ic_summary = self._calc_summary(ic_df)
        self.rank_ic_summary = self._calc_summary(rank_ic_df)
        self.quantile_rets = quantile_rets_dict
        
    def _calc_summary(self, df):
        # Calculate IR = mean / std
        summary = pd.DataFrame({
            'IC_Mean': df.mean(),
            'IC_Std': df.std(),
            'IR': df.mean() / df.std()
        })
        return summary

    def plot_quantile_returns(self, period, save_path=None):
        if period not in self.quantile_rets:
            print(f"Quantile data for period {period} not found")
            return
        
        df_rets = self.quantile_rets[period]
        mean_rets = df_rets.mean()  
        
        plt.figure(figsize=(10, 4))
        mean_rets.plot.bar()
        plt.title(f'Period {period} Average Returns by Quantile')
        plt.xlabel('Quantile')
        plt.ylabel('Average Return')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
    def plot_cumulative_returns(self, period, save_path=None):
        if period not in self.quantile_rets:
            return
        
        df_rets = self.quantile_rets[period]
        cum_rets = (1 + df_rets.fillna(0)).cumprod()
        cum_rets.index = pd.to_datetime(cum_rets.index)

        plt.figure(figsize=(10, 6))
        for col in cum_rets.columns:
            plt.plot(cum_rets.index, cum_rets[col], label=f'Q{col}')
        plt.title(f'Period {period} Cumulative Returns by Quantile')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.gcf().autofmt_xdate()  # Auto-rotate and optimize format

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def get_industry_mapping(industry_path=Config.PATHS['industry_map'], industry_level=Config.BACKTEST['industry_level']):
    try:
        with open(industry_path, 'r', encoding='utf-8') as f:
            ind_dict = json.load(f)
        
        mapping = {}
        for code, info in ind_dict.items():
            if industry_level in info:
                mapping[code] = info[industry_level]['industry_code']
            else:
                mapping[code] = np.nan  # Mark as NaN if no industry classification
        return pd.Series(mapping)
    except Exception as e:
        print(f"Failed to read industry data: {e}")
        return pd.Series(dtype=str)

def preprocess_cross_section(ds, ind_series):
    """Single cross-section extreme value removal and industry standardization"""
    # Replace possible infinite values with NaN first
    ds = ds.replace([np.inf, -np.inf], np.nan)
    df = pd.DataFrame({'factor': ds, 'industry': ind_series})
    df = df.dropna(subset=['factor'])
    if df.empty:
        return ds

    # Remove extreme values (3-sigma) and standardize within industry
    def process_group(g):
        factor = g['factor']
        if factor.empty:
            return factor
        mean = factor.mean()
        std = factor.std()
        if std == 0 or pd.isna(std):
            return pd.Series(np.nan, index=factor.index)
        factor = factor.clip(lower=mean - 3*std, upper=mean + 3*std)
        return (factor - factor.mean()) / factor.std()

    if not df['industry'].isna().all():
        res = df.groupby('industry', group_keys=False).apply(process_group)
    else:
        # Global standardization if no industry classification
        res = process_group(df)
        
    return res

def analyze_factor(factor, quantiles=Config.BACKTEST['quantiles'], periods=Config.BACKTEST['periods'], industry=Config.BACKTEST['industry_level']):
    print("Factor value industry standardization and extreme value removal...")

    # Type conversion: if the input is a Series with a multi-index (time, code), convert to DataFrame first
    if isinstance(factor, pd.Series):
        if isinstance(factor.index, pd.MultiIndex):
            factor = factor.unstack(level=-1)
        else:
            factor = factor.to_frame()

    factor = factor.replace([np.inf, -np.inf], np.nan)
    # Clean up rows (time) or columns (stocks) that are all NaN
    factor = factor.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    ind_series = get_industry_mapping(industry_path=Config.PATHS['industry_map'], industry_level=industry)
    
    # Cross-sectionally process factors
    processed_factor = factor.apply(lambda x: preprocess_cross_section(x, ind_series), axis=1) # type: ignore
    
    print("Reading market data and calculating returns...")
    try:
        price_df = pd.read_csv(Config.PATHS['price_data'])
        close_panel = price_df.pivot(index='time', columns='code', values='close')
    except Exception as e:
        print(f"Error fetching close price panel: {e}")
        return None

    # Align factor and close price formats
    common_idx = processed_factor.index.intersection(close_panel.index)
    close_panel = close_panel.loc[common_idx]
    processed_factor = processed_factor.loc[common_idx]
    
    ic_dict = {}
    rank_ic_dict = {}
    quantile_rets_dict = {}
    
    print("Start calculating indicators...")
    for period in periods:
        # Calculate forward returns: close price after 'period' days / today's close price - 1
        fwd_ret = close_panel.shift(-period) / close_panel - 1
        
        # Calculate IC (Pearson) and Rank IC (Spearman)
        ic_ts = processed_factor.corrwith(fwd_ret, axis=1, method='pearson', drop=True)
        rank_ic_ts = processed_factor.corrwith(fwd_ret, axis=1, method='spearman', drop=True)
        ic_dict[period] = ic_ts
        rank_ic_dict[period] = rank_ic_ts
        
        # Calculate quantile returns
        # Cross-sectionally stratify the factor into quantiles
        def get_quantiles(s):
            try:
                s_clean = s.dropna()
                if s_clean.empty:
                    return pd.Series(index=s.index, dtype='float')
                q = pd.qcut(s_clean, quantiles, labels=False, duplicates='drop') + 1
                return q.reindex(s.index)
            except:
                return pd.Series(index=s.index, dtype='float')
            
        q_groups = processed_factor.apply(get_quantiles, axis=1)
        
        # Calculate the average future return corresponding to each quantile for each period
        q_rets = []
        for t in processed_factor.index:
            if t in fwd_ret.index and not fwd_ret.loc[t].isna().all():
                # Safe grouped average calculation
                valid_mask = ~q_groups.loc[t].isna() & ~fwd_ret.loc[t].isna()
                if valid_mask.any():
                    grp = fwd_ret.loc[t][valid_mask].groupby(q_groups.loc[t][valid_mask]).mean()
                    grp.name = t
                    q_rets.append(grp)
                
        if q_rets:
            q_rets_df = pd.concat(q_rets, axis=1).T
            quantile_rets_dict[period] = q_rets_df
            
    # Organize results
    ic_df = pd.DataFrame(ic_dict)
    rank_ic_df = pd.DataFrame(rank_ic_dict)
    
    return FactorAnalysisResult(ic_df, rank_ic_df, quantile_rets_dict)