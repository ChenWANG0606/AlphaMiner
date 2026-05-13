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
    def __init__(self, ic_df, rank_ic_df, quantile_rets_dict, topn_rets_dict):
        self.ic = ic_df
        self.rank_ic = rank_ic_df
        self.ic_summary = self._calc_summary(ic_df)
        self.rank_ic_summary = self._calc_summary(rank_ic_df)
        self.quantile_rets = quantile_rets_dict
        self.topn_rets = topn_rets_dict

    def _resolve_direction(self, period, direction='auto'):
        direction = (direction or 'auto').lower()
        if direction in ['positive', 'negative']:
            return direction

        if period in self.ic.columns:
            ic_mean = self.ic[period].mean()
            if pd.notna(ic_mean):
                return 'positive' if ic_mean >= 0 else 'negative'
        return 'positive'
        
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

    def plot_topn_nav(self, period, save_path=None, top_n=50, direction='auto'):
        if period not in self.topn_rets:
            print(f"Top-N return data for period {period} not found")
            return

        df = self.topn_rets[period].copy()
        if df.empty:
            print(f"Top-N return data for period {period} is empty")
            return

        df = df.sort_index()
        df.index = pd.to_datetime(df.index)
        current_direction = self._resolve_direction(period, direction)
        if current_direction == 'positive':
            strategy_ret = df['top_ret']
            strategy_name = f'Top {top_n} Long NAV'
        else:
            strategy_ret = df['bottom_ret']
            strategy_name = f'Bottom {top_n} Long NAV'

        nav_df = pd.DataFrame(index=df.index)
        nav_df['strategy_nav'] = (1 + strategy_ret.fillna(0)).cumprod()
        nav_df['benchmark_nav'] = (1 + df['benchmark_ret'].fillna(0)).cumprod()
        nav_df['excess_nav'] = nav_df['strategy_nav'] / nav_df['benchmark_nav']

        plt.figure(figsize=(11, 6))
        plt.plot(nav_df.index, nav_df['strategy_nav'], label=strategy_name)
        plt.plot(nav_df.index, nav_df['benchmark_nav'], label='HS300 Universe Equal-weight NAV')
        plt.plot(nav_df.index, nav_df['excess_nav'], label='Excess NAV (Strategy / Benchmark)')
        plt.title(
            f'Period {period} NAV Curve (Start = 1, Current Direction: {current_direction.capitalize()})'
        )
        plt.xlabel('Time')
        plt.ylabel('NAV')
        plt.axhline(1, color='black', linewidth=1, linestyle='--')
        plt.legend()
        plt.gcf().autofmt_xdate()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_long_short_cumulative(self, period, save_path=None, direction='auto'):
        if period not in self.quantile_rets:
            print(f"Quantile data for period {period} not found")
            return

        df_rets = self.quantile_rets[period].copy()
        if df_rets.empty or df_rets.shape[1] < 2:
            print(f"Not enough quantile groups for long-short plot in period {period}")
            return

        cols_sorted = sorted(df_rets.columns)
        low_q = cols_sorted[0]
        high_q = cols_sorted[-1]
        current_direction = self._resolve_direction(period, direction)
        if current_direction == 'positive':
            long_short = df_rets[high_q] - df_rets[low_q]
            ls_label = f'Long Q{high_q} - Short Q{low_q}'
        else:
            long_short = df_rets[low_q] - df_rets[high_q]
            ls_label = f'Long Q{low_q} - Short Q{high_q}'

        long_short.index = pd.to_datetime(long_short.index)
        long_short_cum = (1 + long_short.fillna(0)).cumprod()

        plt.figure(figsize=(10, 5))
        plt.plot(long_short_cum.index, long_short_cum, label=ls_label)
        plt.title(
            f'Period {period} Long-Short Cumulative Returns (Current Direction: {current_direction.capitalize()})'
        )
        plt.xlabel('Time')
        plt.ylabel('Cumulative Return')
        plt.axhline(1, color='black', linewidth=1, linestyle='--')
        plt.legend()
        plt.gcf().autofmt_xdate()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_ic_timeseries(self, period, save_path=None, rolling_window=22):
        if period not in self.ic.columns or period not in self.rank_ic.columns:
            print(f"IC data for period {period} not found")
            return

        ic_ts = self.ic[period].copy()
        rank_ic_ts = self.rank_ic[period].copy()
        ic_ts.index = pd.to_datetime(ic_ts.index)
        rank_ic_ts.index = pd.to_datetime(rank_ic_ts.index)

        plt.figure(figsize=(11, 5))
        plt.plot(ic_ts.index, ic_ts, alpha=0.4, label='IC')
        plt.plot(rank_ic_ts.index, rank_ic_ts, alpha=0.4, label='Rank IC')
        plt.plot(ic_ts.rolling(rolling_window).mean().index,
                 ic_ts.rolling(rolling_window).mean(),
                 linewidth=2, label=f'IC {rolling_window}D MA')
        plt.axhline(0, color='black', linewidth=1, linestyle='--')
        plt.title(f'Period {period} IC / Rank IC Time Series')
        plt.xlabel('Time')
        plt.ylabel('IC')
        plt.legend()
        plt.gcf().autofmt_xdate()

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

def analyze_factor(
    factor,
    quantiles=Config.BACKTEST['quantiles'],
    periods=Config.BACKTEST['periods'],
    industry=Config.BACKTEST['industry_level'],
    top_n=Config.BACKTEST.get('top_n', 50)
):
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
    topn_rets_dict = {}
    
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
        topn_rows = []
        for t in processed_factor.index:
            if t in fwd_ret.index and not fwd_ret.loc[t].isna().all():
                # Safe grouped average calculation
                valid_mask = ~q_groups.loc[t].isna() & ~fwd_ret.loc[t].isna()
                if valid_mask.any():
                    grp = fwd_ret.loc[t][valid_mask].groupby(q_groups.loc[t][valid_mask]).mean()
                    grp.name = t
                    q_rets.append(grp)

                # Top-N long return and universe equal-weight benchmark return
                valid_factor_mask = ~processed_factor.loc[t].isna() & ~fwd_ret.loc[t].isna()
                if valid_factor_mask.any():
                    f_t = processed_factor.loc[t][valid_factor_mask]
                    r_t = fwd_ret.loc[t][valid_factor_mask]
                    if not f_t.empty:
                        top_codes = f_t.nlargest(min(top_n, len(f_t))).index
                        bottom_codes = f_t.nsmallest(min(top_n, len(f_t))).index
                        top_ret = r_t.loc[top_codes].mean()
                        bottom_ret = r_t.loc[bottom_codes].mean()
                        benchmark_ret = r_t.mean()
                        topn_rows.append({
                            'time': t,
                            'top_ret': top_ret,
                            'bottom_ret': bottom_ret,
                            'benchmark_ret': benchmark_ret
                        })
                
        if q_rets:
            q_rets_df = pd.concat(q_rets, axis=1).T
            quantile_rets_dict[period] = q_rets_df
        if topn_rows:
            topn_df = pd.DataFrame(topn_rows).set_index('time')
            topn_rets_dict[period] = topn_df
            
    # Organize results
    ic_df = pd.DataFrame(ic_dict)
    rank_ic_df = pd.DataFrame(rank_ic_dict)
    
    return FactorAnalysisResult(ic_df, rank_ic_df, quantile_rets_dict, topn_rets_dict)