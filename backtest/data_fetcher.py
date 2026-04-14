import os
import json
import pandas as pd
from jqdatasdk import *
from config import Config

class DataFetcher:
    def __init__(self):
        auth(Config.JQ_USERNAME, Config.JQ_PASSWORD)
        info = get_account_info()
        self.start_date = info['date_range_start'].split()[0]
        self.end_date = info['date_range_end'].split()[0]
        print(f"Data date range: {self.start_date} to {self.end_date}")
        
        self.stocks = get_index_stocks(Config.BACKTEST['index_code'], date=self.end_date)
        print(f"Fetched constituent stocks, total: {len(self.stocks)}")

    def fetch_industry(self):
        print("Fetching industry information...")
        industry_map = get_industry(self.stocks, date=self.end_date)
        with open(Config.PATHS['industry_map'], 'w', encoding='utf-8') as f:
            json.dump(industry_map, f, ensure_ascii=False, indent=4)
        print("Industry data saved.")

    def fetch_price(self):
        print("Fetching market data...")
        df_price = get_price(
            security=self.stocks, start_date=self.start_date, end_date=self.end_date,
            frequency='daily', fields=['open', 'close', 'high', 'low', 'volume', 'money', 'pre_close', 'paused'],
            skip_paused=True, panel=False, fill_paused=True, fq='pre'
        )
        df_price.to_csv(Config.PATHS['price_data'], index=False)
        print("Market data saved.")

    def _fetch_fundamentals_data(self, query_obj, filter_col):
        trade_days = get_trade_days(start_date=self.start_date, end_date=self.end_date)
        count_days = len(trade_days) + 10 
        
        all_data = []
        for stock in self.stocks:
            q = query_obj.filter(filter_col == stock)
            df_stock = get_fundamentals_continuously(q, end_date=self.end_date, count=count_days, panel=False)
            if not df_stock.empty and 'code' not in df_stock.columns:
                df_stock['code'] = stock
            all_data.append(df_stock)
            
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def fetch_fundamentals(self, category_name, query_obj, filter_col):
        print(f"Fetching {category_name} fundamentals data...")
        df = self._fetch_fundamentals_data(query_obj, filter_col)
        df.rename(columns={'day': 'time'}, inplace=True)
        save_path = Config.PATHS[f'{category_name}_data']
        df.to_csv(save_path, index=False)
        print(f"{category_name} fundamentals data saved to: {save_path}")

    def run_all(self):
        self.fetch_industry()
        self.fetch_price()
        
        # Sequentially fetch various fundamental data
        q_valuation = query(
            valuation.capitalization, valuation.circulating_cap, valuation.market_cap, 
            valuation.circulating_market_cap, valuation.turnover_ratio, valuation.pe_ratio, 
            valuation.pe_ratio_lyr, valuation.pb_ratio, valuation.ps_ratio, valuation.pcf_ratio
        )
        q_balance = query(
            balance.cash_equivalents, balance.settlement_provi, balance.lend_capital, balance.trading_assets, 
            balance.bill_receivable, balance.account_receivable, balance.advance_payment, balance.insurance_receivables, 
            balance.reinsurance_receivables, balance.reinsurance_contract_reserves_receivable, balance.interest_receivable, 
            balance.dividend_receivable, balance.other_receivable, balance.bought_sellback_assets, balance.inventories, 
            balance.non_current_asset_in_one_year, balance.other_current_assets, balance.total_current_assets, 
            balance.loan_and_advance, balance.hold_for_sale_assets, balance.hold_to_maturity_investments, 
            balance.longterm_receivable_account, balance.longterm_equity_invest, balance.investment_property, 
            balance.fixed_assets, balance.constru_in_process, balance.construction_materials, balance.fixed_assets_liquidation, 
            balance.biological_assets, balance.oil_gas_assets, balance.intangible_assets, balance.development_expenditure, 
            balance.good_will, balance.long_deferred_expense, balance.deferred_tax_assets, balance.other_non_current_assets, 
            balance.total_non_current_assets, balance.total_assets, balance.shortterm_loan, balance.borrowing_from_centralbank, 
            balance.deposit_in_interbank, balance.borrowing_capital, balance.trading_liability, balance.notes_payable, 
            balance.accounts_payable, balance.advance_peceipts, balance.sold_buyback_secu_proceeds, balance.commission_payable, 
            balance.salaries_payable, balance.taxs_payable, balance.interest_payable, balance.dividend_payable, 
            balance.other_payable, balance.reinsurance_payables, balance.insurance_contract_reserves, balance.proxy_secu_proceeds, 
            balance.receivings_from_vicariously_sold_securities, balance.non_current_liability_in_one_year, 
            balance.other_current_liability, balance.total_current_liability, balance.longterm_loan, balance.bonds_payable, 
            balance.longterm_account_payable, balance.specific_account_payable, balance.estimate_liability, 
            balance.deferred_tax_liability, balance.other_non_current_liability, balance.total_non_current_liability, 
            balance.total_liability, balance.paidin_capital, balance.capital_reserve_fund, balance.treasury_stock, 
            balance.specific_reserves, balance.surplus_reserve_fund, balance.ordinary_risk_reserve_fund, balance.retained_profit, 
            balance.foreign_currency_report_conv_diff, balance.equities_parent_company_owners, balance.minority_interests, 
            balance.total_owner_equities, balance.total_sheet_owner_equities
        )
        q_cash_flow = query(
            cash_flow.goods_sale_and_service_render_cash, cash_flow.net_deposit_increase, cash_flow.net_borrowing_from_central_bank, 
            cash_flow.net_borrowing_from_finance_co, cash_flow.net_original_insurance_cash, cash_flow.net_cash_received_from_reinsurance_business, 
            cash_flow.net_insurer_deposit_investment, cash_flow.net_deal_trading_assets, cash_flow.interest_and_commission_cashin, 
            cash_flow.net_increase_in_placements, cash_flow.net_buyback, cash_flow.tax_levy_refund, cash_flow.other_cashin_related_operate, 
            cash_flow.subtotal_operate_cash_inflow, cash_flow.goods_and_services_cash_paid, cash_flow.net_loan_and_advance_increase, 
            cash_flow.net_deposit_in_cb_and_ib, cash_flow.original_compensation_paid, cash_flow.handling_charges_and_commission, 
            cash_flow.policy_dividend_cash_paid, cash_flow.staff_behalf_paid, cash_flow.tax_payments, cash_flow.other_operate_cash_paid, 
            cash_flow.subtotal_operate_cash_outflow, cash_flow.net_operate_cash_flow, cash_flow.invest_withdrawal_cash, 
            cash_flow.invest_proceeds, cash_flow.fix_intan_other_asset_dispo_cash, cash_flow.net_cash_deal_subcompany, 
            cash_flow.other_cash_from_invest_act, cash_flow.subtotal_invest_cash_inflow, cash_flow.fix_intan_other_asset_acqui_cash, 
            cash_flow.invest_cash_paid, cash_flow.impawned_loan_net_increase, cash_flow.net_cash_from_sub_company, 
            cash_flow.other_cash_to_invest_act, cash_flow.subtotal_invest_cash_outflow, cash_flow.net_invest_cash_flow, 
            cash_flow.cash_from_invest, cash_flow.cash_from_mino_s_invest_sub, cash_flow.cash_from_borrowing, 
            cash_flow.cash_from_bonds_issue, cash_flow.other_finance_act_cash, cash_flow.subtotal_finance_cash_inflow, 
            cash_flow.borrowing_repayment, cash_flow.dividend_interest_payment, cash_flow.proceeds_from_sub_to_mino_s, 
            cash_flow.other_finance_act_payment, cash_flow.subtotal_finance_cash_outflow, cash_flow.net_finance_cash_flow, 
            cash_flow.exchange_rate_change_effect, cash_flow.cash_equivalent_increase, cash_flow.cash_equivalents_at_beginning, 
            cash_flow.cash_and_equivalents_at_end
        )
        q_income = query(
            income.total_operating_revenue, income.operating_revenue, income.interest_income, income.premiums_earned, 
            income.commission_income, income.total_operating_cost, income.operating_cost, income.interest_expense, 
            income.commission_expense, income.refunded_premiums, income.net_pay_insurance_claims, 
            income.withdraw_insurance_contract_reserve, income.policy_dividend_payout, income.reinsurance_cost, 
            income.operating_tax_surcharges, income.sale_expense, income.administration_expense, income.financial_expense, 
            income.asset_impairment_loss, income.fair_value_variable_income, income.investment_income, 
            income.invest_income_associates, income.exchange_income, income.operating_profit, income.non_operating_revenue, 
            income.non_operating_expense, income.disposal_loss_non_current_liability, income.total_profit, 
            income.income_tax_expense, income.net_profit, income.np_parent_company_owners, income.minority_profit, 
            income.basic_eps, income.diluted_eps, income.other_composite_income, income.total_composite_income, 
            income.ci_parent_company_owners, income.ci_minority_owners
        )
        q_indicator = query(
            indicator.eps, indicator.adjusted_profit, indicator.operating_profit, indicator.value_change_profit, 
            indicator.roe, indicator.inc_return, indicator.roa, indicator.net_profit_margin, indicator.gross_profit_margin, 
            indicator.expense_to_total_revenue, indicator.operation_profit_to_total_revenue, indicator.net_profit_to_total_revenue, 
            indicator.operating_expense_to_total_revenue, indicator.ga_expense_to_total_revenue, 
            indicator.financing_expense_to_total_revenue, indicator.operating_profit_to_profit, 
            indicator.invesment_profit_to_profit, indicator.adjusted_profit_to_profit, indicator.goods_sale_and_service_to_revenue, 
            indicator.ocf_to_revenue, indicator.ocf_to_operating_profit, indicator.inc_total_revenue_year_on_year, 
            indicator.inc_total_revenue_annual, indicator.inc_revenue_year_on_year, indicator.inc_revenue_annual, 
            indicator.inc_operation_profit_year_on_year, indicator.inc_operation_profit_annual, indicator.inc_net_profit_year_on_year, 
            indicator.inc_net_profit_annual, indicator.inc_net_profit_to_shareholders_year_on_year, 
            indicator.inc_net_profit_to_shareholders_annual
        )
        self.fetch_fundamentals('valuation', q_valuation, valuation.code)
        self.fetch_fundamentals('balance', q_balance, balance.code)
        self.fetch_fundamentals('cash_flow', q_cash_flow, cash_flow.code)
        self.fetch_fundamentals('income', q_income, income.code)
        self.fetch_fundamentals('indicator', q_indicator, indicator.code)

if __name__ == "__main__":
    fetcher = DataFetcher()
    fetcher.run_all()