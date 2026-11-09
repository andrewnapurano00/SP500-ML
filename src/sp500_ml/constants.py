FACTORS = {'growth': ['revenue_growth_yoy', 'revenue_cagr_3y', 'eps_growth_yoy', 'net_income_growth_3y', 'ebitda_growth_yoy', 'ocf_growth_yoy', 'fcf_growth_3y'], 'profitability': ['gross_margin', 'ebitda_margin', 'operating_margin', 'net_margin', 'roe', 'roic', 'fcf_margin'], 'valuation': ['earnings_yield', 'sales_yield', 'ebitda_ev_yield', 'fcf_yield', 'book_to_market', 'ocf_yield', 'net_shareholder_yield'], 'balance_sheet': ['debt_to_equity', 'debt_to_assets', 'net_debt_to_ebitda', 'interest_coverage', 'cash_to_assets', 'current_ratio', 'ocf_to_debt']}

BASE_FACTORS = sum(FACTORS.values(), [])

EXTRAS = ['roa', 'equity_to_assets']

ALL_FACTORS = BASE_FACTORS + EXTRAS

FAMILY = {f: g for g, fs in FACTORS.items() for f in fs} | {'roa': 'profitability', 'equity_to_assets': 'balance_sheet'}
PROXIES = {'Communication Services': 'VOX', 'Consumer Discretionary': 'XLY', 'Consumer Staples': 'XLP', 'Energy': 'XLE', 'Financials': 'XLF', 'Health Care': 'XLV', 'Industrials': 'XLI', 'Information Technology': 'XLK', 'Materials': 'XLB', 'Real Estate': 'XLRE', 'Utilities': 'XLU'}

CONTEXT_FEATURES = [*[f'sector_return_{n}d' for n in [21, 63, 126, 252]], 'sector_momentum_12_1', *[f'sector_realized_vol_{n}d' for n in [21, 63]], 'sector_price_to_sma50', 'sector_price_to_sma200', 'sector_drawdown_252d', *[f'sector_excess_market_return_{n}d' for n in [21, 63, 126]], 'sector_strength_rank_63d']

RELATIVE_FEATURES = ['stock_excess_sector_return_21d', 'stock_excess_sector_return_63d', 'stock_to_sector_vol_ratio_63d']

SECTOR_FEATURES = CONTEXT_FEATURES + RELATIVE_FEATURES
