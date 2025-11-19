import yfinance as yf
import pandas as pd
import numpy as np
import time

class YahooFinanceFetcher:
    """
    从 Yahoo Finance 抓取股票数据
    """
    def __init__(self):
        pass

    def fetch_stock_data(self, ticker:str, period:str = None, interval:str = None) -> pd.DataFrame:
        """
        ticker: 股票代码，例如 'AAPL'
        start: 开始日期 'YYYY-MM-DD'
        end: 结束日期 'YYYY-MM-DD'
        interval: 数据间隔 '1d', '1wk', '1mo', '6mo'
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            df.reset_index(inplace=True)  # 把日期变成列
            df.insert(0, 'ticker_id', ticker)  # 在第一列插入股票代码列

            # 需要的列
            required_cols = [
                'ticker_id', 'Date', 'Open', 'High', 'Low', 'Close',
                'Volume', 'Dividends', 'Stock Splits'
            ]

            # 只选择存在的列（自动过滤掉 Capital Gains）
            df = df[[col for col in required_cols if col in df.columns]]

            return df
        
        except Exception as e:
            print(f"获取 {self.ticker} 数据失败: {e}")
            return pd.DataFrame()  # 返回空 DataFrame 便于调用方处理
    

    def fetch_multiple_daily_stocks(self, tickers:list) -> pd.DataFrame:
        """
        抓取多个股票的数据并合并到一个 DataFrame
        tickers: 股票代码列表，例如 ['AAPL', 'MSFT']
        """
        all_data = pd.DataFrame()
        for ticker in tickers:
            df = self.fetch_stock_data(ticker, period='1y', interval='1d')
            all_data = pd.concat([all_data, df], ignore_index=True)

            random_delay = 1 * np.random.rand()
            time.sleep(random_delay)
        
        return all_data