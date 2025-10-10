import yfinance as yf
import pandas as pd

class YahooFinanceFetcher:
    """
    从 Yahoo Finance 抓取股票数据
    """
    def __init__(self):
        pass

    def fetch_stock_data(self, ticker, start=None, end=None, interval="1d"):
        """
        ticker: 股票代码，例如 'AAPL'
        start: 开始日期 'YYYY-MM-DD'
        end: 结束日期 'YYYY-MM-DD'
        interval: 数据间隔 '1d', '1wk', '1mo'
        """
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, interval=interval)
        df.reset_index(inplace=True)  # 把日期变成列
        return df
    
    
