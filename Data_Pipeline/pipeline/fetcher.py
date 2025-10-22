import yfinance as yf
import pandas as pd

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

            return df
        
        except Exception as e:
            print(f"获取 {self.ticker} 数据失败: {e}")
            return pd.DataFrame()  # 返回空 DataFrame 便于调用方处理
    