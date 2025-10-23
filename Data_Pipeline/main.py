# main.py
from db.connection import PostgresConnector
from pipeline.extractor import Extractor
from pipeline.loader import Loader
from pipeline.fetcher import YahooFinanceFetcher
from pipeline.transformer import Transformer


# 数据库配置
CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "Finance",
    "user": "postgres",
    "password": "Nature1992."
}

def stock_list():

    stock_lists = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'OKLO', 'NVDA', 'META', 'RKLB', 'CRWV']
    return stock_lists
    

def main():

    with PostgresConnector(CONFIG) as cur:
        
        yahoo_fetcher = YahooFinanceFetcher()
        stock_daily_data = yahoo_fetcher.fetch_multiple_daily_stocks(stock_list())

        transformer = Transformer()
        stock_daily_data = transformer.remove_date_time(stock_daily_data)
        print(stock_daily_data.head())

        loader = Loader(cur)
        loader.load_ticker_daily_to_db(stock_daily_data)

        # # Pipeline example
        # raw_users = extractor.extract()
        # cleaned_users = transformer.clean_emails(raw_users)
        # loader.load(cleaned_users)

if __name__ == "__main__":
    main()
