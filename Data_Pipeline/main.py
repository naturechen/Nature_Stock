# main.py
from db.connection import PostgresConnector
from pipeline.extractor import Extractor
from pipeline.transformer import Transformer
from pipeline.loader import Loader
from pipeline.fetcher import YahooFinanceFetcher
from db.repository import StockRepository

# 数据库配置
CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "Finance",
    "user": "postgres",
    "password": "Nature1992."
}

def main():

    with PostgresConnector(CONFIG) as cur:
        
        db_repo = StockRepository(cur)
        #extractor = Extractor(cur)
        transformer = Transformer()
        #loader = Loader(cur)
        

        yahoo_fetcher = YahooFinanceFetcher()
        stock_daily_data = yahoo_fetcher.fetch_stock_data('TSLA', period='1y', interval='1d')
        stock_daily_data = transformer.remove_date_time(stock_daily_data)
        print(stock_daily_data.head())

        db_repo = StockRepository(cur)
        db_repo.insert_daily_data('TSLA', stock_daily_data)

        # djj = db_repo.get_all_users()
        # print(djj)

        # # Pipeline 流程
        # raw_users = extractor.extract()
        # cleaned_users = transformer.clean_emails(raw_users)
        # loader.load(cleaned_users)

if __name__ == "__main__":
    main()
