# pipeline/loader.py
from db.repository import StockRepository

class Loader:
    def __init__(self, cursor):
        self.repo = StockRepository(cursor)

    def load_ticker_daily_to_db(self, data):
        
        print("🔹 Loading cleaned data into database...")
        
        self.repo.insert_daily_data(data)

        print("✅ Data loaded successfully")
