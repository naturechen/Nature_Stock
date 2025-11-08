# pipeline/extractor.py
from db.repository import StockRepository

class Extractor:
    def __init__(self, cursor):
        self.repo = StockRepository(cursor)

    def extract_all_daily_stock_data(self):
        print("🔹 Extracting data from database...")
        all_daily = self.repo.get_all_daily_data()
        all_daily[['open','high','low','close','volume']] = all_daily[['open','high','low','close','volume']].astype(float)
        print(f"✅ Extracted {len(all_daily)} records successfully")
        return all_daily
