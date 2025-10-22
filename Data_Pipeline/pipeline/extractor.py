# pipeline/extractor.py
from db.repository import StockRepository

class Extractor:
    def __init__(self, cursor):
        self.repo = StockRepository(cursor)

    def extract(self):
        print("🔹 Extracting data from database...")
        users = self.repo.get_all_users()
        print(f"✅ Extracted {len(users)} users")
        return users
