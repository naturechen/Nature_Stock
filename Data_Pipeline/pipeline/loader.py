# pipeline/loader.py
from db.repository import UserRepository

class Loader:
    def __init__(self, cursor):
        self.repo = UserRepository(cursor)

    def load(self, users):
        print("🔹 Loading cleaned data into database...")
        for _, name, email in users:
            self.repo.insert_user(name, email)
        print("✅ Data loaded successfully")
