# db/repository.py
class UserRepository:
    def __init__(self, cursor):
        self.cur = cursor

    def get_all_users(self):
        self.cur.execute("SELECT * FROM stock.ticker_daily")
        return self.cur.fetchall()

    def insert_daily_stock_data(self, name, email):
        self.cur.execute(
            "INSERT INTO users (name, email) VALUES (%s, %s)", (name, email)
        )

    def delete_user(self, user_id):
        self.cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
