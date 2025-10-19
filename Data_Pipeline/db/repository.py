# db/repository.py
class StockRepository:
    def __init__(self, cursor):
        self.cur = cursor

    def get_all_users(self):
        self.cur.execute("SELECT * FROM stock.ticker_daily")
        return self.cur.fetchall()

    def insert_daily_data(self, ticker, data):
        """
        将抓取的每日股票数据插入数据库
        cursor: 数据库游标
        ticker: 股票代码
        data: 包含每日数据的 DataFrame
        """
        insert_query = """
        INSERT INTO daily_stock_data (ticker, date, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, date) DO NOTHING;
        """
        for _, row in data.iterrows():
            self.cur.execute(insert_query, (
                ticker,
                row['Date'],
                row['Open'],
                row['High'],
                row['Low'],
                row['Close'],
                row['Volume']
            ))
        print(f"✅ Inserted daily data for {ticker} successfully")


    def delete_user(self, user_id):
        self.cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
