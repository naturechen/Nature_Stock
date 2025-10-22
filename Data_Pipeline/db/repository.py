# db/repository.py
import io

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
        buffer = io.StringIO()
        data.to_csv(buffer, sep='\t', header=False, index=False, na_rep='\\N')
        buffer.seek(0)
        
        try:
            self.cur.execute("SET search_path = stock") # 设置 schema,否则找不到表
            self.cur.copy_from(buffer, 'ticker_daily',
                columns=('ticker_id', 'date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits'))
            
            print(f"✅ Inserted daily data for {ticker} successfully")
        except Exception as e:
            print(f"❌ Failed to insert daily data for {ticker}: {e}")
            return
        

    def delete_user(self, user_id):
        self.cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
