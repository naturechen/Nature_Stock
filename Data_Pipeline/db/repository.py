# db/repository.py
import io

class StockRepository:
    def __init__(self, cursor):
        self.cur = cursor

    def get_all_users(self):
        self.cur.execute("SELECT * FROM stock.ticker_daily")
        return self.cur.fetchall()

    def insert_daily_data(self, data):
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
            self.cur.execute("""
                CREATE TEMP TABLE tmp_ticker_daily (
                    ticker_id TEXT,
                    date DATE,
                    open FLOAT,
                    high FLOAT,
                    low FLOAT,
                    close FLOAT,
                    volume BIGINT,
                    dividends FLOAT,
                    stock_splits FLOAT
                ) ON COMMIT DROP;
            """)

            # COPY 到临时表
            self.cur.copy_from(buffer, 'tmp_ticker_daily')

            # 合并更新（防止重复主键）
            self.cur.execute("""
                INSERT INTO ticker_daily
                SELECT * FROM tmp_ticker_daily
                ON CONFLICT (ticker_id, date) DO UPDATE
                SET open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    dividends = EXCLUDED.dividends,
                    stock_splits = EXCLUDED.stock_splits;
            """)
            
            print(f"✅ Inserted daily data successfully")
        except Exception as e:
            print(f"❌ Failed to insert daily data for: {e}")
            return
        

    def delete_user(self, user_id):
        self.cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
