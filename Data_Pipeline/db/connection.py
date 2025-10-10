# db/connection.py
import psycopg2

class PostgresConnector:
    def __init__(self, config):
        self.config = config

    def __enter__(self):
        self.conn = psycopg2.connect(**self.config)
        self.cur = self.conn.cursor()
        return self.cur

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.commit()
        self.cur.close()
        self.conn.close()
