# main.py
from db.connection import PostgresConnector
from pipeline.extractor import Extractor
from pipeline.transformer import Transformer
from pipeline.loader import Loader

# 数据库配置
CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "demo_db",
    "user": "postgres",
    "password": "123456"
}

def main():
    with PostgresConnector(CONFIG) as cur:
        extractor = Extractor(cur)
        transformer = Transformer()
        loader = Loader(cur)

        # Pipeline 流程
        raw_users = extractor.extract()
        cleaned_users = transformer.clean_emails(raw_users)
        loader.load(cleaned_users)

if __name__ == "__main__":
    main()
