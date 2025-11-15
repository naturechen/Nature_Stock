# main.py
from db.connection import PostgresConnector
from pipeline.extractor import Extractor
from pipeline.loader import Loader
from pipeline.fetcher import YahooFinanceFetcher
from pipeline.transformer import Transformer
from pipeline.stock_indicator import indicator_calculator


# 数据库配置
CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "Finance",
    "user": "postgres",
    "password": "Nature1992."
}

def stock_list():

    stock_lists = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'OKLO', 'NVDA', 'META', 'RKLB', 'CRWV',
        'ORCL', 'APP', 'IREN', 'BTQ', 'HOOD', 'SPFI', 'CRC', 'BMNR', 'BLSH', 'LHX',
        'WGS', 'RBLX', 'AISP', 'BBAI', 'PAYS', 'NBIS', 'RCAT', 'LAES', 'ACHR', 'QMCO',
        'QUBT', 'RGTI', 'IONQ', 'QBTS', 'PLTR', 'MSTR', 'PDYN', 'SERV', 'ZBRA', 'HON',
        'TMO', 'ISRG', 'LVMUY', 'PFE', 'CERS', 'RXRX', 'GH', 'TWST', 'HIMS', 'CRSP',
        'TEM', 'TMDX', 'PSTG', 'SNDK', 'BWXT', 'SMR', 'DFLI', 'LAC', 'CVX', 'LEU', 'CCJ',
        'VST', 'CEG', 'NNE', 'JPM', 'ARM', 'AES', 'PEG', 'DJT', 'RDDT', 'SMCI',
        'MANU', 'TM', 'UNP', 'GE', 'NOV', 'MU', 'NKE', 'IBM', 'SNOW', 'KO',
        'TSM', 'JNJ', 'INTC', 'PG', 'ADP', 'LULU', 'ASML', 'LMT', 'AMD', 'SBUX',
        'QCOM', 'BRK-B', 'AMGN', 'MRK', 'LKNCY', 'NFLX', 'AVGO', 'CRWD', 'LRCX', 'KTOS',
        'VSEC', 'AVAV', 'RTX', 'GLD', 'ACM', 'D', 'NEE', 'NOC', 'XOM', 'BA',
        'ABNB', 'BABA', 'BEAM', 'BIDU', 'COIN', 'CRCL', 'DE', 'DKNG', 'EXAS',
        'GTLB', 'ILMN', 'IRDM', 'NTLA', 'NTRA', 'PACB', 'PD', 'PINS', 'ROKU', 'SHOP',
        'SLMT', 'SOFI', 'TER', 'TTD', 'TXG', 'VCYT', 'XYZ',
        'MA', 'LLY', 'NET', 'DASH', 'NOW', 'V', 'SPOT', 'WELL', 'CDNS',
        'HD', 'GEV', 'WMT', 'ANET', 'KKR', 'COST', 'UBER', 'DUOL', 'DDOG',
        'ALNY', 'DIS', 'ARES', 'CSGP', 'SPGI', 'SYK', 'SNPS', 'WCN', 'HUBS',
        'AXP', 'W', 'WDAY', 'MS', 'TRU', 'NDAQ', 'AFRM', 'HLT', 'MELI',
        'INTU', 'TJX', 'CMG', 'VRTX', 'WSO', 'PODD', 'DOCS', 'IOT', 'U',
        'ENSG', 'ADI', 'PEN', 'CHWY', 'MPWR', 'UCB', 'SN', 'AUR', 'KNF',
        'LMND', 'MSCI', 'CPAY', 'ODD', 'MRNA', 'EW', 'LINE', 'SG',
        'FIG', 'YETI', 'INSP', 'DNLI', 'RIVN', 'GLOB', 'SANA', 'DNA',
        'ADBE', 'CRM', 'PANW', 'FTNT', 'MDB', 'AI', 'UPST', 'PATH',
        'BLK', 'GS', 'BAC', 'WFC', 'F', 'HOG', 'PCAR', 'AGCO', 'OSK', 'TEX',
        'GD', 'HII', 'ETN', 'EMR', 'ROK', 'ITW', 'BIIB','AZN','IONS','ROIV','ONC','ARGX','BNTX','ADCT','TLSA','BTAI','SNGX','TRVN',
        'CSX', 'NSC', 'TXT', 'SPR', 'DOW', 'DD', 'LYB', 'EMN',
        'MCD', 'BKNG', 'APLD', 'TOST', 'BROS', 'GLBE', 'SCHW', 'BSX', 'NRG',
        'HWM', 'FUTU', 'DAVE', 'HSAI', 'CLS','TEAM', 'OKTA', 'ZS', 'TWLO', 'ZM', 'ETSY', 'CART', 'MRVL', 'LSCC', 'AMAT', 'KLAC', 'REGN', 'ABT', 'DXCM',
        'PYPL', 'CMCSA', 'EBAY', 'KVYO', 'BILL', 'COUR', 'FROG',
        'CL', 'UL', 'PEP', 'MDLZ', 'GIS', 'KHC', 'SYY', 'TAP',
        'OXY', 'PSX', 'KMI', 'COP', 'SO', 'DUK', 'AEP',
        'PGR', 'TRV', 'ALL', 'ABBV', 'BMY', 'GILD', 'VZ', 'T',
        'PLD', 'O', 'TGT', 'KR', 'UPS', 'FDX', 'MCO', 'BK', 'USB'
    ]



    return stock_lists
    

def main():

    with PostgresConnector(CONFIG) as cur:
        
        yahoo_fetcher = YahooFinanceFetcher()
        stock_daily_data = yahoo_fetcher.fetch_multiple_daily_stocks(stock_list())

        transformer = Transformer()
        stock_daily_data = transformer.remove_date_time(stock_daily_data)
        print(stock_daily_data.head())

        loader = Loader(cur)
        loader.load_ticker_daily_to_db(stock_daily_data)

        # calculate indicators and update to db
        extractor = Extractor(cur)
        all_daily_data = extractor.extract_all_daily_stock_data()
        print(all_daily_data)

        ic = indicator_calculator(all_daily_data)
        results = ic.comprehensive_indicator()

        results.to_csv("data.csv", index=False)



if __name__ == "__main__":
    main()
