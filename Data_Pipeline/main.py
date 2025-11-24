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

        # 1. 美股大盘核心 / “压舱石”
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META','NVDA', 'TSLA', 'TSM', 'ASML', 'AVGO', 'AMD',
        'ORCL', 'ADBE', 'CRM', 'INTU','KO', 'PEP', 'MCD', 'WMT', 'TGT', 'KR','DIS', 'NFLX', 'NKE', 'LULU', 'COST', 'TJX',
        'BRK-B', 'UNP', 'CSX', 'NSC','XOM', 'CVX', 'OXY', 'PSX','JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'BK', 'USB',
        'SPGI', 'MSCI', 'NDAQ','LLY', 'JNJ', 'MRK', 'AMGN', 'ABBV', 'BMY','GE', 'ETN', 'EMR', 'ROK', 'ITW',

        # 2. 量子计算 / 量子安全
        'ARQQ', 'QUBT', 'RGTI', 'IONQ', 'QBTS', 'QSI',
        
        # 3. AI / 云 / 网络安全（偏“新质生产力”）
        'PLTR', 'CRWD', 'NET', 'PANW', 'FTNT', 'ZS', 'OKTA', 'S', 'CYBR', 'TENB', 'VRNS',
        'SNOW', 'NOW', 'TTD', 'MDB', 'AI', 'PATH', 'DDOG', 'TEAM', 'GTLB', 'HUBS','PD', 'TWLO', 'ZM', 'ETSY', 'SHOP',
        'BBAI', 'BTQ', 'AISP', 'NBIS', 'HSAI','COIN', 'HOOD','DOCS', 'IOT','COUR', 'FROG',
        
        # 4. SaaS / 软件平台（偏稳定现金流）
        'CDNS', 'SNPS', 'WIX', 'MNDY', 'APPN','WDAY', 'CSGP','DUOL','SPOT',

        # 5. 半导体 / 芯片 / 硬件 & 部分机器人
        'QCOM','INTC', 'MU', 'MRVL', 'LSCC','AMAT', 'KLAC', 'LRCX', 'ON', 'SITM', 'WOLF', 'AEHR', 'RMBS',
        'COHR', 'GFS', 'AMBA', 'POET','ADI', 'MPWR','ARM','ZBRA', 'HON','ISRG',

        # 6. 生物科技 / 制药 / MedTech / 医疗设备
        'PFE','BIIB', 'AZN','GILD','REGN', 'VRTX', 'BNTX', 'ARGX',
        'ARCT', 'BMRN', 'CRBU', 'EDIT','HALO', 'IMVT', 'MNMD', 'KYMR','NTLA', 'NTRA', 'PACB', 'TXG', 'VCYT','CERS',
        'RXRX', 'GH', 'TWST', 'CRSP', 'SDGR', 'DTIL', 'NRIX','ABCL', 'EVGN', 'SANA', 'DNA',
        'IONS', 'ROIV', 'ONC', 'ADCT', 'TLSA', 'BTAI', 'SNGX', 'TRVN','ALNY', 'DNLI', 'INSP',
        'SYK', 'BSX', 'DXCM', 'ABT','MRNA', 'EW', 'UCB', 'MASI', 'INMD', 'TMDX', 'PODD','ENSG', 'PEN', 'HIMS','WELL',

        # 7. 新能源 / EV / 电池 / 氢能 / 核能
        'ENPH', 'SEDG', 'FSLR', 'RUN', 'SPWR','PLUG', 'BLDP', 'FCEL', 'FLNC', 'BE', 'LCID', 'RIVN', 'F','LI', 'XPEV', 'BYDDY','DFLI', 'WKHS',
        'BLNK', 'CHPT', 'WBX', 'ALB', 'LAC','NEE', 'AES', 'PEG', 'NRG','BWXT', 'SMR', 'LTBR', 'LEU', 'CCJ','VST', 'CEG', 'OKLO','HASI', 'COP',

        # 8. 航空航天 / 国防军工 / 太空 & eVTOL
        'RKLB', 'PL', 'ASTS', 'SPIR', 'RDW', 'SPCE', 'SATL', 'MNTS','JOBY', 'ACHR', 'AERO',
        'LMT', 'NOC', 'RTX', 'GD', 'HII','KTOS', 'AVAV', 'VSEC','BA', 'TXT', 'SPR', 'HWM',

        # 9. 金融 / 银行 / 券商 / 保险 / 支付 & 金融科技
        'SCHW', 'SPFI','V', 'MA', 'AXP', 'CPAY','TRU','PGR', 'TRV', 'ALL','NU', 'PYPL', 'SOFI', 'AFRM', 'UPST','FUTU', 'PAYS',

        # 10. 消费互联网 / 广告 / 电商 / 出行
        'ABNB', 'UBER', 'DASH', 'MELI', 'BKNG','PINS', 'ROKU', 'BABA', 'BIDU', 'GLBE', 'CART', 'W', 'SNAP', 'RBLX',

        # 11. 线下消费 / 连锁餐饮 / 零售品牌
        'CMG', 'SBUX', 'BROS', 'TOST','YETI', 'CHWY',

        # 12. 日化 / 食品饮料 / 必选消费
        'CL', 'UL', 'MDLZ', 'GIS', 'KHC', 'SYY', 'TAP','PG',

        # 13. 工业 / 机械 / 运输 / 物流
        'UPS', 'FDX','DE', 'PCAR', 'AGCO', 'OSK', 'TEX','ACM',

        # 14. 公用事业 / 电力 / 通信运营商
        'SO', 'DUK', 'AEP', 'VZ', 'T',

        # 15. REIT / 基础设施
        'PLD', 'O', 'WELL', 'WCN',

        # 16. 原材料 / 化工
        'DOW', 'DD', 'LYB', 'EMN', 'ORGN',

        # 17. AR / VR / 传感器 / 自动驾驶（偏硬科技）
        'VUZI', 'LIDR', 'AEVA','MBLY', 'LAZR', 'AUR', 'GOEV','U',

        # 18. 纯“故事股”/ 高波动创新小票
        'OKLO', 'CRWV', 'IREN', 'BTQ','WGS', 'RCAT', 'LAES', 'DFLI', 'CRCL','SLMT', 'XYZ', 'GEV', 'DAVE', 'HSAI', 'HOG','MANU', 'DJT', 'RDDT',
        'TEM', 'NNE',"ADP", "ANET", "APLD", "APP", "ARES","BEAM", "BILL", "BLSH", "BMNR", "CLS","CMCSA", "COP", "CRC", "D", "DKNG","EBAY", "EXAS", "FIG", "GLD", "GLOB",
        "GOEV", "HD", "HLT", "IBM", "ILMN","IRDM", "KKR", "KMI", "KNF", "KVYO","LHX", "LINE", "LKNCY", "LMND", "LVMUY",
        "MCO", "MSTR", "NOV", "ODD", "PDYN","PSTG", "QMCO", "SERV", "SG", "SMCI", "SN", "SNDK", "TER", "TM", "TMO","TRVN", "WSO"
    ]

    return stock_lists
    

def main():

    with PostgresConnector(CONFIG) as cur:
        
        # yahoo_fetcher = YahooFinanceFetcher()
        # stock_daily_data = yahoo_fetcher.fetch_multiple_daily_stocks(stock_list())

        # transformer = Transformer()
        # stock_daily_data = transformer.remove_date_time(stock_daily_data)
        # print(stock_daily_data.head())

        # loader = Loader(cur)
        # loader.load_ticker_daily_to_db(stock_daily_data)

        # calculate indicators and update to db
        extractor = Extractor(cur)
        all_daily_data = extractor.extract_all_daily_stock_data()
        print(all_daily_data)

        ic = indicator_calculator(all_daily_data)
        results = ic.comprehensive_indicator()

        results.to_csv("data.csv", index=False)



if __name__ == "__main__":
    main()
