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

    stock_lists = stock_categories = [

        # 1. 美股大盘核心 / “压舱石”
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META','NVDA', 'TSLA', 'TSM', 'ASML', 'AVGO', 'AMD',
        'ORCL', 'ADBE', 'CRM', 'INTU','KO', 'PEP', 'MCD', 'WMT', 'TGT', 'KR','DIS', 'NFLX', 'NKE', 'LULU', 'COST', 'TJX',
        'BRK-B', 'UNP', 'CSX', 'NSC','XOM', 'CVX', 'OXY', 'PSX','JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'BK', 'USB',
        'SPGI', 'MSCI', 'NDAQ','LLY', 'JNJ', 'MRK', 'AMGN', 'ABBV', 'BMY','GE', 'ETN', 'EMR', 'ROK', 'ITW',

    # 2. 量子计算 / 量子安全
        'ARQQ', 'QUBT', 'RGTI', 'IONQ', 'QBTS', 'QSI'
    
    # 3. AI / 云 / 网络安全（偏“新质生产力”）
        'PLTR', 'CRWD', 'NET', 'PANW', 'FTNT', 'ZS', 'OKTA', 'S', 'CYBR', 'TENB', 'VRNS',
        'SNOW', 'NOW', 'TTD', 'MDB', 'AI', 'PATH', 'DDOG', 'TEAM', 'GTLB', 'HUBS','PD', 'TWLO', 'ZM', 'ETSY', 'SHOP',
        'BBAI', 'BTQ', 'AISP', 'NBIS', 'HSAI','COIN', 'HOOD','DOCS', 'IOT','COUR', 'FROG',
    
    # 4. SaaS / 软件平台（偏稳定现金流）
        'ADBE', 'CRM', 'INTU','NOW', 'SNOW', 'TTD', 'CDNS', 'SNPS', 'CDNS',
        'WIX', 'MNDY', 'APPN', 'BPMC','WDAY', 'CSGP','DUOL','HUBS', 'DOCS', 'PATH','SPOT','CSGP',

    # 5. 半导体 / 芯片 / 硬件 & 部分机器人
        'NVDA', 'AMD', 'AVGO', 'QCOM','TSM', 'INTC', 'MU', 'MRVL', 'LSCC','AMAT', 'KLAC', 'LRCX', 'ASML','ON', 'SITM', 'WOLF', 'AEHR', 'RMBS',
        'COHR', 'GFS', 'AMBA', 'POET','ADI', 'MPWR','ARM','ZBRA', 'HON','ISRG',   # 机器人手术，半导体+MedTech交叉

    # 6. 生物科技 / 制药 / MedTech / 医疗设备
        # 大型制药 & 大型生科
        'PFE', 'AMGN', 'MRK', 'JNJ', 'LLY','BIIB', 'AZN', 'ABBV', 'BMY', 'GILD','REGN', 'VRTX', 'BNTX', 'ARGX',

        # Biotech 平台 / 创新药
        'ARCT', 'BMRN', 'BLUE', 'CRBU', 'EDIT','HALO', 'IMVT', 'MNMD', 'KYMR','NTLA', 'NTRA', 'PACB', 'TXG', 'VCYT','CERS', 'RXRX', 'GH', 'TWST', 'CRSP',
        'SDGR', 'DTIL', 'NRIX','ABCL', 'EVGN', 'SANA', 'DNA',

        # 其他创新 / 罕见病 / 细分
        'IONS', 'ROIV', 'ONC', 'ADCT', 'TLSA', 'BTAI', 'SNGX', 'TRVN','ALNY', 'DNLI', 'INSP',

        # MedTech / 器械 / 服务
        'TMO', 'ISRG', 'SYK', 'BSX', 'DXCM', 'ABT','MRNA', 'EW', 'UCB', 'MASI', 'INMD', 'TMDX', 'PODD','ENSG', 'PEN', 'HIMS','WELL',   # 医疗 REIT 但偏医疗属性

    # 7. 新能源 / EV / 电池 / 氢能 / 核能

        # 太阳能
        'ENPH', 'SEDG', 'FSLR', 'RUN', 'SPWR',

        # 氢能 / 燃料电池
        'PLUG', 'BLDP', 'FCEL', 'FLNC', 'BE',

        # 电动车 / 三电
        'TSLA', 'LCID', 'RIVN', 'F','LI', 'XPEV', 'BYDDY','DFLI', 'WKHS',

        # 充电桩 / 充电基础设施
        'BLNK', 'CHPT', 'WBX',

        # 电池材料
        'ALB', 'PLL', 'LAC',

        # 风光+绿电运营 / 公用事业里偏绿电
        'NEE', 'AES', 'PEG', 'NRG',

        # 核电 / 核燃料 / 小堆
        'BWXT', 'SMR', 'LTBR', 'LEU', 'CCJ','VST', 'CEG', 'OKLO',

        # 清洁能源 REIT / ESG
        'HASI',


    # 8. 航空航天 / 国防军工 / 太空 & eVTOL
   
        # 太空 & 发射
        'RKLB', 'MAXR', 'PL', 'ASTS', 'SPIR', 'RDW', 'SPCE', 'SATL', 'MNTS',

        # eVTOL / 新型飞行器
        'JOBY', 'ACHR', 'LILM', 'AERO',

        # 传统军工 & 防务
        'LMT', 'NOC', 'RTX', 'GD', 'HII','KTOS', 'AVAV', 'VSEC', 'BWXT',

        # 航空 / 机体制造
        'BA', 'TXT', 'SPR', 'HWM',


    # 9. 金融 / 银行 / 券商 / 保险 / 支付 & 金融科技

        # 银行 & 资产管理
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'BK', 'USB', 'SCHW', 'SPFI',

        # 支付 & 卡组织 & 信评
        'V', 'MA', 'AXP', 'CPAY','MSCI', 'SPGI', 'NDAQ', 'TRU',

        # 保险
        'PGR', 'TRV', 'ALL',

        # 互联网金融/Fintech
        'SQ', 'NU', 'PYPL', 'SOFI', 'AFRM', 'UPST','HOOD', 'COIN', 'FUTU', 'PAYS',


    # 10. 消费互联网 / 广告 / 电商 / 出行
        'ABNB', 'UBER', 'DASH', 'MELI', 'BKNG','PINS', 'ROKU', 'SHOP', 'ETSY','SPOT', 'RBLX','BABA', 'BIDU', 'FUTU','GLBE', 'CART', 'W', 'DUOL', 'META', 'SNAP', 
   

    # 11. 线下消费 / 连锁餐饮 / 零售品牌
        'MCD', 'CMG', 'SBUX', 'BROS', 'TOST','WMT', 'TGT', 'KR', 'COST', 'TJX','YETI', 'CHWY','NKE', 'LULU',

    # 12. 日化 / 食品饮料 / 必选消费
        'CL', 'UL', 'PEP', 'KO', 'MDLZ', 'GIS', 'KHC', 'SYY', 'TAP','PG',

    # 13. 工业 / 机械 / 运输 / 物流
        'UNP', 'CSX', 'NSC','UPS', 'FDX','DE', 'PCAR', 'AGCO', 'OSK', 'TEX','GE', 'HWM', 'ETN', 'EMR', 'ROK', 'ITW', 'ACM','TXT', 'SPR','NRG',   # 也可归入公用事业

    # 14. 公用事业 / 电力 / 通信运营商
        'SO', 'DUK', 'AEP', 'D', 'NEE', 'AES', 'PEG', 'VST', 'CEG', 'NRG','VZ', 'T',

    # 15. REIT / 基础设施
        'PLD', 'O', 'WELL', 'WCN',               

    # 16. 原材料 / 化工
        'DOW', 'DD', 'LYB', 'EMN', 'ALB', 'PLL', 'LAC', 'ORGN',

    # 17. AR / VR / 传感器 / 自动驾驶（偏硬科技）
        'VUZI', 'LIDR', 'AEVA','MBLY', 'LAZR', 'AUR', 'GOEV','TSLA', 'RIVN', 'LCID','LI', 'XPEV', 'BYDDY','ON', 'NVDA', 'U', 'META',  

    # 18. 纯“故事股”/ 高波动创新小票（部分难以简单归类）

        'OKLO', 'RKLB', 'CRWV', 'IREN', 'BTQ','WGS', 'RBLX', 'RCAT', 'LAES', 'DFLI', 'CRCL','SLMT', 'XYZ', 'GEV', 'DAVE', 'HSAI', 'HWM', 'HOG','MANU', 'DJT', 'RDDT',
        'SPCE', 'SATL', 'MNTS', 'AERO','BMNR', 'BLSH', 'TEM', 'NNE', 'SLMT', 'XYZ',
        # 以及你列表里一些我没法确定行业的小市值股票
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
