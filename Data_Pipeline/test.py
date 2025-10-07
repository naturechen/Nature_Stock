# ==================== 数据存储层 ====================
class DatabaseManager:

    
    def __init__(self, config: DatabaseConfig):
        """
        初始化数据库管理器
        
        初始化流程：
        1. 保存配置
        2. 创建连接池
        3. 创建表结构和索引
        
        为什么在初始化时创建表：
        - 确保表存在，避免后续插入失败
        - 集中管理DDL，便于维护
        - 使用IF NOT EXISTS，幂等操作，可重复执行
        """
        self.config = config
        self.pool = self._create_connection_pool()
        self._create_tables()
    
    def _create_connection_pool(self) -> ThreadedConnectionPool:
        """
        创建数据库连接池
        
        为什么使用ThreadedConnectionPool：
        1. 线程安全：多线程并发获取连接不会冲突
        2. 自动管理：自动处理连接的分配和回收
        3. 资源限制：max_conn限制最大连接数，防止资源耗尽
        4. 预热机制：min_conn保持最小连接数，减少冷启动延迟
        
        连接池参数说明：
        - minconn：最小连接数，始终保持活跃
        - maxconn：最大连接数，限制并发上限
        - 其他参数：数据库连接信息
        
        返回：
        - ThreadedConnectionPool对象，线程安全的连接池
        """
        return ThreadedConnectionPool(
            self.config.min_conn,  # 最小连接数
            self.config.max_conn,  # 最大连接数
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password
        )
    


# 导入yfinance：用于从Yahoo Finance获取股票历史数据
# 选择理由：官方推荐的Yahoo Finance API封装，免费且稳定
import yfinance as yf

# 导入pandas：用于数据处理和分析
# 选择理由：yfinance返回DataFrame格式，pandas是最佳处理工具
import pandas as pd

# 导入psycopg2：PostgreSQL数据库驱动
# 选择理由：Python最成熟的PostgreSQL适配器，性能优秀
import psycopg2
from psycopg2.extras import execute_batch  # 批量插入，比逐行插入快10-100倍
from psycopg2.pool import ThreadedConnectionPool  # 连接池，避免频繁创建连接

# 导入ABC（抽象基类）：用于定义接口和规范
# 设计理由：通过抽象类定义数据获取接口，未来可轻松切换数据源（如Alpha Vantage）
from abc import ABC, abstractmethod

# 导入类型提示：提高代码可读性和IDE支持
from typing import List, Dict, Optional

# 导入datetime：处理时间数据
from datetime import datetime, timedelta

# 导入dataclass：简化数据类定义
# 选择理由：自动生成__init__、__repr__等方法，减少样板代码
from dataclasses import dataclass

# 导入Enum：定义枚举类型
# 选择理由：时间间隔使用枚举，避免魔法字符串，提高代码安全性
from enum import Enum

# 导入logging：日志记录
# 设计理由：生产环境必须有完善的日志，便于调试和监控
import logging

# 导入并发工具：实现多线程并发
# 选择理由：I/O密集型任务（网络请求、数据库写入）使用线程池可显著提升性能
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入time：用于重试延迟
import time


# ==================== 配置类 ====================
@dataclass
class DatabaseConfig:

    host: str  # 数据库主机地址
    port: int  # 数据库端口，PostgreSQL默认5432
    database: str  # 数据库名称
    user: str  # 数据库用户名
    password: str  # 数据库密码
    min_conn: int = 2  # 连接池最小连接数，保持基础连接避免冷启动
    max_conn: int = 10  # 连接池最大连接数，控制并发上限


@dataclass
class PipelineConfig:
    """
    管道配置类
    
    为什么需要这些参数：
    - batch_size：批量插入大小，平衡内存使用和数据库性能
      · 太小：频繁IO，降低吞吐量
      · 太大：占用内存，可能超时
      · 100是经验值，适合大多数场景
    
    - max_workers：并发线程数
      · 太少：无法充分利用网络和CPU
      · 太多：线程切换开销大，可能触发API限流
      · 5是平衡值，适合Yahoo Finance API的限流策略
    
    - retry_attempts：重试次数，应对网络抖动和临时错误
    - retry_delay：重试延迟（秒），避免立即重试加重服务器负担
    """
    batch_size: int = 100  # 批量插入数据库的记录数
    max_workers: int = 5  # 并发工作线程数
    retry_attempts: int = 3  # 失败重试次数
    retry_delay: int = 2  # 重试延迟时间（秒）


class TimeInterval(Enum):
    """
    时间间隔枚举类
    
    为什么使用枚举而不是字符串：
    1. 类型安全：编译时检查，避免拼写错误（"1d" vs "1day"）
    2. IDE支持：自动补全，降低使用门槛
    3. 文档化：枚举本身就是可用值的文档
    4. 易于扩展：添加新间隔只需增加枚举值
    
    Yahoo Finance支持的间隔：
    - 1d: 日线数据，适合长期分析
    - 60m: 60分钟（1小时）数据，适合日内分析
    - 1wk: 周线数据，适合宏观趋势分析
    """
    DAILY = "1d"  # 日线数据
    HOURLY = "60m"  # 60分钟（小时）数据
    WEEKLY = "1wk"  # 周线数据


# ==================== 日志配置 ====================
"""
配置全局日志系统

为什么需要日志：
1. 生产环境必备：便于问题定位和性能监控
2. 异步任务追踪：并发场景下追踪每个任务的执行状态
3. 审计记录：记录数据获取和存储的完整过程

日志配置说明：
- level=INFO：记录关键操作，不记录调试细节（生产环境标准）
- format包含时间戳、模块名、级别、消息，便于分析
"""
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 数据模型 ====================
@dataclass
class StockData:
    """
    股票数据模型类
    
    为什么需要数据模型：
    1. 类型安全：明确每个字段的数据类型，避免类型错误
    2. 数据验证：在构造时进行类型检查（Python 3.10+支持运行时检查）
    3. 解耦：隔离数据结构和业务逻辑，便于修改和测试
    4. 文档化：字段定义即文档，清晰表达数据结构
    
    字段设计说明：
    - symbol：股票代码（如AAPL），主键之一
    - timestamp：时间戳，精确到秒，主键之一
    - price字段使用float：财务数据需要高精度
    - volume使用int：交易量是整数
    - interval：时间间隔，用于区分不同粒度的数据（主键之一）
    
    为什么timestamp和interval都作为主键：
    - 同一股票在同一时间点可能有不同间隔的数据
    - 例如：AAPL在2024-01-01既有日线数据，也有周线数据
    """
    symbol: str  # 股票代码
    timestamp: datetime  # 数据时间戳
    open_price: float  # 开盘价
    high_price: float  # 最高价
    low_price: float  # 最低价
    close_price: float  # 收盘价
    volume: int  # 成交量
    interval: str  # 时间间隔（1d, 60m, 1wk等）


# ==================== 数据获取层 ====================
class DataFetcher(ABC):
    """
    数据获取抽象基类
    
    为什么使用抽象基类（设计模式：策略模式）：
    1. 开闭原则：对扩展开放，对修改关闭
       - 未来可以添加AlphaVantageFetcher、TiingoFetcher等
       - 不需要修改Pipeline代码，只需实现新的Fetcher
    
    2. 依赖倒置：高层模块（Pipeline）依赖抽象，不依赖具体实现
       - Pipeline不关心数据从哪来，只关心接口契约
    
    3. 便于测试：可以创建MockFetcher用于单元测试
       - 不需要真实调用API，提高测试速度和可靠性
    
    4. 多数据源支持：同一个Pipeline可以同时支持多个数据源
    """
    
    @abstractmethod
    def fetch(self, symbol: str, interval: TimeInterval, 
              start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        获取股票数据的抽象方法（必须由子类实现）
        
        为什么返回DataFrame：
        - pandas是数据分析的标准库，生态丰富
        - DataFrame便于后续数据清洗、转换、分析
        - 与yfinance的返回格式一致，减少转换成本
        
        参数说明：
        - symbol: 股票代码，遵循Yahoo Finance规范（如AAPL, 0700.HK）
        - interval: 时间间隔枚举，确保类型安全
        - start_date/end_date: 时间范围，datetime类型避免字符串解析错误
        
        返回值：
        - 成功：包含OHLCV数据的DataFrame
        - 失败：空DataFrame（不抛异常，让调用方决定如何处理）
        """
        pass


class YahooFinanceFetcher(DataFetcher):
    """
    Yahoo Finance 数据获取器实现类
    
    为什么选择Yahoo Finance：
    1. 免费：无需API密钥，适合个人和小型项目
    2. 数据全面：覆盖全球主要市场，历史数据丰富
    3. 稳定性好：Yahoo Finance是成熟的金融数据提供商
    4. yfinance库成熟：社区活跃，文档完善
    
    实现要点：
    - 继承DataFetcher，实现统一接口
    - 包含重试机制，提高可靠性
    - 添加详细日志，便于监控和调试
    """
    
    def __init__(self, config: PipelineConfig):
        """
        初始化数据获取器
        
        为什么需要注入配置：
        - 依赖注入模式，提高可测试性
        - 配置集中管理，便于调整参数
        - 避免硬编码，提高代码灵活性
        """
        self.config = config
    
    def fetch(self, symbol: str, interval: TimeInterval, 
              start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        从Yahoo Finance获取股票数据
        
        实现逻辑：
        1. 重试机制：应对网络抖动和临时错误
        2. 数据验证：检查返回数据是否为空
        3. 数据增强：添加symbol和interval字段，便于后续处理
        4. 异常处理：捕获所有异常，记录日志，返回空DataFrame
        
        为什么使用重试机制：
        - 网络请求不可靠：可能遇到超时、连接重置等临时问题
        - API限流：偶尔遇到限流，等待后重试可能成功
        - 提高成功率：3次重试可覆盖大部分临时故障
        
        为什么延迟重试：
        - 立即重试可能仍然失败（问题未恢复）
        - 避免加重服务器负担（触发更严格的限流）
        - 2秒是经验值，适合大多数场景
        
        Args:
            symbol: 股票代码
            interval: 时间间隔
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 股票数据（包含Date/Datetime, Open, High, Low, Close, Volume, Symbol, Interval）
                      如果失败或无数据，返回空DataFrame
        """
        for attempt in range(self.config.retry_attempts):
            try:
                # 创建Ticker对象
                # yfinance的Ticker是股票的抽象，封装了所有数据获取方法
                ticker = yf.Ticker(symbol)
                
                # 获取历史数据
                # history方法返回OHLCV数据的DataFrame
                # interval参数决定数据粒度（日线、小时线等）
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval.value  # 使用枚举的value属性获取字符串值
                )
                
                # 数据验证：检查是否获取到数据
                # 原因：某些股票在特定时间范围内可能没有数据（如新上市股票）
                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    return pd.DataFrame()
                
                # 重置索引：将日期从索引转换为列
                # 原因：后续处理需要将日期作为字段存储到数据库
                df = df.reset_index()
                
                # 添加元数据字段
                # 原因：数据库需要symbol和interval字段来区分不同的数据
                df['Symbol'] = symbol
                df['Interval'] = interval.value
                
                logger.info(f"Successfully fetched {len(df)} records for {symbol} ({interval.value})")
                return df
                
            except Exception as e:
                # 记录错误日志，包含尝试次数
                logger.error(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                
                # 判断是否需要重试
                if attempt < self.config.retry_attempts - 1:
                    # 延迟后重试
                    time.sleep(self.config.retry_delay)
                else:
                    # 所有重试都失败，记录最终错误
                    logger.error(f"Failed to fetch data for {symbol} after {self.config.retry_attempts} attempts")
                    return pd.DataFrame()


# ==================== 数据转换层 ====================
class DataTransformer:
    """
    数据转换器类
    
    为什么需要独立的转换层：
    1. 单一职责：专注于数据格式转换，与获取和存储解耦
    2. 可复用性：转换逻辑可用于不同的数据源
    3. 易于测试：纯函数转换，容易编写单元测试
    4. 易于扩展：可以添加数据清洗、特征工程等功能
    
    为什么使用静态方法：
    - 转换逻辑无状态，不需要实例属性
    - 静态方法表明这是一个纯函数转换
    - 可以直接通过类调用，无需实例化
    """
    
    @staticmethod
    def transform(df: pd.DataFrame) -> List[StockData]:
        """
        转换DataFrame为StockData对象列表
        
        为什么需要这个转换：
        1. 类型安全：DataFrame是动态类型，StockData是强类型
        2. 数据验证：转换过程中可以验证数据完整性
        3. 解耦：数据库层不需要知道pandas，只需要知道StockData
        4. 可移植性：StockData可以用于其他存储系统（如MongoDB、CSV）
        
        转换逻辑：
        1. 检查空DataFrame，避免无效迭代
        2. 遍历每一行，转换为StockData对象
        3. 异常处理：跳过有问题的行，不影响整体处理
        4. 类型转换：确保数据类型正确（float、int）
        
        为什么使用iterrows而不是apply：
        - iterrows更直观，便于理解和调试
        - 这里需要创建对象，不适合向量化操作
        - 性能影响不大（数据量通常不超过几千行）
        
        为什么捕获异常后continue：
        - 部分数据损坏不应影响整体处理
        - 记录错误日志，便于后续排查
        - 最大化数据利用率，不因个别错误而全盘失败
        
        Args:
            df: 原始DataFrame，包含Yahoo Finance返回的数据
            
        Returns:
            List[StockData]: 转换后的数据对象列表
                           如果输入为空或全部转换失败，返回空列表
        """
        # 空数据检查：避免无效处理
        if df.empty:
            return []
        
        stock_data_list = []
        
        # 遍历DataFrame的每一行
        for _, row in df.iterrows():
            try:
                # 创建StockData对象
                # 字段映射说明：
                # - Date/Datetime: yfinance根据interval返回不同的时间字段名
                #   · 日线和周线使用'Date'
                #   · 分钟线使用'Datetime'
                # - Open/High/Low/Close/Volume: yfinance的标准字段名
                stock_data = StockData(
                    symbol=row['Symbol'],
                    # 处理不同的时间字段名（日线用Date，分钟线用Datetime）
                    timestamp=row['Date'] if 'Date' in row else row['Datetime'],
                    # 显式转换为float，确保数据类型正确
                    open_price=float(row['Open']),
                    high_price=float(row['High']),
                    low_price=float(row['Low']),
                    close_price=float(row['Close']),
                    # 显式转换为int，交易量必须是整数
                    volume=int(row['Volume']),
                    interval=row['Interval']
                )
                stock_data_list.append(stock_data)
            except Exception as e:
                # 记录转换失败的行，便于后续排查
                # 不抛出异常，允许处理继续进行
                logger.error(f"Error transforming row: {str(e)}")
                continue
        
        return stock_data_list


# ==================== 数据存储层 ====================
class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool = self._create_connection_pool()
        self._create_tables()
    
    def _create_connection_pool(self) -> ThreadedConnectionPool:
        """创建连接池"""
        return ThreadedConnectionPool(
            self.config.min_conn,
            self.config.max_conn,
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password
        )
    
    def _create_tables(self):
        """创建数据表"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS stock_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            open_price DECIMAL(20, 4),
            high_price DECIMAL(20, 4),
            low_price DECIMAL(20, 4),
            close_price DECIMAL(20, 4),
            volume BIGINT,
            interval VARCHAR(10) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp, interval)
        );
        
        CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
        ON stock_data(symbol, timestamp);
        
        CREATE INDEX IF NOT EXISTS idx_interval 
        ON stock_data(interval);
        """
        
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(create_table_sql)
                conn.commit()
                logger.info("Database tables created successfully")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating tables: {str(e)}")
            raise
        finally:
            self.pool.putconn(conn)
    
    def insert_batch(self, data_list: List[StockData]) -> int:
        """
        批量插入数据
        
        Args:
            data_list: 数据列表
            
        Returns:
            int: 插入的行数
        """
        if not data_list:
            return 0
        
        insert_sql = """
        INSERT INTO stock_data 
        (symbol, timestamp, open_price, high_price, low_price, close_price, volume, interval)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, timestamp, interval) 
        DO UPDATE SET
            open_price = EXCLUDED.open_price,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price,
            close_price = EXCLUDED.close_price,
            volume = EXCLUDED.volume
        """
        
        values = [
            (
                data.symbol,
                data.timestamp,
                data.open_price,
                data.high_price,
                data.low_price,
                data.close_price,
                data.volume,
                data.interval
            )
            for data in data_list
        ]
        
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cursor:
                execute_batch(cursor, insert_sql, values, page_size=100)
                conn.commit()
                logger.info(f"Successfully inserted/updated {len(values)} records")
                return len(values)
        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting batch: {str(e)}")
            raise
        finally:
            self.pool.putconn(conn)
    
    def close(self):
        """关闭连接池"""
        if self.pool:
            self.pool.closeall()
            logger.info("Database connection pool closed")


# ==================== 主管道 ====================
class StockDataPipeline:
    """股票数据管道"""
    
    def __init__(self, 
                 db_config: DatabaseConfig,
                 pipeline_config: PipelineConfig):
        self.db_manager = DatabaseManager(db_config)
        self.fetcher = YahooFinanceFetcher(pipeline_config)
        self.transformer = DataTransformer()
        self.pipeline_config = pipeline_config
    
    def _process_single_stock(self, 
                             symbol: str,
                             interval: TimeInterval,
                             start_date: datetime,
                             end_date: datetime) -> int:
        """
        处理单个股票数据
        
        Returns:
            int: 插入的记录数
        """
        try:
            # 获取数据
            df = self.fetcher.fetch(symbol, interval, start_date, end_date)
            
            if df.empty:
                return 0
            
            # 转换数据
            stock_data_list = self.transformer.transform(df)
            
            # 存储数据
            inserted_count = self.db_manager.insert_batch(stock_data_list)
            
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            return 0
    
    def run(self,
            symbols: List[str],
            intervals: List[TimeInterval],
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None) -> Dict[str, int]:
        """
        运行数据管道
        
        Args:
            symbols: 股票代码列表
            intervals: 时间间隔列表
            start_date: 开始日期（默认：1年前）
            end_date: 结束日期（默认：今天）
            
        Returns:
            Dict: 统计信息
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        logger.info(f"Starting pipeline for {len(symbols)} symbols and {len(intervals)} intervals")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        
        total_inserted = 0
        stats = {}
        
        # 创建任务列表
        tasks = [
            (symbol, interval, start_date, end_date)
            for symbol in symbols
            for interval in intervals
        ]
        
        # 并发执行任务
        with ThreadPoolExecutor(max_workers=self.pipeline_config.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    self._process_single_stock,
                    symbol, interval, start_date, end_date
                ): (symbol, interval)
                for symbol, interval, start_date, end_date in tasks
            }
            
            for future in as_completed(future_to_task):
                symbol, interval = future_to_task[future]
                try:
                    count = future.result()
                    total_inserted += count
                    key = f"{symbol}_{interval.value}"
                    stats[key] = count
                except Exception as e:
                    logger.error(f"Task failed for {symbol} ({interval.value}): {str(e)}")
        
        logger.info(f"Pipeline completed. Total records inserted: {total_inserted}")
        return stats
    
    def close(self):
        """关闭管道"""
        self.db_manager.close()


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 配置数据库
    db_config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="stock_data",
        user="your_username",
        password="your_password",
        min_conn=2,
        max_conn=10
    )
    
    # 配置管道
    pipeline_config = PipelineConfig(
        batch_size=100,
        max_workers=5,
        retry_attempts=3,
        retry_delay=2
    )
    
    # 创建管道
    pipeline = StockDataPipeline(db_config, pipeline_config)
    
    try:
        # 定义要获取的股票和时间间隔
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        intervals = [TimeInterval.DAILY, TimeInterval.HOURLY, TimeInterval.WEEKLY]
        
        # 运行管道
        stats = pipeline.run(
            symbols=symbols,
            intervals=intervals,
            start_date=datetime.now() - timedelta(days=30),  # 最近30天
            end_date=datetime.now()
        )
        
        # 打印统计信息
        print("\n=== Pipeline Statistics ===")
        for key, count in stats.items():
            print(f"{key}: {count} records")
            
    finally:
        pipeline.close()

