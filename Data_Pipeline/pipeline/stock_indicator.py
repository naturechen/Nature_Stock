import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class indicator_calculator:

    def __init__(self, data_set: pd.DataFrame):
        self.data_set = data_set


    def calculate_rsi(self):
        # 低于 40 就是 超卖， 高于 70 就是 超买，40-70 之间是正常范围
        window: int = 14  # RSI 计算的时间窗口

         # 确保按股票和日期排序
        df = self.data_set.sort_values(by=['ticker_id', 'date']).copy()
        df['rsi'] = None
        df['rsi_score'] = None

        # 按 ticker_id 分组计算 RSI（使用 Wilder 的指数平滑）
        for ticker, group in df.groupby('ticker_id'):
            prices = group['close']
            delta = prices.diff()

            # 分离上涨和下跌部分
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)

            # Wilder 平滑（alpha = 1/window），更符合经典 RSI 定义
            avg_gain = gain.ewm(alpha=1.0/window, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1.0/window, adjust=False).mean()

            # 避免除以 0
            rs = avg_gain / (avg_loss.replace(0, 1e-10))
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(50)  # 起始若无数据时取中性 50

            # 映射到 -100 .. 100
            # 这里用 rsi=50 -> 0；rsi=100 -> +100（强烈超买）；rsi=0 -> -100（强烈超卖）
            rsi_score = 2 * (rsi - 50)
            rsi_score = rsi_score.clip(-100, 100)

            df.loc[group.index, 'rsi'] = rsi
            df.loc[group.index, 'rsi_score'] = rsi_score
        
        return df
    

    
    def calculate_macd(self):

        # MACD: 
        #   macd > 0：短期 EMA 高于长期 EMA，说明短期涨势强于长期 → 潜在上涨趋势
        #   macd < 0：短期 EMA 低于长期 EMA，说明短期跌势强于长期 → 潜在下跌趋势
        #   数值大小：
        #   数值越大，短期涨势相对于长期趋势越强
        #   数值越小（负数越小），短期下跌压力越大

        # signal_line:
        #   当 macd 上穿 signal_line → 潜在买入信号
        #   当 macd 下穿 signal_line → 潜在卖出信号

        # histogram:
        #   histogram > 0 且变大 → 上涨动能增强
        #   histogram > 0 且变小 → 上涨动能减弱
        #   histogram < 0 且变小 → 下跌动能增强
        #   histogram < 0 且变大 → 下跌动能减弱

        # - macd_score > 0 → 买入倾向（多头强）0附近买是最好的
        # - macd_score < 0 → 卖出倾向（空头强）

        fast = 9
        slow = 23
        signal = 6
        eps = 1e-9
        tanh_k = 3.0              # 越小越敏感；常用 2~4
        crossover_boost = 15.0    # 上穿/下穿加减分
        divergence_boost = 20.0   # 若存在背离列时的加减分（可调）

        df = self.data_set.sort_values(by=['ticker_id', 'date']).copy()
        df['macd'] = np.nan
        df['signal_line'] = np.nan
        df['histogram'] = np.nan
        df['macd_score'] = np.nan  # 保证 float dtype

        for ticker, group in df.groupby('ticker_id', sort=False):
            prices = group['close'].astype(float)

            ema_fast = prices.ewm(span=fast, adjust=False).mean()
            ema_slow = prices.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            histogram = macd - signal_line

            # z-score 标准化（基于该 ticker 的 histogram 分布）
            hist_mean = histogram.mean()
            hist_std = histogram.std()
            z = (histogram - hist_mean) / (hist_std + eps)

            # 直接映射到 -100..100（对称）
            macd_score = np.tanh(z / tanh_k) * 100.0

            # 上下穿检测（仅对当天上穿/下穿做一次性加减）
            cross_up = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
            cross_down = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))

            macd_score = macd_score + cross_up.astype(float) * crossover_boost
            macd_score = macd_score - cross_down.astype(float) * crossover_boost

            # 若存在背离列则融合（假设列名为 bullish_divergence / bearish_divergence 且为 bool/0-1）
            if 'bullish_divergence' in df.columns and 'bearish_divergence' in df.columns:
                # 注意：使用 group.index 对齐
                b_div = df.loc[group.index, 'bullish_divergence'].astype(float)
                s_div = df.loc[group.index, 'bearish_divergence'].astype(float)
                # 看涨背离加分，看跌背离扣分
                macd_score = macd_score + b_div * divergence_boost
                macd_score = macd_score - s_div * divergence_boost

            # 限制范围 -100..100
            macd_score = macd_score.clip(-100, 100)

            # 写回（确保为 float）
            df.loc[group.index, 'macd'] = macd.values.astype(float)
            df.loc[group.index, 'signal_line'] = signal_line.values.astype(float)
            df.loc[group.index, 'histogram'] = histogram.values.astype(float)
            df.loc[group.index, 'macd_score'] = macd_score.values.astype(float)

        return df


    def calculate_bollinger_bands(self):

        # 看涨信号（考虑买入）：
        #   价格 < 下轨 → 可能超卖，准备反弹。
        #   价格 > 上轨 + 布林带从“窄”变“宽” → 强势上涨开始。宽窄其实是上轨减去下轨的值。

        # 看跌信号（考虑卖出）：
        #   价格 > 上轨 → 可能超买，准备回调。
        #   价格 < 下轨 + 布林带从“窄”变“宽” → 强势下跌开始。

        # 趋势方向（看中轨）：
        #   价格 > 中轨 → 短期趋势偏多。
        #   价格 < 中轨 → 短期趋势偏空。

        # bb_score = -100 → 价格远低于下轨
        # bb_score = 0 → 价格位于中轨附近
        # bb_score = +100 → 价格远高于上轨

        window: int = 20   # 时间窗口
        num_std: int = 2   # 标准差倍数

        df = self.data_set.sort_values(by=['ticker_id', 'date']).copy()
        df['bb_upper'] = None
        df['bb_middle'] = None
        df['bb_lower'] = None
        df['bb_score'] = None  # 新增列

        for ticker, group in df.groupby('ticker_id'):
            prices = group['close']

            # 中轨、上轨、下轨
            middle_band = prices.rolling(window=window).mean()
            std = prices.rolling(window=window).std()
            upper_band = middle_band + (std * num_std)
            lower_band = middle_band - (std * num_std)

            # 计算布林带百分比（%B）
            bb_percent = (prices - lower_band) / (upper_band - lower_band)

            # 转换为 -100 ~ 100 区间
            bb_score = (bb_percent - 0.5) * 200
            bb_score = bb_score.clip(-100, 100)  # 限制范围

            # 写回
            df.loc[group.index, 'bb_upper'] = upper_band
            df.loc[group.index, 'bb_middle'] = middle_band
            df.loc[group.index, 'bb_lower'] = lower_band
            df.loc[group.index, 'bb_score'] = bb_score

        return df
    


    # 成交量加权平均价
    def calculate_vwap(self):

        # 判断趋势（短线或日内交易）:
        #   价格在 VWAP 之上 → 买方强势，多头市场
        #   价格在 VWAP 之下 → 卖方强势，空头市场

        # 支撑与阻力:
        #   当价格回到 VWAP 附近，VWAP 可能成为支撑或阻力
        #   交易执行参考（机构用）

        #   如果你是大资金买入，目标是尽量低于 VWAP 买入，高于 VWAP 卖出，降低市场冲击成本

        # ≥ +60	明显强于均价 → 买方主导	顺势持有 / 加仓
        # +20 ～ +60	偏多状态	保持多头
        # -20 ～ +20	震荡区间	观望 / 等待信号
        # ≤ -60	明显弱于均价 → 卖方主导	减仓 / 观望

        window: int = 5
        df = self.data_set.sort_values(by=['ticker_id', 'date']).copy()
        df['vwap'] = None
        df['vwap_score'] = None

        eps = 1e-9

        for ticker, group in df.groupby('ticker_id'):
            high = group['high']
            low = group['low']
            close = group['close']
            volume = group['volume']

            typical_price = (high + low + close) / 3
            numerator = (typical_price * volume).rolling(window=window, min_periods=1).sum()
            denominator = volume.rolling(window=window, min_periods=1).sum()
            vwap = numerator / denominator

            # 使用 rolling std 作为尺度（更稳健）
            rolling_std = close.rolling(window=window, min_periods=1).std().fillna(0)
            # 防止 std 为 0
            rolling_std = rolling_std.replace(0, eps)

            z = (close - vwap) / rolling_std

            # 使用 arctan 将 z 映射到 (-100, 100)（平滑且无爆炸）
            vwap_score = np.arctan(z) * (200.0 / np.pi)

            # 可选：再 clip 一次以防万一
            vwap_score = vwap_score.clip(-100, 100)

            df.loc[group.index, 'vwap'] = vwap
            df.loc[group.index, 'vwap_score'] = vwap_score


        return df



    def calculate_stochastic(self):
        """
        计算随机指标（Stochastic Oscillator）

        用法说明：
            %K：衡量收盘价相对于近期价格区间的位置。14 天内的最高价和最低价。
            %D：%K 的平滑移动平均线。3 天的简单移动平均。

        交易信号逻辑：
            - 当 %K 上穿 %D → 买入信号（价格可能转强）
            - 当 %K 下穿 %D → 卖出信号（价格可能转弱）
            - %K 或 %D 高于 80 → 超买区间
            - %K 或 %D 低于 20 → 超卖区间

        参数：
            k_window: %K 计算窗口长度（默认14）Fast stochastic line 衡量当前价格在最近一段区间内的位置
            d_window: %D 平滑窗口长度（默认3）Slow stochastic line %K的移动平均，更稳定、滞后性更强

            ≥ +80	超买区间	可能回调 / 卖出参考
            +20 ～ +80	偏强	多头趋势
            -20 ～ +20	区间震荡	观望
            ≤ -80	超卖区间	可能反弹 / 买入参考

        """
        k_window: int = 14
        d_window: int = 3

        df = self.data_set.sort_values(by=['ticker_id', 'date']).copy()
        df['%K'] = None
        df['%D'] = None
        df['stoch_score'] = None  # 新增列

        for ticker, group in df.groupby('ticker_id'):
            high = group['high']
            low = group['low']
            close = group['close']

            lowest_low = low.rolling(window=k_window, min_periods=1).min()
            highest_high = high.rolling(window=k_window, min_periods=1).max()

            # 避免分母为零
            denom = (highest_high - lowest_low).replace(0, 1e-9)

            k_percent = 100 * (close - lowest_low) / denom
            d_percent = k_percent.rolling(window=d_window, min_periods=1).mean()

            # 映射到 -100 ~ 100
            stoch_score = (k_percent - 50) * 2
            stoch_score = stoch_score.clip(-100, 100)

            df.loc[group.index, '%K'] = k_percent
            df.loc[group.index, '%D'] = d_percent
            df.loc[group.index, 'stoch_score'] = stoch_score

        return df
    

    
    def calculate_sma_crossover(self):
        """
        计算简单移动平均线交叉策略（Simple Moving Average Crossover）

        用法说明：
            快速均线（fast SMA）：对价格变化反应较快，通常用于捕捉短期趋势。
            慢速均线（slow SMA）：对价格变化反应较慢，通常用于判断长期趋势。

        交易信号逻辑：
            - 当 快速均线 上穿 慢速均线 → 买入信号（趋势可能转强）
            - 当 快速均线 下穿 慢速均线 → 卖出信号（趋势可能转弱）
            - 两条均线的交叉点可视为趋势反转信号。

        参数：
            fast_window: 快速均线的时间窗口长度（默认 5）
            slow_window: 慢速均线的时间窗口长度（默认 20）

        返回：
            DataFrame，包含以下列：
                - sma_fast：快速均线
                - sma_slow：慢速均线
                - crossover_signal：当快速均线高于慢速均线时为 True，否则为 False

        - sma_score：SMA 指标评分（-100 到 100）            
            -100 → sma_fast 最低（相对于本组历史），说明短期均线远低于长期均线 → 下行压力大。
            0 → sma_fast 位于本组历史差值中点。
            100 → sma_fast 最高，说明短期均线上穿长期均线很明显 → 强势上涨。

        """
        fast_window: int = 5
        slow_window: int = 20

        df = self.data_set.sort_values(by=['ticker_id', 'date']).copy()
        df['sma_fast'] = None
        df['sma_slow'] = None
        df['crossover_signal'] = None
        df['sma_score'] = None  # 新增列

        scaler = MinMaxScaler(feature_range=(-100, 100))

        for ticker, group in df.groupby('ticker_id'):
            close = group['close']
            sma_fast = close.rolling(window=fast_window, min_periods=1).mean()
            sma_slow = close.rolling(window=slow_window, min_periods=1).mean()
            crossover_signal = sma_fast > sma_slow

            diff = (sma_fast - sma_slow).fillna(0).values.reshape(-1,1)
            sma_score = scaler.fit_transform(diff).flatten()

            df.loc[group.index, 'sma_fast'] = sma_fast
            df.loc[group.index, 'sma_slow'] = sma_slow
            df.loc[group.index, 'crossover_signal'] = crossover_signal
            df.loc[group.index, 'sma_score'] = sma_score

        return df
    



    def calculate_obv(self):
        """
        计算能量潮（OBV）并检测背离（向量化 & 可调 lookback）

        参数:
            lookback: 用于检测“更低低点/更高高点”的历史窗口（以天为单位）

        返回:
            DataFrame，包含列:
                - obv: 累计 OBV 值
                - obv_signal: 当日 OBV 是否上升（True/False）
                    OBV 上升 → 买盘强劲，资金流入 → 股价可能上涨。
                    OBV 下降 → 卖盘强劲，资金流出 → 股价可能下跌。

                - bullish_divergence: 价格形成更低低点但 OBV 形成更高低点（True/False）正向背离：股价创新低，但 OBV 却创新高 → 潜在买入信号。
                - bearish_divergence: 价格形成更高高点但 OBV 形成更低高点（True/False）负向背离：股价创新高，但 OBV 下跌 → 潜在卖出信号。

        - obv_score: OBV 指标评分（-100 到 100）
            +50 ~ +100	买盘强势，资金流入	多头参考 / 顺势持仓
            +0 ~ +50	买盘略强	偏多趋势
            -50 ~ 0	卖盘略强	偏空趋势
            -100 ~ -50	卖盘强势，资金流出	空头参考 / 减仓
        """

        lookback: int = 5
        df = self.data_set.sort_values(by=['ticker_id', 'date']).copy()
        df['obv'] = np.nan
        df['obv_signal'] = False
        df['bullish_divergence'] = False
        df['bearish_divergence'] = False
        df['obv_score'] = np.nan  # 新增列

        eps = 1e-9

        for ticker, group in df.groupby('ticker_id', sort=False):
            close = group['close'].astype(float)
            volume = group['volume'].astype(float)

            price_diff = close.diff().fillna(0)
            sign = np.sign(price_diff)
            obv_series = (sign * volume).cumsum()
            obv_series.iloc[0] = 0

            obv_signal = obv_series.diff() > 0

            prev_close_min = close.shift(1).rolling(window=lookback, min_periods=1).min()
            prev_close_max = close.shift(1).rolling(window=lookback, min_periods=1).max()
            prev_obv_min = obv_series.shift(1).rolling(window=lookback, min_periods=1).min()
            prev_obv_max = obv_series.shift(1).rolling(window=lookback, min_periods=1).max()

            bullish_div = (close < prev_close_min) & (obv_series > prev_obv_min)
            bearish_div = (close > prev_close_max) & (obv_series < prev_obv_max)

            # OBV score: 用 diff 做 z-score 映射，再 arctan 映射到 -100 ~ 100
            rolling_std = obv_series.diff().rolling(window=lookback, min_periods=1).std().fillna(eps).replace(0, eps)
            z = obv_series.diff() / rolling_std
            obv_score = np.arctan(z) * (200.0 / np.pi)
            obv_score = obv_score.clip(-100, 100)

            df.loc[group.index, 'obv'] = obv_series.values
            df.loc[group.index, 'obv_signal'] = obv_signal.values
            df.loc[group.index, 'bullish_divergence'] = bullish_div.values
            df.loc[group.index, 'bearish_divergence'] = bearish_div.values
            df.loc[group.index, 'obv_score'] = obv_score.values

        return df
    

    
    def calculate_mfi(self):
        """
                计算资金流量指标（MFI, Money Flow Index）并检测背离（向量化 & 可调 lookback）

                参数:
                    lookback: 用于计算 MFI 的周期（默认 14）
                            用于检测背离的窗口同样采用该周期。

                返回:
                    DataFrame，包含列:
                        - mfi: 资金流量指标（0~100）
                        - mfi_signal: MFI 是否上升（True/False）
                            MFI 上升 → 买盘强劲 → 股价可能上涨；
                            MFI 下降 → 卖盘强劲 → 股价可能下跌。
                        - bullish_divergence: 价格创新低但 MFI 形成更高低点 → 潜在买入信号。
                        - bearish_divergence: 价格创新高但 MFI 形成更低高点 → 潜在卖出信号。
        """

        lookback: int = 14
        df = self.data_set.sort_values(by=['ticker_id', 'date']).copy()
        df['mfi'] = np.nan
        df['mfi_signal'] = False
        df['bullish_divergence'] = False
        df['bearish_divergence'] = False

        for ticker, group in df.groupby('ticker_id', sort=False):
            high = group['high'].astype(float)
            low = group['low'].astype(float)
            close = group['close'].astype(float)
            volume = group['volume'].astype(float)

            # 1️⃣ Typical Price
            typical_price = (high + low + close) / 3

            # 2️⃣ Raw Money Flow
            money_flow = typical_price * volume

            # 3️⃣ 判断正/负资金流
            tp_diff = typical_price.diff()
            positive_flow = np.where(tp_diff > 0, money_flow, 0)
            negative_flow = np.where(tp_diff < 0, money_flow, 0)

            # 注意：这里必须指定 index
            pos_mf_sum = pd.Series(positive_flow, index=group.index).rolling(window=lookback, min_periods=1).sum()
            neg_mf_sum = pd.Series(negative_flow, index=group.index).rolling(window=lookback, min_periods=1).sum()

            # 4️⃣ 计算 Money Flow Ratio & MFI
            money_flow_ratio = pos_mf_sum / neg_mf_sum.replace(0, np.nan)
            mfi = 100 - (100 / (1 + money_flow_ratio))
            mfi = mfi.fillna(50)

            # 5️⃣ MFI 上升信号
            mfi_signal = mfi.diff() > 0

            # 6️⃣ 背离检测
            prev_close_min = close.shift(1).rolling(window=lookback, min_periods=1).min()
            prev_close_max = close.shift(1).rolling(window=lookback, min_periods=1).max()
            prev_mfi_min = mfi.shift(1).rolling(window=lookback, min_periods=1).min()
            prev_mfi_max = mfi.shift(1).rolling(window=lookback, min_periods=1).max()

            bullish_div = (close < prev_close_min) & (mfi > prev_mfi_min)
            bearish_div = (close > prev_close_max) & (mfi < prev_mfi_max)

            # 7️⃣ 基础 MFI score (-100~100)
            base_score = (mfi - 50) / 50 * 100
            # 强化趋势
            base_score = base_score * (1 + 0.5 * mfi_signal.astype(float))

            # 8️⃣ 融合背离信号
            # 背离加权：看涨背离 +20%，看跌背离 -20%（可调）
            bullish_weight = 0.2
            bearish_weight = 0.2
            score = base_score.copy()
            score += bullish_div.astype(float) * 100 * bullish_weight
            score -= bearish_div.astype(float) * 100 * bearish_weight
            # 将 score 限制在 -100 到 100 之间
            score = score.clip(-100, 100)


            # ✅ 写回主 DataFrame
            df.loc[group.index, 'mfi'] = mfi
            df.loc[group.index, 'mfi_signal'] = mfi_signal
            df.loc[group.index, 'mfi_bullish_divergence'] = bullish_div
            df.loc[group.index, 'mfi_bearish_divergence'] = bearish_div
            df.loc[group.index, 'mfi_score'] = score # 在0左右可以加仓

        return df
    
    
    def comprehensive_indicator(self):
        """
        计算综合技术指标评分（Technical Indicator Score）

        返回：
            DataFrame，包含原始数据及各指标评分，以及综合评分列 'technical_indicator_score'
        """

        # 调用各指标方法，直接保存完整 DataFrame
        rsi_df = self.calculate_rsi()
        macd_df = self.calculate_macd()
        bb_df = self.calculate_bollinger_bands()
        vwap_df = self.calculate_vwap()
        stoch_df = self.calculate_stochastic()
        sma_df = self.calculate_sma_crossover()
        obv_df = self.calculate_obv()
        mfi_df = self.calculate_mfi()

        # 合并所有评分
        df = self.data_set.sort_values(by=['ticker_id', 'date']).copy()
        score_dfs = [rsi_df, macd_df, bb_df, vwap_df, stoch_df, sma_df, obv_df, mfi_df]

        for score_df in score_dfs:
            score_cols = [col for col in score_df.columns if col.endswith('_score')]
            df = df.merge(score_df[['ticker_id', 'date'] + score_cols], on=['ticker_id', 'date'], how='left')

        # 权重列表，明确写出
        weights = {
            'rsi_score': 0.8,
            'stoch_score': 0.8,
            'mfi_score': 1.0,   # 平衡指标稍高
            'macd_score': 1.2,
            'sma_score': 1.2,
            'bb_score': 1.0,
            'vwap_score': 1.0,
            'obv_score': 0.8
        }

        # 计算加权平均
        total_weight = sum(weights.values())
        df['comprehensive_score'] = sum(df[ind] * w for ind, w in weights.items()) / total_weight
        # 限制在 -100 ~ 100
        df['comprehensive_score'] = df['comprehensive_score'].clip(-100, 100)

        # 计算平滑分数（3日、5日、10日、15日、20日）
        for window in [3, 5, 10, 15, 20]:
            df[f'comprehensive_score_{window}_days'] = df.groupby('ticker_id')['comprehensive_score'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

        market_daily = df.groupby('date')['comprehensive_score'].mean()   # 每天所有股票的平均值
        df['market_avg_comprehensive_today'] = df['date'].map(market_daily)

        for window in [3, 5, 10, 15, 20]:
            market_rolling = market_daily.rolling(window=window, min_periods=1).mean()
            df[f'market_avg_comprehensive_{window}_days'] = df['date'].map(market_rolling)

        return df


        




