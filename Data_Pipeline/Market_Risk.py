import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# ==============================
# 1. 下载数据（你的原代码）
# ==============================
end = datetime.date.today()
start = end - datetime.timedelta(days=365)

tickers = "^VIX ^VIX3M HYG LQD ^TNX SPY RSP EWI IWN"
df = yf.download(
    tickers,
    start=start,
    end=end + datetime.timedelta(days=1),
    auto_adjust=True,
    progress=False
)

df.to_csv("market_risk_data.csv")


# ==============================
# 2. 提取收盘价（兼容多重列）
# ==============================
if isinstance(df.columns, pd.MultiIndex):
    if 'Adj Close' in df.columns.levels[0]:
        close_df = df['Adj Close'].copy()
    elif 'Close' in df.columns.levels[0]:
        close_df = df['Close'].copy()
    else:
        raise ValueError("No 'Close' or 'Adj Close' level found.")
else:
    close_df = df.copy()

wanted = ["^VIX", "^VIX3M", "HYG", "LQD", "^TNX", "SPY", "RSP", "EWI", "IWN"]
close_df = close_df[[c for c in close_df.columns if c in wanted]]


# ==============================
# 3. 工具函数：series → 0~100 分数
# ==============================
def to_score_0_100(series: pd.Series, invert: bool = False) -> pd.Series:
    """
    把一个时间序列映射到 0~100：
    - 用 rank(pct=True) 得到 0~1 的百分位
    - invert=True 时用 (1 - pct) 反向（适合 VIX 这类 risk-off）
    - 再乘以 100 映射到 0~100
    """
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return pd.Series(index=series.index, dtype=float)

    pct = s.rank(pct=True)  # 0~1
    if invert:
        pct = 1 - pct

    score = pct * 100
    return score.reindex(series.index)


# ==============================
# 4. 构造 7 个 CNN 风格因子
# ==============================

fg_df = pd.DataFrame(index=close_df.index)

# 1) 股市动量：SPY vs 125 日均线
spy = close_df["SPY"]
spy_ma125 = spy.rolling(window=125, min_periods=60).mean()
momentum_raw = spy / spy_ma125 - 1
fg_df["momentum_score"] = to_score_0_100(momentum_raw, invert=False)

# 2) 广度近似：RSP / SPY（等权 vs 市值权重）
rsp = close_df["RSP"]
breadth_us_raw = rsp / spy - 1
fg_df["breadth_us_score"] = to_score_0_100(breadth_us_raw, invert=False)

# 3) 小盘 vs 大盘：IWN / SPY
iwn = close_df["IWN"]
smallcap_raw = iwn / spy - 1
fg_df["smallcap_score"] = to_score_0_100(smallcap_raw, invert=False)

# 4) 波动率期限结构：VIX / VIX3M（越高越恐惧 → risk-off）
vix = close_df["^VIX"]
vix3m = close_df["^VIX3M"]
vix_term_raw = vix / vix3m
fg_df["vix_term_structure_score"] = to_score_0_100(vix_term_raw, invert=True)

# 5) 信用利差：HYG / LQD（垃圾债相对 IG 债表现）
hyg = close_df["HYG"]
lqd = close_df["LQD"]
credit_spread_raw = hyg / lqd - 1
fg_df["credit_spread_score"] = to_score_0_100(credit_spread_raw, invert=False)

# 6) 波动率水平：VIX vs 50 日均线（越高越恐惧 → risk-off）
vix_ma50 = vix.rolling(window=50, min_periods=25).mean()
vix_level_raw = vix / vix_ma50 - 1
fg_df["vix_level_score"] = to_score_0_100(vix_level_raw, invert=True)

# 7) 避险需求：SPY / LQD（股票 vs 债券）
safe_haven_raw = spy / lqd - 1
fg_df["safe_haven_score"] = to_score_0_100(safe_haven_raw, invert=False)


# ==============================
# 5. 组合成 “Fear & Greed” 总分
# ==============================
# CNN 是 0~100，我们也用 0~100
fg_df["fear_greed_score"] = fg_df.mean(axis=1, skipna=True)

# 方便调试：看最新一行各因子 & 总分
print(fg_df.tail(3))


# ==============================
# 6. 画图：CNN 风格 Fear & Greed 曲线
# ==============================
plt.figure(figsize=(14, 6))

plt.plot(fg_df.index, fg_df["fear_greed_score"], label="CNN-style Fear & Greed (0-100)")

# 通常 0-25 极度恐惧，75-100 极度贪婪（仿 CNN）
plt.axhline(25, linestyle="--", linewidth=1)
plt.axhline(75, linestyle="--", linewidth=1)

plt.title("CNN-style Fear & Greed Index (Replica, 0-100)")
plt.xlabel("Date")
plt.ylabel("Score")
plt.legend()

plt.show()

# 如果你还想保留原来的 -100~100 Market Risk Score，可以另外算一列：
# fg_df["market_risk_score_-100_100"] = fg_df["fear_greed_score"] * 2 - 100
# fg_df.to_csv("cnn_like_fear_greed_scores.csv")
