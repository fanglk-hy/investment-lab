"""
标普500与沪深300历史相关性分析
==============================
- 标普500: 东方财富 API（带指数退避重试）
- 沪深300: 腾讯数据源（akshare）
- 计算相关系数，绘制60天滚动相关图
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import akshare as ak

# ============================================================
# 0. 网络配置 — 绕过不稳定的系统代理
# ============================================================
session = requests.Session()
session.trust_env = False

# ============================================================
# 1. 下载标普500（东方财富 API，加重试）
# ============================================================
def fetch_sp500_eastmoney(max_retries=8):
    """
    从东方财富获取标普500历史日线，带指数退避重试。
    网络环境不稳定时，每次重试等待时间翻倍。
    """
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": "100.SPX",
        "klt": "101",
        "fqt": "1",
        "lmt": "50000",
        "end": "20500000",
        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "f057cbcbce2a86e2866ab8877db1d059",
        "forcect": "1",
    }

    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=20)
            data = resp.json()
            if data.get("rc") != 0:
                raise RuntimeError(f"API rc={data.get('rc')}")
            klines = data["data"]["klines"]
            if not klines:
                raise RuntimeError("空数据")

            rows = []
            for line in klines:
                parts = line.split(",")
                rows.append({
                    "日期": pd.to_datetime(parts[0]),
                    "SP500": float(parts[2]),
                })
            df = pd.DataFrame(rows).set_index("日期").sort_index()
            print(f"   ✓ 标普500: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}, {len(df)} 行")
            return df

        except Exception as e:
            wait = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s...
            print(f"   ⚠ 尝试 {attempt+1}/{max_retries} 失败 ({e}), 等待{wait}s...")
            time.sleep(wait)

    raise RuntimeError("标普500数据下载失败，已达最大重试次数")

# ============================================================
# 2. 下载沪深300（腾讯数据源，已验证可用）
# ============================================================
def fetch_csi300_tencent():
    """使用腾讯数据源获取沪深300历史日线（sz399300）"""
    print("   正在下载沪深300（腾讯数据源）...")
    df = ak.stock_zh_index_daily_tx(symbol="sz399300")
    df = df.rename(columns={"date": "日期", "close": "CSI300"})
    df["日期"] = pd.to_datetime(df["日期"])
    df = df[["日期", "CSI300"]].set_index("日期").sort_index()
    print(f"   ✓ 沪深300: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}, {len(df)} 行")
    return df

# ============================================================
# 主流程
# ============================================================
print("=" * 60)
print("标普500 vs 沪深300 相关性分析")
print("=" * 60)

print("\n1. 下载标普500...")
sp500 = fetch_sp500_eastmoney()

print("\n2. 下载沪深300...")
csi300 = fetch_csi300_tencent()

# ============================================================
# 3. 截取过去10年并对齐交易日
# ============================================================
print("\n3. 数据处理...")
start_date = "2016-05-01"
end_date   = "2026-05-01"

sp500  = sp500.loc[start_date:end_date]
csi300 = csi300.loc[start_date:end_date]
common = sp500.join(csi300, how="inner").dropna()

print(f"   10年共同交易日: {len(common)} 天")
print(f"   日期范围: {common.index[0].strftime('%Y-%m-%d')} → {common.index[-1].strftime('%Y-%m-%d')}")

# ============================================================
# 4. 计算相关系数
# ============================================================
returns = np.log(common / common.shift(1)).dropna()
overall_corr = returns["SP500"].corr(returns["CSI300"])
print(f"\n   10年整体日收益率相关系数: {overall_corr:.4f}")

returns_y = returns.copy()
returns_y["year"] = returns_y.index.year
print("\n   各年度相关系数:")
for yr in sorted(returns_y["year"].unique()):
    grp = returns_y[returns_y["year"] == yr]
    corr = grp["SP500"].corr(grp["CSI300"])
    bar = "█" * max(1, int(abs(corr) * 20))
    print(f"   {yr}: {corr:+.4f} {bar}")

window = 60
rolling_corr = returns["SP500"].rolling(window=window).corr(returns["CSI300"])

print(f"\n   滚动{window}天相关系数统计:")
print(f"   均值:      {rolling_corr.mean():.4f}")
print(f"   中位数:    {rolling_corr.median():.4f}")
print(f"   最大值:    {rolling_corr.max():.4f}  ({rolling_corr.idxmax().strftime('%Y-%m-%d')})")
print(f"   最小值:    {rolling_corr.min():.4f}  ({rolling_corr.idxmin().strftime('%Y-%m-%d')})")
print(f"   最新值:    {rolling_corr.iloc[-1]:.4f}  ({rolling_corr.index[-1].strftime('%Y-%m-%d')})")
print(f"   正值占比:  {(rolling_corr > 0).mean() * 100:.1f}%")

# ============================================================
# 5. 画图
# ============================================================
print("\n4. 绘制图表...")

plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# ---- 子图1: 归一化价格走势 ----
ax1 = axes[0]
norm = common / common.iloc[0] * 100
ax1.plot(norm.index, norm["SP500"],  lw=1.5, color="#1f77b4", label="标普500")
ax1.plot(norm.index, norm["CSI300"], lw=1.5, color="#d62728", label="沪深300")
ax1.set_title("标普500 vs 沪深300 — 过去10年归一化价格走势", fontsize=14, fontweight="bold")
ax1.set_ylabel("归一化价格 (2016-05=100)")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# ---- 子图2: 滚动60天相关系数 ----
ax2 = axes[1]
ax2.plot(rolling_corr.index, rolling_corr.values, lw=1.0, color="#2ca02c",
         label=f"滚动{window}天相关系数")
ax2.axhline(y=overall_corr, color="black", ls="--", lw=1.2,
            label=f"10年整体相关系数: {overall_corr:.3f}")
ax2.axhline(y=0, color="gray", ls=":", lw=0.8)

pos_mask = rolling_corr.values > 0
ax2.fill_between(rolling_corr.index, 0, rolling_corr.values,
                 where=pos_mask,  color="#2ca02c", alpha=0.12)
ax2.fill_between(rolling_corr.index, 0, rolling_corr.values,
                 where=~pos_mask, color="#d62728", alpha=0.12)

# 标注最新值
lx, ly = rolling_corr.index[-1], rolling_corr.iloc[-1]
ax2.annotate(f"最新: {ly:.3f}",
             xy=(lx, ly), xytext=(15, 15), textcoords="offset points",
             fontsize=10, fontweight="bold",
             arrowprops=dict(arrowstyle="->", lw=1.2))

ax2.set_title(f"标普500 vs 沪深300 — 滚动{window}天日收益率相关系数", fontsize=14, fontweight="bold")
ax2.set_ylabel("皮尔逊相关系数")
ax2.set_xlabel("日期")
ax2.legend(loc="upper left")
ax2.set_ylim(-0.6, 0.8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_path = "../research/sp500_csi300_correlation.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"   ✓ 图表已保存: research/sp500_csi300_correlation.png")

# ============================================================
# 6. 补充统计
# ============================================================
print("\n" + "=" * 60)
print("补充: 年度总收益率 & 累计收益")
print("=" * 60)

annual_ret = common.resample("YE").last().pct_change()
for yr in sorted(returns_y["year"].unique()):
    yr_mask = annual_ret.index.year == yr
    if yr_mask.any():
        row = annual_ret[yr_mask]
        sp_r = row["SP500"].values[0]
        cs_r = row["CSI300"].values[0]
        print(f"   {yr}:  标普500 {sp_r:+.2%}  |  沪深300 {cs_r:+.2%}")

cum = (common.iloc[-1] / common.iloc[0]) - 1
print(f"\n   10年累计总收益:")
print(f"   标普500: {cum['SP500']:+.2%}")
print(f"   沪深300: {cum['CSI300']:+.2%}")

print("\n分析完成。")
