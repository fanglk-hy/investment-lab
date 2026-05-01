"""
全球半导体 ETF 绩效对比分析
============================
选取 5 只规模最大、流动性最好的半导体 ETF（3 美股 + 2 A股），
下载近三年净值数据，计算夏普比率与最大回撤，生成对比图表。

数据源:
  - 美股 ETF: Sina 财经 (akshare stock_us_daily)
  - A股 ETF: Sina 财经 (akshare fund_etf_hist_sina)
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ============================================================
# 0. 参数设置
# ============================================================
START_DATE = "2023-05-01"
END_DATE   = "2026-05-01"

# 无风险利率（年化）
# 美股：美联储利率在 4-5.5% 区间，取中间值 4.5%
# A股：中国 10Y 国债 ~2.5%
RISK_FREE_US = 0.045
RISK_FREE_CN = 0.025

# 5 只 ETF 定义
ETF_LIST = [
    {"name": "SMH",     "label": "SMH (VanEck半导体)",       "market": "US", "source": "sina"},
    {"name": "SOXX",    "label": "SOXX (iShares半导体)",      "market": "US", "source": "sina"},
    {"name": "XSD",     "label": "XSD (SPDR等权半导体)",       "market": "US", "source": "sina"},
    {"name": "sh588200","label": "科创芯片ETF (588200)",       "market": "CN", "source": "sina_fund"},
    {"name": "sz159995","label": "芯片ETF华夏 (159995)",       "market": "CN", "source": "sina_fund"},
]

TRADING_DAYS = 252  # 年化交易日数

# ============================================================
# 1. 数据下载
# ============================================================
print("=" * 64)
print("全球半导体 ETF 绩效对比分析")
print("=" * 64)

def fetch_us_etf(symbol, label):
    """从 Sina 财经获取美股 ETF 日线数据"""
    print(f"   下载 {label} (Sina)...", end=" ")
    raw = ak.stock_us_daily(symbol=symbol, adjust="qfq")
    df = pd.DataFrame({
        "date":  pd.to_datetime(raw["date"]),
        "close": raw["close"].astype(float),
    })
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    df = df.set_index("date").sort_index()
    print(f"{len(df)} 行, {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
    return df["close"].rename(label)

def fetch_cn_etf(symbol, label):
    """从 Sina 财经获取 A 股 ETF 日线数据"""
    print(f"   下载 {label} (Sina)...", end=" ")
    raw = ak.fund_etf_hist_sina(symbol=symbol)
    df = pd.DataFrame({
        "date":  pd.to_datetime(raw["date"]),
        "close": raw["close"].astype(float),
    })
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    df = df.set_index("date").sort_index()
    print(f"{len(df)} 行, {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
    return df["close"].rename(label)

print("\n1. 下载数据...")
nav_data = {}
for etf in ETF_LIST:
    if etf["source"] == "sina":
        nav_data[etf["label"]] = fetch_us_etf(etf["name"], etf["label"])
    else:
        nav_data[etf["label"]] = fetch_cn_etf(etf["name"], etf["label"])

# ============================================================
# 2. 合并数据 & 计算指标
# ============================================================
print("\n2. 计算绩效指标...")

# 对齐所有 ETF 到同一日期范围，缺失值前向填充
df_all = pd.concat(nav_data.values(), axis=1)
df_all = df_all.ffill().dropna()
# 对齐到每个 ETF 都存在的日期（取各 ETF 起始日期中最早的那个）
df_all = df_all[df_all.index >= df_all.apply(lambda col: col.first_valid_index()).max()]

print(f"   共同分析区间: {df_all.index[0].strftime('%Y-%m-%d')} → {df_all.index[-1].strftime('%Y-%m-%d')}")
print(f"   有效交易日: {len(df_all)} 天")

# 日对数收益率
returns = np.log(df_all / df_all.shift(1)).dropna()

# ---- 计算各指标 ----
def calc_metrics(ret_series, rf_annual, trading_days=TRADING_DAYS):
    """
    计算年化收益率、波动率、夏普比率、最大回撤、Calmar 比率
    """
    n = len(ret_series)
    rf_daily = rf_annual / trading_days

    # 年化收益率
    ann_return = ret_series.mean() * trading_days
    # 年化波动率
    ann_vol = ret_series.std() * np.sqrt(trading_days)
    # 夏普比率
    excess = ret_series - rf_daily
    sharpe = (excess.mean() / excess.std()) * np.sqrt(trading_days) if excess.std() > 0 else 0

    # 最大回撤（基于累计收益率）
    cum = (1 + ret_series).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()

    # Calmar 比率 = 年化收益 / |最大回撤|
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    return {
        "年化收益率":   ann_return,
        "年化波动率":   ann_vol,
        "夏普比率":     sharpe,
        "最大回撤":     max_dd,
        "最大回撤日期": max_dd_date,
        "Calmar比率":   calmar,
    }

metrics = {}
for etf in ETF_LIST:
    label = etf["label"]
    rf = RISK_FREE_US if etf["market"] == "US" else RISK_FREE_CN
    metrics[label] = calc_metrics(returns[label], rf)
    m = metrics[label]
    print(f"\n   {label}")
    print(f"   年化收益率: {m['年化收益率']:+.2%}")
    print(f"   年化波动率: {m['年化波动率']:.2%}")
    print(f"   夏普比率:   {m['夏普比率']:.2f}")
    print(f"   最大回撤:   {m['最大回撤']:+.2%}  ({m['最大回撤日期'].strftime('%Y-%m-%d')})")
    print(f"   Calmar比率: {m['Calmar比率']:.2f}")

# ============================================================
# 3. 画图
# ============================================================
print("\n3. 绘制对比图表...")

plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

# 颜色方案
colors = {
    "SMH (VanEck半导体)":     "#1f77b4",
    "SOXX (iShares半导体)":    "#ff7f0e",
    "XSD (SPDR等权半导体)":     "#2ca02c",
    "科创芯片ETF (588200)":     "#d62728",
    "芯片ETF华夏 (159995)":     "#9467bd",
}

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# ---- 子图 1: 累计收益率 ----
ax1 = axes[0, 0]
for label in df_all.columns:
    cum = df_all[label] / df_all[label].iloc[0] - 1
    ax1.plot(cum.index, cum * 100, color=colors[label], lw=1.5, label=label)
ax1.set_title("累计收益率走势 (2023.05 — 2026.05)", fontsize=13, fontweight="bold")
ax1.set_ylabel("累计收益率 (%)")
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

# ---- 子图 2: 最大回撤曲线 ----
ax2 = axes[0, 1]
for label in df_all.columns:
    cum = (1 + returns[label]).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max * 100
    ax2.fill_between(dd.index, 0, dd.values, color=colors[label], alpha=0.12)
    ax2.plot(dd.index, dd.values, color=colors[label], lw=0.8, label=label)
ax2.set_title("回撤曲线", fontsize=13, fontweight="bold")
ax2.set_ylabel("回撤 (%)")
ax2.legend(fontsize=8, loc="lower left")
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

# ---- 子图 3: 风险-收益散点图 (气泡图) ----
ax3 = axes[1, 0]
for label in df_all.columns:
    m = metrics[label]
    ax3.scatter(m["年化波动率"] * 100, m["年化收益率"] * 100,
                s=abs(m["夏普比率"]) * 120, color=colors[label],
                alpha=0.8, edgecolors="black", linewidth=0.5, zorder=5)
    ax3.annotate(label.split("(")[0].strip(),
                 (m["年化波动率"] * 100, m["年化收益率"] * 100),
                 fontsize=8, ha="center", va="bottom",
                 xytext=(0, 8), textcoords="offset points")

# 画等夏普比率线
vol_range = np.linspace(15, 50, 50)
for sr in [0.5, 1.0, 1.5]:
    ret_line = sr * vol_range + RISK_FREE_US * 100  # 近似
    ax3.plot(vol_range, ret_line, "--", color="gray", alpha=0.3, lw=0.8)
    ax3.annotate(f"SR={sr}", (vol_range[-1], ret_line[-1]),
                 fontsize=7, color="gray", alpha=0.6)

ax3.set_title("风险-收益散点图 (气泡大小 ∝ 夏普比率)", fontsize=13, fontweight="bold")
ax3.set_xlabel("年化波动率 (%)")
ax3.set_ylabel("年化收益率 (%)")
ax3.grid(True, alpha=0.3)

# ---- 子图 4: 绩效指标横向柱状对比 ----
ax4 = axes[1, 1]
labels_short = [lbl.split("(")[0].strip() for lbl in df_all.columns]
x = np.arange(len(labels_short))
width = 0.25

rets  = [metrics[lbl]["年化收益率"] * 100 for lbl in df_all.columns]
vols  = [metrics[lbl]["年化波动率"] * 100 for lbl in df_all.columns]
sharpes = [metrics[lbl]["夏普比率"] for lbl in df_all.columns]

bars1 = ax4.bar(x - width, rets, width, label="年化收益率 (%)", color="#1f77b4", alpha=0.85)
bars2 = ax4.bar(x,        vols, width, label="年化波动率 (%)", color="#d62728", alpha=0.85)

# 在柱子上标注夏普比率
for i, (sr, ret) in enumerate(zip(sharpes, rets)):
    ax4.text(i + width, ret + 0.5, f"SR={sr:.2f}", ha="center", fontsize=8, fontweight="bold")

ax4.set_title("收益 / 波动 / 夏普比率对比", fontsize=13, fontweight="bold")
ax4.set_xticks(x)
ax4.set_xticklabels(labels_short, fontsize=8)
ax4.legend(fontsize=9, loc="upper right")
ax4.axhline(y=0, color="black", lw=0.5)
ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
output_path = "../research/semiconductor_etf_comparison.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"   ✓ 图表已保存: research/semiconductor_etf_comparison.png")

# ============================================================
# 4. 汇总排名表
# ============================================================
print("\n" + "=" * 64)
print("综合排名汇总 (按夏普比率排序)")
print("=" * 64)

sorted_etfs = sorted(metrics.items(), key=lambda x: x[1]["夏普比率"], reverse=True)

print(f"  {'排名':<4} {'ETF':<28} {'年化收益':>8} {'波动率':>8} {'夏普':>6} {'最大回撤':>8} {'Calmar':>6}")
print("  " + "-" * 78)
for rank, (label, m) in enumerate(sorted_etfs, 1):
    print(f"  {rank:<4} {label:<28} {m['年化收益率']:>+7.2%} {m['年化波动率']:>7.2%} "
          f"{m['夏普比率']:>6.2f} {m['最大回撤']:>+7.2%} {m['Calmar比率']:>6.2f}")

print("\n分析完成。")
print(f"注: 美股ETF使用 {RISK_FREE_US:.1%} 无风险利率, A股ETF使用 {RISK_FREE_CN:.1%}")
