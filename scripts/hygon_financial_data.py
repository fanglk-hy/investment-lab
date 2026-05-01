"""
海光信息（688041）财务数据获取脚本
获取历史财务数据、估值数据，用于深度分析
"""
import akshare as ak
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# 输出目录
output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(output_dir, exist_ok=True)

stock_code = "688041"
stock_name = "海光信息"

print(f"正在获取 {stock_name}({stock_code}) 的财务数据...\n")

# ============================================================
# 1. 获取利润表数据
# ============================================================
print("1. 获取利润表数据...")
try:
    income_stmt = ak.stock_financial_report_sina(
        stock=stock_code,
        symbol="利润表"
    )
    print(f"   利润表获取成功，共 {len(income_stmt)} 条记录")
except Exception as e:
    print(f"   利润表获取失败: {e}")
    income_stmt = None

# ============================================================
# 2. 获取资产负债表数据
# ============================================================
print("2. 获取资产负债表数据...")
try:
    balance_sheet = ak.stock_financial_report_sina(
        stock=stock_code,
        symbol="资产负债表"
    )
    print(f"   资产负债表获取成功，共 {len(balance_sheet)} 条记录")
except Exception as e:
    print(f"   资产负债表获取失败: {e}")
    balance_sheet = None

# ============================================================
# 3. 获取现金流量表数据
# ============================================================
print("3. 获取现金流量表数据...")
try:
    cash_flow = ak.stock_financial_report_sina(
        stock=stock_code,
        symbol="现金流量表"
    )
    print(f"   现金流量表获取成功，共 {len(cash_flow)} 条记录")
except Exception as e:
    print(f"   现金流量表获取失败: {e}")
    cash_flow = None

# ============================================================
# 4. 获取历史日K线数据（用于估值分析）
# ============================================================
print("4. 获取历史行情数据...")
try:
    daily_kline = ak.stock_zh_a_hist(
        symbol=stock_code,
        period="daily",
        start_date="20220101",
        end_date="20260430",
        adjust="qfq"  # 前复权
    )
    print(f"   行情数据获取成功，共 {len(daily_kline)} 条记录")
    # 计算一些关键指标
    if len(daily_kline) > 0:
        latest_price = daily_kline.iloc[-1]["收盘"]
        high_52w = daily_kline.tail(252)["最高"].max()
        low_52w = daily_kline.tail(252)["最低"].min()
        print(f"   最新收盘价: {latest_price}")
        print(f"   52周最高: {high_52w}")
        print(f"   52周最低: {low_52w}")
except Exception as e:
    print(f"   行情数据获取失败: {e}")
    daily_kline = None

# ============================================================
# 5. 获取关键财务指标（ROE、毛利率等）
# ============================================================
print("5. 获取关键财务指标...")
try:
    # 使用东方财富接口获取核心财务指标
    fin_indicator = ak.stock_financial_abstract(code=stock_code)
    print(f"   财务指标获取成功，共 {len(fin_indicator)} 条记录")
except Exception as e:
    print(f"   财务指标获取失败: {e}")
    fin_indicator = None

# ============================================================
# 6. 保存数据
# ============================================================
print("\n6. 保存数据到 data/processed/...")
timestamp = datetime.now().strftime("%Y%m%d")
summary = {
    "stock_code": stock_code,
    "stock_name": stock_name,
    "update_date": timestamp,
}

if daily_kline is not None:
    daily_kline.to_csv(os.path.join(output_dir, f"hygon_daily_kline_{timestamp}.csv"), index=False)
    summary["latest_price"] = float(latest_price)
    summary["52w_high"] = float(high_52w)
    summary["52w_low"] = float(low_52w)
    summary["trading_days"] = len(daily_kline)
    print(f"   已保存: hygon_daily_kline_{timestamp}.csv")

if income_stmt is not None:
    income_stmt.to_csv(os.path.join(output_dir, f"hygon_income_stmt_{timestamp}.csv"), index=False)
    print(f"   已保存: hygon_income_stmt_{timestamp}.csv")

if balance_sheet is not None:
    balance_sheet.to_csv(os.path.join(output_dir, f"hygon_balance_sheet_{timestamp}.csv"), index=False)
    print(f"   已保存: hygon_balance_sheet_{timestamp}.csv")

if cash_flow is not None:
    cash_flow.to_csv(os.path.join(output_dir, f"hygon_cash_flow_{timestamp}.csv"), index=False)
    print(f"   已保存: hygon_cash_flow_{timestamp}.csv")

if fin_indicator is not None:
    fin_indicator.to_csv(os.path.join(output_dir, f"hygon_fin_indicator_{timestamp}.csv"), index=False)
    print(f"   已保存: hygon_fin_indicator_{timestamp}.csv")

# 保存汇总 JSON
with open(os.path.join(output_dir, f"hygon_summary_{timestamp}.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n数据获取完成！汇总文件: hygon_summary_{timestamp}.json")
print(json.dumps(summary, ensure_ascii=False, indent=2))
