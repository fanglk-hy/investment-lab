"""
贵州茅台（600519）财务数据获取脚本
获取历史财务数据、估值数据，用于深度分析
"""
import akshare as ak
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(output_dir, exist_ok=True)

stock_code = "600519"
stock_name = "贵州茅台"

print(f"正在获取 {stock_name}({stock_code}) 的财务数据...\n")

# 1. 利润表
print("1. 获取利润表数据...")
try:
    income_stmt = ak.stock_financial_report_sina(stock=stock_code, symbol="利润表")
    print(f"   利润表获取成功，共 {len(income_stmt)} 条记录")
    income_stmt.to_csv(os.path.join(output_dir, f"moutai_income_stmt_{datetime.now().strftime('%Y%m%d')}.csv"), index=False)
except Exception as e:
    print(f"   利润表获取失败: {e}")

# 2. 资产负债表
print("2. 获取资产负债表数据...")
try:
    balance_sheet = ak.stock_financial_report_sina(stock=stock_code, symbol="资产负债表")
    print(f"   资产负债表获取成功，共 {len(balance_sheet)} 条记录")
    balance_sheet.to_csv(os.path.join(output_dir, f"moutai_balance_sheet_{datetime.now().strftime('%Y%m%d')}.csv"), index=False)
except Exception as e:
    print(f"   资产负债表获取失败: {e}")

# 3. 现金流量表
print("3. 获取现金流量表数据...")
try:
    cash_flow = ak.stock_financial_report_sina(stock=stock_code, symbol="现金流量表")
    print(f"   现金流量表获取成功，共 {len(cash_flow)} 条记录")
    cash_flow.to_csv(os.path.join(output_dir, f"moutai_cash_flow_{datetime.now().strftime('%Y%m%d')}.csv"), index=False)
except Exception as e:
    print(f"   现金流量表获取失败: {e}")

# 4. 关键财务指标摘要
print("4. 获取关键财务指标...")
try:
    fin_indicator = ak.stock_financial_abstract_ths(symbol=stock_code, indicator="按报告期")
    print(f"   财务指标获取成功，共 {len(fin_indicator)} 条记录")
    fin_indicator.to_csv(os.path.join(output_dir, f"moutai_fin_indicator_{datetime.now().strftime('%Y%m%d')}.csv"), index=False)
except Exception as e:
    print(f"   财务指标获取失败: {e}")

print("\n数据获取完成！")
