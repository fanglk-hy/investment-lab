"""
宏观数据获取脚本 — 中美利差、汇率、流动性指标
"""
import akshare as ak
import pandas as pd
import json
import os
from datetime import datetime

output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("宏观数据获取 — 2026年5月")
print("=" * 60)

results = {}

# 1. 中国10年期国债收益率
print("\n1. 中国10年期国债收益率...")
try:
    bond_cn = ak.bond_china_yield(start_date="2024-01-01", end_date="2026-05-01")
    if bond_cn is not None and len(bond_cn) > 0:
        bond_cn.to_csv(os.path.join(output_dir, f"china_bond_yield_{datetime.now().strftime('%Y%m%d')}.csv"), index=False)
        latest = bond_cn.iloc[-1]
        print(f"   获取成功，最新数据: {latest.to_dict()}")
except Exception as e:
    print(f"   获取失败: {e}")

# 2. 人民币汇率
print("\n2. 人民币汇率...")
try:
    fx = ak.currency_boc_sina(symbol="美元", start_date="2024-01-01", end_date="2026-05-01")
    if fx is not None and len(fx) > 0:
        fx.to_csv(os.path.join(output_dir, f"usdcny_{datetime.now().strftime('%Y%m%d')}.csv"), index=False)
        print(f"   获取成功，共 {len(fx)} 条")
except Exception as e:
    print(f"   获取失败: {e}")

# 3. 中国社融/M2数据
print("\n3. 中国社融与M2...")
try:
    macro_cn = ak.macro_china()
    if macro_cn is not None and len(macro_cn) > 0:
        macro_cn.to_csv(os.path.join(output_dir, f"china_macro_{datetime.now().strftime('%Y%m%d')}.csv"), index=False)
        print(f"   获取成功，共 {len(macro_cn)} 条")
except Exception as e:
    print(f"   获取失败: {e}")

# 4. 黄金价格
print("\n4. 黄金价格...")
try:
    gold = ak.spot_gold(symbol="伦敦金")
    if gold is not None and len(gold) > 0:
        print(f"   获取成功")
except Exception as e:
    print(f"   获取失败: {e}")

print("\n数据获取完成！")
