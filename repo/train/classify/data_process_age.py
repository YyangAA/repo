import pandas as pd

# ========================
# 参数区（按需修改）
# ========================
input_excel = "膝关节软骨-年龄-性别.xlsx"      # 原始Excel路径
output_excel = "output_filtered_age.xlsx"  # 输出Excel路径
min_age = 40                    # 指定年龄阈值

# ========================
# 读取 Excel
# ========================
df = pd.read_excel(input_excel)

# ========================
# 去除年龄小于指定值的行
# ========================
df_filtered = df[df["年龄"] >= min_age]
df_filtered = df_filtered[df_filtered["problem"].isna()]

# ========================
# 保存结果
# ========================
df_filtered.to_excel(output_excel, index=False)

print(f"处理完成，共保留 {len(df_filtered)} 行数据，结果已保存到 {output_excel}")
