import pandas as pd

# ===============================
# 1. 路径配置
# ===============================
original_csv = "/mnt/sda/yx/knee/5t/classify/knee_radiomics_original_features.csv"
wavelet_csv  = "/mnt/sda/yx/knee/5t/classify/knee_radiomics_wavelet_features.csv"
shape_csv    = "/mnt/sda/yx/knee/5t/classify/knee_radiomics_shape_features.csv"
output_csv   = "/mnt/sda/yx/knee/5t/classify/knee_combined_features.csv"

# ===============================
# 2. 读取数据
# ===============================
df_ori = pd.read_csv(original_csv)
df_wav = pd.read_csv(wavelet_csv)
# df_shape = pd.read_csv(shape_csv)

print("Original CSV shape:", df_ori.shape)
print("Wavelet  CSV shape:", df_wav.shape)
# print("Shape  CSV shape:", df_shape.shape)

# ===============================
# 3. 定义元信息列（非常关键）
# ===============================
meta_cols = ["case_id", "region", "grade"]

# 检查元信息列是否存在
for col in meta_cols:
    assert col in df_ori.columns, f"{col} not in original csv"
    assert col in df_wav.columns, f"{col} not in wavelet csv"
    # assert col in df_shape.columns, f"{col} not in shape csv"

# ===============================
# 4. 合并（按病例 + 区域）
# ===============================
df_merged = pd.merge(
    df_ori,
    df_wav,
    on=meta_cols,
    how="inner"
)

# df_merged = pd.merge(
#     df_merged,
#     df_shape,
#     on=meta_cols,
#     how="inner"
# )

print("Merged CSV shape:", df_merged.shape)

# ===============================
# 5. Sanity check（强烈推荐）
# ===============================

# (1) 样本是否重复
dup = df_merged[["case_id", "region"]].duplicated().sum()
print("Duplicated samples:", dup)
# assert dup == 0, "❌ 存在重复样本"

# (2) original / wavelet 特征数量
original_feats = [c for c in df_merged.columns if c.startswith("original_") and not c.startswith("original_shape")]
wavelet_feats  = [c for c in df_merged.columns if c.startswith("wavelet-")]
# shape_feats = [c for c in df_merged.columns if c.startswith("original_shape")]

print(f"Original features: {len(original_feats)}")
print(f"Wavelet  features: {len(wavelet_feats)}")
# print(f"Shape  features: {len(shape_feats)}")

assert len(original_feats) > 0, "❌ 没有 original 特征"
assert len(wavelet_feats) > 0, "❌ 没有 wavelet 特征"
# assert len(shape_feats) > 0, "❌ 没有 shape 特征"

# ===============================
# 6. 保存结果
# ===============================
df_merged.to_csv(output_csv, index=False)
print(f"✅ Finished! Merged features saved to:\n{output_csv}")

