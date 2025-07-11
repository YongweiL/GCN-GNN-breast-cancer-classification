import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import snf

# === Load reduced omics datasets ===
base_dir = r"C:\UTM Degree\y4s2\Bio Modelling Simulation_Dr Azurah\Project\autoencoder_removeNan"
cnv_reduced = pd.read_csv(os.path.join(base_dir, "DNA_cnv_reduced.csv"), index_col=0)
methylation_reduced = pd.read_csv(os.path.join(base_dir, "DNA_methylation_reduced.csv"), index_col=0)
mrna_reduced = pd.read_csv(os.path.join(base_dir, "DNA_mrna_reduced.csv"), index_col=0)

# === Load target labels ===
label_path = os.path.join(base_dir, "..", "TCGA_BRCA_sampleID_label_filtered.csv")
labels_df = pd.read_csv(label_path)  # expects columns: sampleID,label

# === Find common samples across all omics ===
common_samples = cnv_reduced.index.intersection(methylation_reduced.index).intersection(mrna_reduced.index)

# === Diagnostics ===
print("Total samples in target label:", len(labels_df))
print("Samples in CNV:", len(cnv_reduced))
print("Samples in Methylation:", len(methylation_reduced))
print("Samples in mRNA:", len(mrna_reduced))
print("Samples in ALL 3 (used in SNF):", len(common_samples))

missing_samples = set(labels_df['sampleID']) - set(common_samples)
print(f"⚠️ {len(missing_samples)} samples dropped because not present in all omics files.")
print("Dropped sample IDs (first 10):", list(missing_samples)[:10])

# === Filter omics and labels to common samples ===
cnv_reduced = cnv_reduced.loc[common_samples]
methylation_reduced = methylation_reduced.loc[common_samples]
mrna_reduced = mrna_reduced.loc[common_samples]

labels_df = labels_df[labels_df['sampleID'].isin(common_samples)]
labels_df = labels_df.set_index("sampleID").loc[common_samples]  # ensure same order

# === Save aligned labels ===
labels_df.to_csv("aligned_labels.csv")

# === Scale omics features ===
scaler = StandardScaler()
cnv_scaled = scaler.fit_transform(cnv_reduced.values)
methylation_scaled = scaler.fit_transform(methylation_reduced.values)
mrna_scaled = scaler.fit_transform(mrna_reduced.values)

# === Compute cosine similarity matrices ===
cnv_sim = cosine_similarity(cnv_scaled)
methylation_sim = cosine_similarity(methylation_scaled)
mrna_sim = cosine_similarity(mrna_scaled)

# === Apply SNF ===
K = 20
t = 20
fused_network = snf.snf([cnv_sim, methylation_sim, mrna_sim], K=K, t=t)

# === Save fused matrix ===
fused_df = pd.DataFrame(fused_network, index=common_samples, columns=common_samples)
fused_df.to_csv("fused_similarity_matrix_removeNan.csv")

# === Done ===
print("\n✅ SNF fusion and label alignment completed.")
print("- Fused matrix saved to: fused_similarity_matrix_removeNan.csv")
print("- Aligned labels saved to: aligned_labels.csv")
