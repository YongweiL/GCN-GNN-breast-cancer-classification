import pandas as pd
import numpy as np
import os

def load_and_preprocess(file_path, transpose=True):
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    initial_shape = df.shape

    if transpose:
        df = df.T
        transposed_shape = df.shape
    else:
        transposed_shape = df.shape

    df = df.loc[~df.index.duplicated(keep='first')]
    after_dedup_shape = df.shape

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '_')

    final_shape = df.shape

    if df.select_dtypes(include=[np.number]).shape[1] > 0:
        summary = df.describe().T[['mean', 'min', 'max', 'std']]
    else:
        summary = pd.DataFrame()

    preprocessing_steps = {
        'Initial': initial_shape,
        'After Transposition': transposed_shape,
        'After Duplicate Removal': after_dedup_shape,
        'Final': final_shape
    }

    return df, preprocessing_steps, summary


def filter_known_samples_by_columns(df, valid_sample_ids):
    """
    Filters the dataframe to only include columns whose names are in valid_sample_ids.
    """
    return df.loc[:, df.columns.isin(valid_sample_ids)]


# === Load sampleID-label file ===
label_path = r"C:\UTM Degree\y4s2\Bio Modelling Simulation_Dr Azurah\Project\TCGA_BRCA_sampleID_label_filtered.csv"
label_df = pd.read_csv(label_path)
label_df = label_df.dropna()

# Get known sample IDs
valid_sample_ids = label_df['sampleID'].tolist()

# === File paths ===
cnv_path = r"C:\UTM Degree\y4s2\Bio Modelling Simulation_Dr Azurah\Project\DATAORI_CNV.cct"
methylation_path = r"C:\UTM Degree\y4s2\Bio Modelling Simulation_Dr Azurah\Project\DATAORI_DNAmethylation.cct"
mrna_path = r"C:\UTM Degree\y4s2\Bio Modelling Simulation_Dr Azurah\Project\DATAORI_mRNA.cct"

# === Load raw datasets ===
cnv_raw = pd.read_csv(cnv_path, sep='\t', index_col=0)
methylation_raw = pd.read_csv(methylation_path, sep='\t', index_col=0)
mrna_raw = pd.read_csv(mrna_path, sep='\t', index_col=0)

# === Filter columns to only known sample IDs ===
cnv_filtered = filter_known_samples_by_columns(cnv_raw, valid_sample_ids)
methylation_filtered = filter_known_samples_by_columns(methylation_raw, valid_sample_ids)
mrna_filtered = filter_known_samples_by_columns(mrna_raw, valid_sample_ids)

# === Save temporary filtered versions for preprocessing ===
tmp_dir = "filtered_tmp"
os.makedirs(tmp_dir, exist_ok=True)

cnv_filtered_path = os.path.join(tmp_dir, "filtered_cnv.tsv")
meth_filtered_path = os.path.join(tmp_dir, "filtered_methylation.tsv")
mrna_filtered_path = os.path.join(tmp_dir, "filtered_mrna.tsv")

cnv_filtered.to_csv(cnv_filtered_path, sep='\t')
methylation_filtered.to_csv(meth_filtered_path, sep='\t')
mrna_filtered.to_csv(mrna_filtered_path, sep='\t')

# === Preprocess datasets ===
cnv_df, cnv_steps, cnv_summary = load_and_preprocess(cnv_filtered_path)
methylation_df, meth_steps, meth_summary = load_and_preprocess(meth_filtered_path)
mrna_df, mrna_steps, mrna_summary = load_and_preprocess(mrna_filtered_path, transpose=False)

# === Save final preprocessed datasets ===
output_dir = "preprocess_removeNan"
os.makedirs(output_dir, exist_ok=True)

cnv_df.to_csv(os.path.join(output_dir, "DATA_CNV.csv"))
methylation_df.to_csv(os.path.join(output_dir, "DATA_DNA.csv"))
mrna_df.to_csv(os.path.join(output_dir, "DATA_mRNA.csv"))

# === Print summary tables ===
print("CNV Dataset Processing Steps:")
print(pd.DataFrame.from_dict(cnv_steps, orient='index', columns=['Rows', 'Columns']))
print("\nMethylation Dataset Processing Steps:")
print(pd.DataFrame.from_dict(meth_steps, orient='index', columns=['Rows', 'Columns']))
print("\nmRNA Dataset Processing Steps:")
print(pd.DataFrame.from_dict(mrna_steps, orient='index', columns=['Rows', 'Columns']))

print("\nCNV Summary Statistics:")
print(cnv_summary.head() if not cnv_summary.empty else "No numeric columns to summarize.")
print("\nMethylation Summary Statistics:")
print(meth_summary.head() if not meth_summary.empty else "No numeric columns to summarize.")
print("\nmRNA Summary Statistics:")
print(mrna_summary.head() if not mrna_summary.empty else "No numeric columns to summarize.")
