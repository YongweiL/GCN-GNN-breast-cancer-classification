import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
import os

# Define paths to your preprocessed files
base_dir = r"C:\UTM Degree\y4s2\Bio Modelling Simulation_Dr Azurah\Project\preprocess_removeNan"
cnv_path = os.path.join(base_dir, "DATA_CNV.csv")
methylation_path = os.path.join(base_dir, "DATA_DNA.csv")
mrna_path = os.path.join(base_dir, "DATA_mRNA.csv")

# Load datasets with sample IDs as index
try:
    cnv_data = pd.read_csv(cnv_path, index_col=0)
    methylation_data = pd.read_csv(methylation_path, index_col=0)
    mrna_data = pd.read_csv(mrna_path, index_col=0)
except FileNotFoundError as e:
    print(f"Error: {e}. Check if files exist in:\n{base_dir}")
    exit()

def autoencoder_feature_selection(data, n_features=100):
    """Reduce features using Autoencoder."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Autoencoder architecture
    input_dim = scaled_data.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(n_features, activation='relu')(input_layer)
    decoder = Dense(input_dim, activation='sigmoid')(encoder)
    
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(scaled_data, scaled_data, epochs=50, batch_size=32, verbose=0)
    
    # Extract reduced features
    encoder_model = Model(inputs=input_layer, outputs=encoder)
    reduced_data = encoder_model.predict(scaled_data)

    # Preserve original row indices
    return pd.DataFrame(reduced_data, index=data.index)

# Apply autoencoder feature reduction
print("Running autoencoder for CNV...")
cnv_reduced = autoencoder_feature_selection(cnv_data)
print("Running autoencoder for Methylation...")
methylation_reduced = autoencoder_feature_selection(methylation_data)
print("Running autoencoder for mRNA...")
mrna_reduced = autoencoder_feature_selection(mrna_data)

# Define the output directory for reduced files
output_dir = r"C:\UTM Degree\y4s2\Bio Modelling Simulation_Dr Azurah\Project\autoencoder_removeNan"
os.makedirs(output_dir, exist_ok=True)

# Save reduced features with index (sample IDs)
cnv_reduced.to_csv(os.path.join(output_dir, "cnv_reduced.csv"))
methylation_reduced.to_csv(os.path.join(output_dir, "methylation_reduced.csv"))
mrna_reduced.to_csv(os.path.join(output_dir, "mrna_reduced.csv"))

print(f"Reduced datasets saved to:\n{output_dir}")
