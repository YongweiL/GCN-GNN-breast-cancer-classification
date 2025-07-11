# GCN-GNN-breast-cancer-classification
Integrative Multi-Omics Approach with Deep Learning GCN and GNN for Breast Cancer Molecular Subtype Classification  

# ABSTRACT
Breast cancer molecular subtype classification is critical for personalized treatment strategies, yet challenges persist due to tumor heterogeneity and multi-omics complexity. This study evaluates Graph Convolutional Networks (GCN) and GraphSAGE for classifying breast cancer subtypes using integrated multi-omics data from TCGA. Genomic, epigenomic, and transcriptomic data were preprocessed, reduced via autoencoders, and fused using Similarity Network Fusion (SNF). Performance analysis revealed that GCN achieved moderate accuracy (82.57%), with reasonable performance for Luminal A and Normal-like subtypes but limitations in rare subtypes (HER2-enriched, Basal-like). GraphSAGE, while attaining higher nominal accuracy (84.40%), exhibited severe majority-class bias, rendering it clinically unreliable. Key challenges included dataset size, class imbalance, and feature representation. Future directions emphasize larger datasets, biologically informed node features, and advanced imbalance mitigation. This work highlights the potential of graph-based deep learning for multi-omics integration while underscoring the need for robust solutions in low-resource scenarios.
//Keywords: Breast cancer subtypes, multi-omics integration, graph neural networks, GCN, GraphSAGE, deep learning, precision oncology.

In this study, integration of multi-omics data as shown in Table 2 (CNV, DNA methylation, and mRNA expression) from TCGA-BRCA samples to classify breast cancer subtypes（https://linkedomics.org/data_download/TCGA-BRCA/. The workflow as shown in Figure 4 includes data preprocessing, feature reduction using autoencoders, similarity network fusion (SNF), graph construction, and graph neural network (GNN) classification.
<img width="798" height="235" alt="image" src="https://github.com/user-attachments/assets/ebffb603-43e5-4e24-b0b1-77137bca849b" />

## Model Train Classification Report:
<img width="1285" height="367" alt="image" src="https://github.com/user-attachments/assets/3a9eaf8f-9fd4-40f9-8be0-78c496a64237" />
<img width="1320" height="397" alt="image" src="https://github.com/user-attachments/assets/df1ed101-e2ed-4da7-ad8d-a8c53e902859" />

## Precission-Recall Curve:
<img width="1627" height="697" alt="image" src="https://github.com/user-attachments/assets/0d2bb14f-db1e-4521-a6f7-d554caace346" />

## Prediction Results:
<img width="1712" height="765" alt="image" src="https://github.com/user-attachments/assets/9cc474de-3d9f-4923-8b16-a702ccf9dcbd" />
<img width="1088" height="652" alt="image" src="https://github.com/user-attachments/assets/ec4c4615-5117-4f88-85a9-1d98926f54b0" />

