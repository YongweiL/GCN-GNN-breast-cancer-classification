# GCN-GNN-breast-cancer-classification
Integrative Multi-Omics Approach with Deep Learning GCN and GNN for Breast Cancer Molecular Subtype Classification  

# ABSTRACT
Breast cancer molecular subtype classification is critical for personalized treatment strategies, yet challenges persist due to tumor heterogeneity and the complexity of multi-omics data. This study evaluates the performance of Graph Convolutional Networks (GCN) and GraphSAGE, two deep learning models, for classifying breast cancer subtypes using integrated multi-omics data from The Cancer Genome Atlas (TCGA). The methodology involves preprocessing genomic, epigenomic, and transcriptomic data, followed by dimensionality reduction via autoencoders and integration using Similarity Network Fusion (SNF). The fused data is then modeled using GCN and GraphSAGE, with performance assessed through confusion matrices, precision-recall curves, and key metrics such as accuracy, precision, recall, and F1-score. Results indicate that GCN achieves moderate performance for subtypes like Luminal A and Normal-like but struggles with rare subtypes (e.g., HER2-enriched, Basal-like), while GraphSAGE exhibits severe bias toward the majority class, rendering it ineffective. The study identifies limitations in dataset size, class imbalance, and feature representation as key factors impacting model performance. Recommendations for future work include expanding datasets, incorporating biologically meaningful node features, and employing advanced techniques to address imbalance. This research underscores the potential of graph-based deep learning for multi-omics integration while highlighting critical challenges in low-resource settings.
Keywords: Breast cancer subtypes, multi-omics integration, graph neural networks, GCN, GraphSAGE, deep learning, precision oncology.

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

