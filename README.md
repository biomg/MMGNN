# Multi-Modal Graph Neural Networks with Cross-Modal Attention and Contrastive Learning for Molecular Property Prediction
Molecular property prediction plays a pivotal role in drug discovery, yet current approaches frequently encounter challenges in effectively integrating multimodal information including molecular graph structures and fingerprint features. 
This study presents a novel multimodal graph neural network (MM-GNN) framework that achieves synergistic optimization of molecular representations through hierarchical attention mechanisms and contrastive pretraining strategies. 
The architecture employs dual graph encoders (graph isomorphism network and graph attention network) to respectively capture local atomic interactions and global graph semantic features. 
A cross-modal attention module is innovatively designed to dynamically align latent spaces between graph features and molecular fingerprints, complemented by a gated fusion mechanism for adaptive integration of multiscale representations. 
During pretraining, a contrastive learning paradigm with NT-Xent loss is implemented, leveraging molecular graph data augmentation to learn invariant representations. Ablation studies confirm the critical contributions of cross-modal attention and contrastive pretraining in capturing complementary molecular information. 
This work establishes an innovative framework for molecular representation learning, demonstrating significant potential in virtual drug screening applications. The proposed methodology advances current techniques through its systematic integration of structural and substructural features, offering a robust solution for multimodal molecular data analysis in computational drug discovery.

# Dependency
python 3.7.16<br>
torch 1.12.1<br>
tensorflow 2.11.0 

# Pre-training
python main_pretrain.py<br>

# Pretrain
python main_gnn_classification.py<br>
python main_gnn_regression.py
