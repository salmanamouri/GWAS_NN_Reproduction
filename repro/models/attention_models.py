""" 
GINN + Attention → same Shapley-style interaction score
Suggested attention architecture
For each gene:
20 SNPs → 32 → embedding_dim

All genes:
10 × embedding_dim
    ↓
Multi-head self-attention
    ↓
Residual connection
    ↓
Layer normalization
    ↓
Mean pooling across genes
    ↓
MLP predictor
    ↓
Phenotype


Recommended first settings: embedding_dim = 16
num_heads = 4
attention_dropout = 0.1
predictor_hidden_dim = 100
"""

