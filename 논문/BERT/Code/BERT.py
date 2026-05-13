import torch
import torch.nn as nn

class BertEmbeddings(nn.Module):
    """3가지 임베딩을 합산: Token + Segment + Position"""
    def __init__(self, vocab_size=30000, hidden_size=768, 
                 max_len=512, num_segments=2):
        super().__init__()
        self.token_emb    = nn.Embedding(vocab_size, hidden_size)
        self.segment_emb  = nn.Embedding(num_segments, hidden_size)
        self.position_emb = nn.Embedding(max_len, hidden_size)
        self.layer_norm   = nn.LayerNorm(hidden_size)
        self.dropout      = nn.Dropout(0.1)

    def forward(self, input_ids, segment_ids):
        # input_ids:   (batch, seq_len)
        # segment_ids: (batch, seq_len)
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        
        emb = (self.token_emb(input_ids) 
             + self.segment_emb(segment_ids) 
             + self.position_emb(positions))   # (B, L, H)
        return self.dropout(self.layer_norm(emb))


class BertLayer(nn.Module):
    """Transformer Encoder 1개 블록: Self-Attention + FFN"""
    def __init__(self, hidden_size=768, num_heads=12, ffn_size=3072):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, 
                                               dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),                 # ⚠️ ReLU 아님!
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(0.1),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, attention_mask=None):
        # Self-Attention (양방향: 모든 토큰이 모든 토큰을 봄)
        attn_out, _ = self.attention(x, x, x, key_padding_mask=attention_mask)
        x = self.norm1(x + attn_out)           # Add & Norm

        # Feed-Forward Network
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)            # Add & Norm
        return x


class BERT(nn.Module):
    """BERT-Base: L=12, H=768, A=12"""
    def __init__(self, vocab_size=30000, hidden_size=768, 
                 num_layers=12, num_heads=12, ffn_size=3072):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size)
        self.encoder = nn.ModuleList([
            BertLayer(hidden_size, num_heads, ffn_size) 
            for _ in range(num_layers)
        ])
        
        # Pre-training Heads
        self.mlm_head = nn.Linear(hidden_size, vocab_size)  # MLM
        self.nsp_head = nn.Linear(hidden_size, 2)           # NSP

    def forward(self, input_ids, segment_ids, attention_mask=None):
        # 1. Embedding
        x = self.embeddings(input_ids, segment_ids)         # (B, L, H)
        
        # 2. Transformer 블록 L번 통과
        for layer in self.encoder:
            x = layer(x, attention_mask)                    # (B, L, H)
        
        # 3. 출력
        C = x[:, 0, :]                # [CLS] 벡터: (B, H)
        T = x                         # 토큰별 벡터: (B, L, H)
        
        mlm_logits = self.mlm_head(T) # (B, L, vocab_size)
        nsp_logits = self.nsp_head(C) # (B, 2)
        return mlm_logits, nsp_logits, C, T


# === 사용 예시 ===
model = BERT()
input_ids   = torch.randint(0, 30000, (2, 128))   # (batch=2, seq=128)
segment_ids = torch.zeros(2, 128, dtype=torch.long)

mlm_logits, nsp_logits, C, T = model(input_ids, segment_ids)
print(mlm_logits.shape)  # torch.Size([2, 128, 30000])
print(nsp_logits.shape)  # torch.Size([2, 2])
print(C.shape)           # torch.Size([2, 768])