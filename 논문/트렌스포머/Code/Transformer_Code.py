import torch                                           # PyTorch 라이브러리 본체를 불러옴
import torch.nn as nn                                  # PyTorch의 신경망(Neural Network) 모듈을 nn이라는 짧은 이름으로 불러옴
import torch.nn.functional as F                        # 학습 가능한 파라미터가 없는 연산을 쓸 때 주로 사용
import math                                            # PyTorch 텐서가 아닌 일반 파이썬 스칼라 값을 계산할 때 사용

# 1) Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # 논문 식 (1)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)   # 마스킹
    attn = F.softmax(scores, dim=-1)
    return attn @ V


# 2) Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, h=8):
        super().__init__()
        self.h = h
        self.d_k = d_model // h     # 64
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        # (B, seq, d_model) -> (B, h, seq, d_k)  [헤드 분할]
        Q = self.W_q(Q).view(B, -1, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, -1, self.h, self.d_k).transpose(1, 2)
        out = scaled_dot_product_attention(Q, K, V, mask)
        # 다시 합치기
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.W_o(out)


# 3) Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)   # 짝수 차원
        pe[:, 1::2] = torch.cos(pos * div)   # 홀수 차원
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# 4) Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# 5) Encoder Layer (한 층)
class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, h)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 잔차 연결 + LayerNorm
        x = self.norm1(x + self.dropout(self.attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# 6) Decoder Layer (한 층)
class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h)     # masked
        self.cross_attn = MultiHeadAttention(d_model, h)    # enc-dec
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        # Q는 디코더, K/V는 인코더 출력
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


# 7) 전체 Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, N=6, h=8, d_ff=2048):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pe = PositionalEncoding(d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, h, d_ff) for _ in range(N)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, h, d_ff) for _ in range(N)])
        self.out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 인코더
        e = self.pe(self.src_embed(src) * math.sqrt(512))
        for layer in self.encoder:
            e = layer(e, src_mask)
        # 디코더
        d = self.pe(self.tgt_embed(tgt) * math.sqrt(512))
        for layer in self.decoder:
            d = layer(d, e, src_mask, tgt_mask)
        return self.out(d)
