import torch                                           # PyTorch 라이브러리 본체를 불러옴
import torch.nn as nn                                  # PyTorch의 신경망(Neural Network) 모듈을 nn이라는 짧은 이름으로 불러옴
import torch.nn.functional as F                        # 학습 가능한 파라미터가 없는 연산을 쓸 때 주로 사용
import math                                            # PyTorch 텐서가 아닌 일반 파이썬 스칼라 값을 계산할 때 사용

# 1) Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V, mask=None):  # Query, Key, Value 세 텐서와 선택적 마스크를 받는 함수를 정의
    d_k = Q.size(-1)                                   # Q의 마지막 차원(키 벡터의 차원, 보통 64)을 가져옴
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # 논문 식 (1)
    if mask is not None:                                
        scores = scores.masked_fill(mask == 0, -1e9)   # 마스크가 0인 위치(보면 안 되는 위치)에 매우 작은 값(-1e9)을 넣음
    attn = F.softmax(scores, dim=-1)                   # softmax를 거치면 그 자리는 거의 0이 됨 / 마지막 차원(키 축)을 따라 softmax를 적용해 확률 분포 형태의 attention 가중치를 얻음
    return attn @ V                                    # 가중치와 V를 곱해 가중합을 반환


# 2) Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, h=8):       # 모델 차원 512, 헤드 수 8개로 초기화
        super().__init__()
        self.h = h
        self.d_k = d_model // h                 # 각 헤드가 담당할 차원은 512/8 = 64
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # Q, K, V를 만들기 위한 선형 투영 3개와, 헤드들을 합친 뒤 적용할 출력 투영 1개를 준비

    def forward(self, Q, K, V, mask=None):      # 배치 크기를 가져옴
        B = Q.size(0)
        
        # (B, seq, d_model) -> (B, h, seq, d_k)  [헤드 분할]
        # 선형변환 후 (B, seq, d_model) → (B, seq, h, d_k)로 reshape 하고, transpose로 (B, h, seq, d_k) 모양으로 만듦
        # 이렇게 하면 헤드 차원이 앞으로 와서 헤드별로 독립적인 attention을 병렬 계산할 수 있음
        Q = self.W_q(Q).view(B, -1, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, -1, self.h, self.d_k).transpose(1, 2)
        
        # 각 헤드별로 attention을 계산
        out = scaled_dot_product_attention(Q, K, V, mask)
        
        # 다시 합치기
        # (B, h, seq, d_k)를 다시 (B, seq, h, d_k)로 되돌리고, 마지막 두 축을 합쳐 (B, seq, d_model)로 만
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.W_o(out)        # 헤드들을 합친 결과에 출력 투영을 적용해 반환


# 3) Positional Encoding
# 최대 길이 5000, 차원 512짜리 위치 인코딩 행렬을 0으로 초기화
class PositionalEncoding(nn.Module):               
    def __init__(self, d_model=512, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        # 0부터 4999까지의 위치 인덱스를 (5000, 1) 모양으로 만듦
        pos = torch.arange(0, max_len).unsqueeze(1).float()

        # 논문 공식의 분모 10000^(2i/d_model)을 계산
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(math.log(10000.0) / d_model))

        # 짝수 차원에는 sin, 홀수 차원에는 cos 값을 채움
        # 이렇게 하면 위치마다 고유한 패턴이 생기고, 상대적 위치 관계도 학습할 수 있게 됨
        pe[:, 0::2] = torch.sin(pos * div)   # 짝수 차원
        pe[:, 1::2] = torch.cos(pos * div)   # 홀수 차원

        # 배치 차원을 추가해 (1, max_len, d_model)로 만들고 buffer로 등록
        self.register_buffer('pe', pe.unsqueeze(0))
        
    # 입력 시퀀스 길이만큼 잘라서 그대로 더함
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# 4) Feed-Forward Network
# 512 → 2048 → 512로 차원을 늘렸다 줄이는 2층 FFN
class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    # 순서대로 통과
    def forward(self, x):
        return self.net(x)


# 5) Encoder Layer (한 층)
# 한 층을 구성하는 부품: self-attention, FFN, LayerNorm 2개, Dropout
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
        # self-attention(Q=K=V=x)을 돌리고, dropout 적용 후 잔차 연결(x + ...)과 LayerNorm을 거침
        x = self.norm1(x + self.dropout(self.attn(x, x, x, mask)))

        # FFN에도 같은 패턴(잔차 + 정규화)을 적용하고 반환
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# 6) Decoder Layer (한 층)
# 디코더는 attention이 두 개라 LayerNorm도 3개
#  첫 attention은 미래를 보지 못하게 마스킹하는 self-attention, 두 번째는 인코더 출력을 보는 cross-attention
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

    # 디코더 입력으로 self-attention을 돌림
    # tgt_mask로 미래 토큰을 가림
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        
        # Q는 디코더, K/V는 인코더 출력
        # Q는 디코더(x)에서, K/V는 인코더 출력(enc_out)에서 가져옴
        # 즉 디코더가 인코더 정보를 "참조"하는 단계 (src_mask는 인코더 쪽 패딩을 가림)
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))

        # 마지막으로 FFN + 잔차 + 정규화 후 반환
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
