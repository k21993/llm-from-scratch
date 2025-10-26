from typing import Tuple
import torch #type: ignore
import torch.nn as nn # type: ignore



class RoPE(nn.Module):
    """
    q = [q1, q2, ... qs]
    -> [[q1, q2], [q3, q4], ...[qs-1, qs]]
    -> let x = [x1, x2]; rotation R(x) = [[cost, -sint], [sint, cost]] @ x = e^it*(x1 + ix2) 
    """
    def __init__(self, seq_len:int, head_dim:int):
        super().__init__()
        self.d = head_dim
        self.d_half = head_dim // 2
        pos = torch.arange(0,seq_len) #(s,)
        freqs = 1/torch.pow(10000, torch.arange(0,self.d_half,2)/self.d) # 10000 ^ (-2*i/d) (d/2,)
        thetas = torch.einsum('s,d->sd',pos, freqs)
        thetas = torch.polar(torch.ones_like(thetas), thetas) # convert to polar with mag 1 and angle theta = e^it
        self.register_buffer("thetas", thetas) # (s, d/2)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        b,h,s,d = x.shape
        x = x.float().view(b, h, s, self.d_half, 2) #( b, h, s, d/2, 2)
        x = torch.view_as_complex(x) #(b, h, s, d/2)
        x = torch.einsum('bhsd,sd->bhsd', x, self.thetas) #TODO: if s < self.seq_len?
        x = torch.view_as_real(x).reshape(b, h, s, d) #(b, h, s, d/2, 2) -> (b, h, s, d)

        return x #(b, h, s, d)

class MultiHeadAttention(nn.Module):
    """
    A class that implements MHA, with RoPE
    """
    def __init__(self, d_model:int, num_heads:int, seq_len:int) -> None:
        super().__init__()
        self.w_qkv = nn.Linear(d_model, 3*d_model)
        self.num_heads = num_heads
        self.out = nn.Linear(d_model, d_model)
        self.register_buffer("causal_mask", #so that its part of the model state dict and is automatically placed on device
                                                torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
                                                ) #(1, 1, S, S)
        self.rope = RoPE(seq_len=seq_len, head_dim=d_model // num_heads)

    def _get_qkv(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        #x.shape -> (B, S, D)
        x = self.w_qkv(x)  #(x.shape -> (B, S, 3*D))
        q, k, v = torch.chunk(x, chunks=3, dim=-1) #(B, S, D))
        B, S, D = q.shape
        assert D % self.num_heads == 0, f'''num heads should be 
        divisible by model dim D, found: num heads: {self.num_heads}, model dim: {D}'''

        q = q.view(B, S, self.num_heads, D // self.num_heads).transpose(1,2) # (B, H, S, d)
        k = k.view(B, S, self.num_heads, D // self.num_heads).transpose(1,2) # (B, H, S, d)
        v = v.view(B, S, self.num_heads, D // self.num_heads).transpose(1, 2) # (B, H, S, d)

        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._get_qkv(x) #(b, h, s, d)
        q = self.rope(q) #(b, h, s, d)
        k = self.rope(k) #(b, h, s, d)

        attn_scores = torch.einsum('bhqd,bhkd->bhqk', q, k)
        # attn_scores = attn_scores/q.shape[-1]**0.5 #(b, h, q, k)
        s = q.shape[2] #get the seq len from q, if cur seq length < global seq length for shorter sequences
        attn_scores = (attn_scores/q.shape[-1]**0.5).masked_fill(self.causal_mask[:, :,:s, :s]==0, float('-inf')) #causal mask
        attn_scores = torch.softmax(attn_scores, dim=-1) 
        o = torch.einsum('bhqk,bhkd->bhqd', attn_scores, v) #(b, h, s, d)
        b, h, s, d_head = o.shape
        o = o.transpose(1, 2).contiguous().reshape(b, s, h*d_head) #(b, s, d_model)
        o = self.out(o)

        return o

class LayerNorm(nn.Module):
    """
    layernorm applies normalization along the token emb dimension d
    mean = mean(x[b, s, :]) -> (b, s)
    var = var(x[b,s,:]) -> (b, s)
    x = (x - mean)/(sqrt(var) + eps)
    x = gamma*x + beta (rescale and shift)
    """
    def __init__(self, d_model:int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = 1e-6
    
    def forward(self, x):
        # x.shape (b,s,d)
        mean = torch.mean(x, dim=-1, keepdim=True) #(b, s)
        #(b, s) (unbiased=False -> var = 1/N*sum(x_i - mean)**2) and not 1/N-1*()
        #apparently has smoother gradients.
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False) 
        x = (x - mean)/(torch.sqrt(var) + self.eps)
        x = self.gamma*x + self.beta

        return x

class TransformerBlock(nn.Module):
    """
    The class which defines the Transformer block (mha with rope, ffn, and ln).
    """

    def __init__(self, vocab_size: int, d_model:int, num_heads:int, seq_len:int) -> None:
        super().__init__()

        self.num_heads = num_heads

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, seq_len=seq_len)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
            )
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        # MHA + RoPE + residual + ln1
        x = x + self.mha(self.ln1(x)) # x.shape -> (b, s, d_model)

        # ffn + residual + ln2
        x = x + self.ffn(self.ln2(x)) # x.shape -> (b, s, d_model)

        return x
    
class TransformerLM(nn.Module):
    """
    The entry point to define the language model object which combines:
    1. embedding
    2. transformer block (mha+ rope, ffn, ln)
    3. final ffn + softmax over the vocab

    Forward:
    x: (batch, seq_len) token indices
    Returns:
    logits: (batch, seq_len, vocab_size) raw scores before softmax

    """

    def __init__(self,vocab_size: int, d_model:int, num_heads:int, seq_len:int, n_layers:int):
        super().__init__()
        self.n_layers = n_layers
        self.token_emb = nn.Embedding(vocab_size, d_model)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(vocab_size, d_model, num_heads, seq_len) for _ in range(n_layers)
            ]
        )
        self.ln = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        #[optional] weight tying: to reduce num params
        self.lm_head.weight = self.token_emb.weight
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #embed tokens: (b, s) -> (b, s, d)
        x = self.token_emb(x)
        # print("x.shape after emb: ", x.shape)
        #mha with rope + ln + ffn + ln: (b, s, d) -> (b, s, d)
        for block in self.transformer_blocks:
            x = block(x)
            # print("x.shape after mha: ", x.shape)

        #ln + final ffn
        x = self.ln(x)
        x = self.lm_head(x) 

        return x
    

if __name__ == "__main__":
    b = 1
    s = 12
    num_heads = 4
    vocab_size = 16
    d_model = 8
    n_layers = 2

    test_llm = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        seq_len=s,
        n_layers=n_layers
    )

    x = torch.randint(0, vocab_size, size=(b,s))
    print(x.shape)

    print(test_llm(x).shape)







         
