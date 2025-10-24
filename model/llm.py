from typing import Tuple, Dict
import torch
import torch.nn as nn
from torch.nn


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
        self.d_half = d_model // 2

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
    
    def _get_rope(self, x:torch.Tensor) -> torch.Tensor:
        """
        q = [q1, q2, ... qs]
        -> [[q1, q2], [q3, q4], ...[qs-1, qs]]
        -> let x = [x1, x2]; rotation R(x) = [[cost, -sint], [sint, cost]] @ x = e^it*(x1 + ix2) 
        """
        b, h, s, d = x.shape

        #convert x to complex number
        x = x.float().reshape(b, h, s, self.d_half, 2)
        x_complex = torch.view_as_complex(x)   # (b, h, s, d_half)

        #pos and thetas
        pos = torch.arange(s, device=x.device) #(s,)
        freqs = 1/torch.pow(10000, torch.arange(0, self.d_half, device=x.device)/self.d_half) #(d_half,)
        thetas = torch.einsum('s,d->sd',pos,freqs) #outer product thetas = (s,d_half)
        rot = torch.polar(torch.ones_like(thetas), thetas) #mag = 1, angle = thetas (s,d_half)

        #rotate x
        x = x_complex * rot[None, None, :, :] #(b, h, s, d_half) * (1,1,s,d_half) (element wise multiply)
        x = torch.view_as_real(x).reshape(b, h, s, d)

        return x #(b,h,s,d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._get_qkv(x)
        q = self._get_rope(q)
        k = self._get_rope(k)

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
    def __init__(self, model_dim:int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(model_dim))
        self.beta = nn.Parameter(torch.zeros(model_dim))
        self.eps = 1e-6
    
    def forward(self, x):
        # x.shape (b,s,d)
        mean = torch.mean(x, dim=-1, keep_dim=True) #(b, s)
        #(b, s) (unbiased=False -> var = 1/N*sum(x_i - mean)**2) and not 1/N-1*()
        #apparently has smoother gradients.
        var = torch.var(x, dim=-1, keep_dim=True, unbiased=False) 
        x = (x - mean)/(torch.sqrt(var) + self.eps)
        x = self.gamma*x + self.beta

        return x

class TransformerBlock(nn.Module):
    """
    The class which defines the small language model to be trained on tiny stories dataset.
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
        #x.shape -> (b, s)
        #1. embed the tokens
        x = self.token_emb(x) # x.shape -> (b, s, d_model)
        
        #2. layer norm before mha? some archs have that. for v1, let's stick with layernorm after mha
        #...
        
        #3. MHA + RoPE + residual + ln1
        x = x + self.mha(self.ln1(x)) # x.shape -> (b, s, d_model)

        #5. ffn + residual + ln2
        x = x + self.ffn(self.ln2(x)) # x.shape -> (b, s, d_model)

        return x





         
