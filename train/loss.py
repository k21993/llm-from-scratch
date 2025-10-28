from typing import Any
import torch
import torch.nn as nn


class CrossEntropyLoss:

    def __init__(self):
        pass
    
    def _softmax(self, x:torch.Tensor) -> torch.Tensor:
        #x.shape (B, S, V)
        return torch.exp(x)/torch.sum(torch.exp(x), dim=-1, keepdim=True)
    
    def __call__(self, y_true:torch.Tensor, y_pred:torch.Tensor):
        """
        y_true: GT vocab IDs of (B, S)
        y_pred: predicted model logits (raw scores) (B, S, Vocab_Size)
        returns loss = -log(sum(e^x/(sum(e^y)))) x-> correct class idx
        b = 1, s = 2, v = 5
        y_true = [3, 2]
        y_pred = [[1.75, -2.4, 3.2, 5.2, -3.1],[1.75, -2.4, 4.2, 2.2, -3.5]]
        pred_prob = [[0.1, 0.05, 0.3, 0.65, 0.001], [0.1, 0.05, 0.65, 0.3, 0.001]]
        ce_loss = -log(0.65/(sum()))
        """
        #subtract max from y_pred to stabilize softmax
        y_pred = y_pred - torch.max(y_pred, dim=-1, keepdim=True).values #(B, S, V)

        #get log probs (for all vocab indices)
        log_probs = y_pred - torch.log(torch.sum(torch.exp(y_pred), dim=-1, keepdim=True)) #(B, S, V)

        #gather log_props along S dimension
        y_true = y_true.unsqueeze(-1) #(B, S, 1)
        correct_idx_log_probs = torch.gather(input=log_probs, dim=2, index=y_true).squeeze(-1) #gather along the V dim (B, S)

        #take mean along sequence and along batch
        return -torch.mean(correct_idx_log_probs)





