import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCrossModalAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadCrossModalAttention, self).__init__()
        assert d_model % n_heads == 0, "Number of heads must evenly divide d_model"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, A, B):
        batch_size = A.size(0)

        Q = self.query(A).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, 4, d_k)
        K = self.key(B).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, 12, d_k)
        V = self.value(B).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, 12, d_k)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.d_k ** 0.5  # (batch_size, n_heads, 4, 12)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, n_heads, 4, 12)

        output = torch.matmul(attn_weights, V)  # (batch_size, n_heads, 4, d_k)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch_size, 4, d_model)

        output = self.fc_out(output)  # Final linear layer to combine heads

        return output

if __name__ == '__main__':

    d_model = 128
    n_heads = 4
    A = torch.rand((1, 4, d_model))
    B = torch.rand((1, 12, d_model))

    attention_layer = MultiHeadCrossModalAttention(d_model, n_heads)
    output = attention_layer(A, B)
    print(output.shape)  # Output shape will be (1, 4, 128)