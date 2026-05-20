from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    f_output = dense_layer(output)
    f_output = dropout(f_output)
    return input + f_output


  def forward(self, hidden_states, attention_mask):
    # Attention Part
    attention_norm = self.attention_layer_norm(hidden_states)
    attention_f_out = self.self_attention.forward(attention_norm, attention_mask)
    attention_p_out = self.add(hidden_states, attention_f_out, self.attention_dense, self.attention_dropout)
    
    # MLP Part
    mlp_norm = self.out_layer_norm(attention_p_out)
    mlp_f_out = self.interm_af(self.interm_dense(mlp_norm))
    mlp_p_out = self.add(attention_p_out, mlp_f_out, self.out_dense, self.out_dropout)
    
    return mlp_p_out
