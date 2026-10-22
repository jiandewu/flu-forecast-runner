import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)
    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=1)
        return torch.sum(weights * x, dim=1), weights

class HybridCNNForecaster(nn.Module):
    """Parallel multi-scale CNN (kernels 3 and 5), bidirectional LSTM, two softmax attention
    pooling heads and one GELU output head per horizon.
    """
    def __init__(self, input_size, hidden_size, output_steps, num_quantiles, num_layers=2, dropout=0.4):
        super().__init__()
        self.output_steps = output_steps
        self.num_quantiles = num_quantiles
        
        # Parallel Inception-style CNN: Captures both weekly jitter (k=3) and monthly trend (k=5)
        self.conv3 = nn.Conv1d(input_size, hidden_size // 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_size, hidden_size // 2, kernel_size=5, padding=2)
        
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        
        self.mag_attn = Attention(hidden_size * 2)
        self.phase_attn = Attention(hidden_size * 2)
        
        self.out = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 4, hidden_size),
                nn.GELU(), # Smoother than ReLU
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_quantiles)
            ) for _ in range(output_steps)
        ])

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x_in = x.transpose(1, 2)
        
        # Multi-scale feature extraction
        x3 = F.gelu(self.conv3(x_in))
        x5 = F.gelu(self.conv5(x_in))
        
        # Concatenate scale features
        x_conv = torch.cat([x3, x5], dim=1).transpose(1, 2) # (batch, seq_len, hidden_size)
        
        lstm_out, _ = self.lstm(x_conv)
        
        mag_context, _ = self.mag_attn(lstm_out)
        phase_context, _ = self.phase_attn(lstm_out)
        
        combined_context = torch.cat([mag_context, phase_context], dim=-1)
        
        preds = []
        for predictor in self.out:
            preds.append(predictor(combined_context))
            
        return torch.stack(preds, dim=1)
