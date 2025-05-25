import torch
import torch.nn as nn

class SimpleRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS (Root Mean Square)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
        # Normalize and scale
        x_norm = x / (rms + self.eps)
        return x_norm * self.weight

def compare_rmsnorm():
    # Create input tensor
    batch_size = 32
    seq_len = 128
    hidden_dim = 512
    x = torch.randn(batch_size, seq_len, hidden_dim)
    x = x.to(device)
    
    # Initialize both implementations
    simple_norm = SimpleRMSNorm(hidden_dim)
    builtin_norm = nn.RMSNorm(hidden_dim)

    simple_norm = simple_norm.to(device)
    builtin_norm = builtin_norm.to(device)
    
    # Get outputs
    out_simple = simple_norm(x)
    out_builtin = builtin_norm(x)
    
    # Compare numerical differences
    max_diff = torch.max(torch.abs(out_simple - out_builtin))
    mean_diff = torch.mean(torch.abs(out_simple - out_builtin))
    print(f"Maximum difference between outputs: {max_diff:.6f}")
    print(f"Mean difference between outputs: {mean_diff:.6f}")

if __name__ == "__main__":
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run comparison
    compare_rmsnorm() 