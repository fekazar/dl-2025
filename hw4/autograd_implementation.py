import torch
from torch.autograd import Function

class ExpCosFunction(Function):
    @staticmethod
    def forward(ctx, x, y):
        # Save tensors needed for backward pass
        ctx.save_for_backward(x, y)
        # Forward pass computation
        exp_x = torch.exp(x)
        cos_y = torch.cos(y)
        return exp_x + cos_y

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        x, y = ctx.saved_tensors
        # Compute gradients
        grad_x = grad_output * torch.exp(x)  # ∂f/∂x = e^x
        grad_y = grad_output * -torch.sin(y)  # ∂f/∂y = -sin(y)
        return grad_x, grad_y

def compute_function_custom(x, y):
    """Compute f(x,y) = e^x + cos(y) using custom autograd Function"""
    # Convert inputs to PyTorch tensors with gradient tracking
    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    y_tensor = torch.tensor(y, dtype=torch.float32, requires_grad=True)
    
    # Apply our custom function
    result = ExpCosFunction.apply(x_tensor, y_tensor)
    
    # Backward pass
    result.backward()
    
    return {
        'value': result.item(),
        'grad_x': x_tensor.grad.item(),
        'grad_y': y_tensor.grad.item()
    }

def compute_function_torch(x, y):
    """Compute f(x,y) = e^x + cos(y) using PyTorch's native operations"""
    # Convert inputs to PyTorch tensors with gradient tracking
    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    y_tensor = torch.tensor(y, dtype=torch.float32, requires_grad=True)
    
    # Forward pass using PyTorch operations
    result = torch.exp(x_tensor) + torch.cos(y_tensor)
    
    # Backward pass
    result.backward()
    
    return {
        'value': result.item(),
        'grad_x': x_tensor.grad.item(),
        'grad_y': y_tensor.grad.item()
    }

def compare_implementations(x_val, y_val):
    """Compare custom and PyTorch implementations"""
    # Get results from both implementations
    custom_result = compute_function_custom(x_val, y_val)
    torch_result = compute_function_torch(x_val, y_val)
    
    print(f"\nComparing implementations for x={x_val}, y={y_val}:")
    print("\nFunction values:")
    print(f"Custom implementation: {custom_result['value']}")
    print(f"PyTorch implementation: {torch_result['value']}")
    print(f"Difference: {abs(custom_result['value'] - torch_result['value'])}")
    
    print("\ngradients with respect to x:")
    print(f"custom implementation: {custom_result['grad_x']}")
    print(f"pytorch implementation: {torch_result['grad_x']}")
    print(f"difference: {abs(custom_result['grad_x'] - torch_result['grad_x'])}")
    
    print("\ngradients with respect to y:")
    print(f"custom implementation: {custom_result['grad_y']}")
    print(f"pytorch implementation: {torch_result['grad_y']}")
    print(f"difference: {abs(custom_result['grad_y'] - torch_result['grad_y'])}")
    
def main():
    test_cases = [
        (1.0, torch.pi/4),
    ]
    
    for x_val, y_val in test_cases:
        compare_implementations(x_val, y_val)
        print("\n" + "="*50 + "\n") 

if __name__ == "__main__":
    main()