import torch
import torch.nn as nn

def estimate_vram_usage(model, input_size, batch_size, dtype=torch.float32, training=True):
    """
    Estimate the VRAM usage of a PyTorch model.

    :param model: PyTorch model
    :param input_size: Size of the input (C, H, W)
    :param batch_size: Batch size
    :param dtype: Data type of the model parameters and inputs (default: torch.float32)
    :param training: Flag indicating if the model is in training mode (default: True)
    :return: Estimated VRAM usage in megabytes (MB)
    """
    # Set model to the appropriate mode
    model.train(mode=training)

    # Create a dummy input tensor
    dummy_input = torch.rand(size=(batch_size,) + input_size, dtype=dtype)

    # Forward pass
    with torch.no_grad():
        if training:
            # Enable gradient calculation
            dummy_input.requires_grad = True

        output = model(dummy_input)

    # Estimate memory usage
    param_size = sum(p.numel() for p in model.parameters()
                     if p.requires_grad) * dtype().itemsize
    input_size = dummy_input.nelement() * dtype().itemsize
    output_size = output.nelement() * dtype().itemsize
    additional_overhead = param_size * \
        2 if training else param_size  # For gradients and optimizers

    total_memory = (param_size + input_size + output_size +
                    additional_overhead) / (1024 ** 2)  # Convert to MB

    return total_memory

if __name__ == "__main__":
    # Example usage
    model = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    input_size = (3, 224, 224)  # Example input size (C, H, W)
    batch_size = 32  # Example batch size

    vram_usage = estimate_vram_usage(
        model, input_size, batch_size, dtype=torch.float32, training=True)
    print(f"Estimated VRAM Usage: {vram_usage:.2f} MB")