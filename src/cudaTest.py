import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.device_count())  # Should print the number of GPUs
print(torch.cuda.get_device_name(0)) # Prints the name of your GPU (if available)
print(torch.version.cuda) # Prints the CUDA version PyTorch is using