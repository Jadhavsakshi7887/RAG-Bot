import torch
print(torch.cuda.is_available())   # True means GPU is available
print(torch.cuda.device_count())   # Number of GPUs
print(torch.cuda.get_device_name(0))  # GPU name
