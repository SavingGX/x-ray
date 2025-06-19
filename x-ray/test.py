import torch

# 检查torch是否有CUDA支持，即是否能用GPU
print(torch.cuda.is_available())

# 如果CUDA可用，它还会打印出当前默认的CUDA设备（通常是第一个GPU）
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
print(torch.version.cuda)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
