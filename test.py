import torch
print("PyTorch version:", torch.__version__)
print("Is ROCm available:", torch.version.hip is not None)
print("Is CUDA available (NVIDIA only):", torch.cuda.is_available())
print("Torch device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
