import torch

print("PyTorch version:", torch.__version__)
print("ROCm support:", torch.version.hip is not None)

if torch.version.hip is not None:
    print("ROCm device is available.")
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())
else:
    print("No ROCm-compatible device detected.")
