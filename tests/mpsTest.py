import torch
print(torch.backends.mps.is_available())  # should be True
print(torch.backends.mps.is_built())      # should be True