import numpy as np
import scipy
import torch

x = torch.rand(5, 3)
print(x)

print(torch.cuda.is_available())
