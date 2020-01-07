import numpy as np
import math
import torch
x= torch.tensor([[1], [2], [3], [4]])
print(torch.unsqueeze(x, 0))