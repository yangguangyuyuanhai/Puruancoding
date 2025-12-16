import torch
from kornia.feature import LoFTR

# Choose: 'outdoor' or 'indoor'
matcher = LoFTR(pretrained="outdoor")
# torch.save(matcher.state_dict(), "./models/loftr_outdoor.pth")
