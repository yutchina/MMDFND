import  torch

from model.layers import clip_fuion

a = torch.rand(64,1024)
clip_fusion = clip_fuion(1024, 320, [348], 0.1)
clip_fusion(a)