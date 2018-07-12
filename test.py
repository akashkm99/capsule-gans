import torch
from torch.autograd import Variable
from models.capsule_layers import ConvCapsule


image = torch.Tensor(16,5,4,28,28)
image_v = Variable(image)
conv = ConvCapsule(10,8,k_size=4,padding=1,stride=2)
out_v = conv(image_v)
print out_v

