#Author: Akash
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class ConvCapsule(nn.Module):
    def __init__(self,caps_num,caps_dim,k_size=3,stride=1,padding=0,routing_num=3):
        super(ConvCapsule,self).__init__()
        self.k_size = k_size
        self.caps_num = caps_num
        self.caps_dim = caps_dim
        self.stride = stride
        self.padding = padding
        self.routing_num = routing_num

    def squash(self,input_tensor):
        norm_squared = torch.sum(input_tensor**2, dim=2, keepdim=True)

        return (input_tensor / torch.sqrt(norm_squared)) * (norm_squared / (1 + norm_squared))

    def forward(self,inp):

        #[batch_size,inp_caps_num,inp_caps_dim,height,width]
        inp_shape = inp.size()
        assert len(inp_shape) == 5, "The input Tensor should have shape=[batch_size,inp_caps_num,inp_caps_dim,height,width]"

        self.batch_size = inp_shape[0]
        self.inp_caps_num = inp_shape[1]
        self.inp_caps_dim = inp_shape[2]
        self.height = inp_shape[3]
        self.width = inp_shape[4]
        softmax = nn.Softmax(dim=2)

        #[batch_size*inp_caps_num,inp_caps_dim,height,width]
        inp_reshaped = inp.view(self.batch_size*self.inp_caps_num,self.inp_caps_dim,self.height,self.width)
        conv = nn.Conv2d(self.inp_caps_dim,self.caps_dim*self.caps_num,stride=self.stride,kernel_size=self.k_size,padding=self.padding)

        #[batch_size*inp_caps_num, caps_dim*caps_num, conv_height,conv_width]
        conv_op = conv(inp_reshaped)
        conv_op_shape = conv_op.size()
        self.conv_height = conv_op_shape[2]
        self.conv_width = conv_op_shape[3]

        #[batch_size, inp_caps_num, caps_num, caps_dim, conv_height,conv_width]
        conv_op_reshaped = conv_op.view(self.batch_size,self.inp_caps_num,self.caps_num,self.caps_dim,self.conv_height,self.conv_width)

        self.lol = 0.1*torch.ones(self.caps_num,self.caps_dim,1, 1)
        #[caps_num, caps_dim, conv_height, conv_width]
        self.lol = self.lol.expand(self.caps_num,self.caps_dim,conv_op_shape[2],conv_op_shape[3])
        #self.bias_ = Parameter(torch.Tensor(self.b))

        #[batch_size, inp_caps_num, caps_num, conv_height, conv_width]
        logits = torch.zeros(self.batch_size,self.inp_caps_num,self.caps_num,self.conv_height,self.conv_width)
        #logits = Variable(logits)

        for iteration in range(self.routing_num):

            #[batch_size, inp_caps_num, caps_num, conv_height, conv_width]
            logits_soft = softmax(logits)

            #[batch_size, caps_num, caps_dim, conv_height, conv_width]
            pre_activation = torch.sum(logits.unsqueeze(3)*conv_op_reshaped,1)
            pre_activation = pre_activation + self.lol
            activation = self.squash(pre_activation)

            #[batch_size, inp_caps_num, caps_num, conv_height, conv_width]
            distance = torch.sum(activation.unsqueeze(1)*conv_op_reshaped,3)
            logits = logits + distance
        return activation

class DeConvCapsule(nn.Module):
    def __init__(self,caps_num,caps_dim,k_size=3,stride=1,padding=0,routing_num=3):
        super(ConvCapsule,self).__init__()
        self.k_size = k_size
        self.caps_num = caps_num
        self.caps_dim = caps_dim
        self.stride = stride
        self.padding = padding
        self.routing_num = routing_num

    def squash(self,input_tensor):
        norm_squared = torch.sum(input_tensor**2, dim=2, keepdim=True)

        return (input_tensor / torch.sqrt(norm_squared)) * (norm_squared / (1 + norm_squared))

    def forward(self,inp):

        #[batch_size,inp_caps_num,inp_caps_dim,height,width]
        inp_shape = inp.size()
        assert len(inp_shape) == 5, "The input Tensor should have shape=[batch_size,inp_caps_num,inp_caps_dim,height,width]"

        self.batch_size = inp_shape[0]
        self.inp_caps_num = inp_shape[1]
        self.inp_caps_dim = inp_shape[2]
        self.height = inp_shape[3]
        self.width = inp_shape[4]
        softmax = nn.Softmax(dim=2)

        #[batch_size*inp_caps_num,inp_caps_dim,height,width]
        inp_reshaped = inp.view(-1,*inp_shape[2:])
        conv = nn.ConvTranspose2d(self.inp_caps_dim,self.caps_dim*self.caps_num,stride=self.stride,kernel_size=self.k_size,padding=self.padding)

        #[batch_size*inp_caps_num, caps_dim*caps_num, height,width]
        conv_op = conv(inp_reshaped)
        conv_op_shape = conv_op.size()

        #[batch_size, inp_caps_num, caps_num, caps_dim, height,width]
        conv_op_reshaped = conv_op.view(self.batch_size,self.inp_caps_num,self.caps_num,self.caps_dim,conv_op_shape[2],conv_op_shape[3])

        self.b = 0.1*torch.ones(self.caps_num,self.caps_dim,1, 1)
        self.b = self.b.expand(self.caps_num,self.caps_dim,conv_op_shape[2],conv_op_shape[3])
        self.bias = nn.Parameter(self.b)
        #[batch_size, inp_caps_num, caps_num, height,width]
        logits = torch.zeros(self.batch_size,self.inp_caps_num,self.caps_num,conv_op_shape[2],conv_op_shape[3])
        logits = Variable(logits)
        for iteration in range(self.routing_num):

            logits_soft = softmax(logits)

            #[batch_size, inp_caps_num, caps_num, (1)caps_dim, height,width]
            pre_activation = torch.sum(torch.unsqueeze(logits,3)*conv_op_reshaped,1)
            pre_activation = pre_activation + self.bias
            activation = self.squash(pre_activation)
            distance = torch.sum(activation.unsqueeze(1)*conv_op_reshaped,3)
            logits += distance
        return activation

if __name__ == '__main__':

    inp = torch.randn(4,32,16,64,64)
    conv1 = ConvCapsule(10,8,3,1,routing_num=3)
    out = conv1(inp)
    print out.shape

