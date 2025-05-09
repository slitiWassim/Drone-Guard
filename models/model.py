import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import Efficientnet_X3D
from models.basic_modules import ConvBnRelu, ConvTransposeBnRelu, initialize_weights
from grouped_query_attention_pytorch.attention import MultiheadGQA
from models.Quantizer import Quantizer
import math

logger = logging.getLogger(__name__)
embedding_dim = 128


class DroneGuard(nn.Module):
    def get_name(self):
        return self.model_name

    def __init__(self, config, pretrained=True):
        super(DroneGuard, self).__init__()
        
        
        frames = config.MODEL.ENCODED_FRAMES
        final_conv_kernel = config.MODEL.EXTRA.FINAL_CONV_KERNEL
        self.model_name = config.MODEL.NAME
        logger.info('=> ' + self.model_name +' in Action' )
        self.batch=config.TRAIN.BATCH_SIZE_PER_GPU
        self._commitment_cost = 0.25

        channels = [192,96,48,24]
        
        
        
        '''   --   Encoder  --   '''
        # Encoder Backbone 
        self.Efficientnet_X3D = Efficientnet_X3D()


        # Grouped Query Self-attention (GQA)
        self.attn = MultiheadGQA(embed_dim=embedding_dim, query_heads=4, kv_heads=2, device="cuda")
    
        # Encoder Conv  Layers
        self.conv_x8 = conv_block(ch_in=channels[2] * frames, ch_out=channels[3])
        self.conv_x2 = conv_block(ch_in=channels[3] * frames, ch_out=channels[3])
        self.conv_x0 = conv_block(ch_in=channels[3] * frames, ch_out=channels[3])
        
        self.conv8 = nn.Conv2d(in_channels=channels[3],out_channels=embedding_dim,kernel_size=1,bias=False)
        self.conv2 = nn.Conv2d(in_channels=channels[3], out_channels=embedding_dim,kernel_size=1,bias=False)
        self.conv0 = nn.Conv2d(in_channels=channels[3],out_channels=embedding_dim,kernel_size=1,bias=False)
        ''' --- End of Encoder --- '''
        


        # Quantization Layer
        self.vq = Quantizer(channels[3],codebook_size =128,Quantizer_name='ResidualVQ')
        self._pre_vq_conv = nn.Conv2d(in_channels=embedding_dim, out_channels=channels[3],kernel_size=1)
      

        '''   --   Decoder  --   '''
        # DeConv + BN + ReLu
        self.up8 = ConvTransposeBnRelu(channels[3], channels[3] ,kernel_size=2)   
        self.up4 = ConvTransposeBnRelu(channels[3]+channels[3], channels[3], kernel_size=2)   
        self.up2 = ConvTransposeBnRelu(channels[3]+channels[3], channels[3], kernel_size=2)   

        self.final = nn.Sequential(
            ConvBnRelu(channels[3], channels[2], kernel_size=1, padding=0),
            ConvBnRelu(channels[2], channels[3], kernel_size=3, padding=1),
            nn.Conv2d(channels[3], 3,
                      kernel_size=final_conv_kernel,
                      padding=1 if final_conv_kernel == 3 else 0,
                      bias=False)
        )

        ''' --- End of Decoder --- '''



        # Initialize  Layers  Weights
        initialize_weights(self.conv_x0, self.conv_x2, self.conv_x8)
        initialize_weights(self.conv8, self.conv2, self.conv0)
        initialize_weights(self.attn)
        initialize_weights(self._pre_vq_conv)
        initialize_weights(self.vq)
        initialize_weights(self.up2, self.up4, self.up8)
        initialize_weights(self.final)

    def forward(self, x):

        '''   --   Encoder  Part  --   '''

        x0,x1,x2=self.Efficientnet_X3D(torch.stack(x,dim=2))

        # Reshape the features from various backbone stages
        x0=x0.view(x0.shape[0], -1, x0.shape[-2], x0.shape[-1])
        x1=x1.view(x1.shape[0], -1, x1.shape[-2], x1.shape[-1])
        x2=x2.view(x2.shape[0], -1, x2.shape[-2], x2.shape[-1])


        x8 = self.conv_x8(x2)
        x2 = self.conv_x2(x1)
        x0 = self.conv_x0(x0)

        # Conv pre GQA 

        x8_ = self.conv8(x8) 
        x2_ = F.interpolate(self.conv2(x2), scale_factor=0.5, mode='bilinear', align_corners=False) 
        x0_ = F.interpolate(self.conv0(x0), scale_factor=0.25, mode='bilinear', align_corners=False) 
        x0_shape = x0_.shape
        
        # Multi Scale Grouped Query Attention Block 
        x8 = x8_.view(x0_shape[0],-1,embedding_dim) # V
        x2_ = x2_.view(x0_shape[0],-1,embedding_dim) # K
        x0_ = x0_.view(x0_shape[0],-1,embedding_dim) # Q
        x8 = self.attn(x0_,x2_,x8)[0].view(x0_shape) 
        
        ''' --- End of Encoder Part --- '''


        ## Apply Quantization to the Encoder output
        x8 =self._pre_vq_conv(x8)
        x8 , _ =self.vq(x8)    # During training, retrieve the _loss_commit to optimize the model.



        '''   --   Decoder  Part  --   '''

        x = self.up8(x8) 
        x = self.up4(torch.cat([x2, x], dim=1))
        x = self.up2(torch.cat([x0, x], dim=1))

        ''' --- End of Decoder Part --- '''


        return self.final(x)
    

    def compute_loss(self, x):

        '''   --   Encoder  Part  --   '''

        x0,x1,x2=self.Efficientnet_X3D(torch.stack(x,dim=2))

        # Reshape the features from various backbone stages
        x0=x0.view(x0.shape[0], -1, x0.shape[-2], x0.shape[-1])
        x1=x1.view(x1.shape[0], -1, x1.shape[-2], x1.shape[-1])
        x2=x2.view(x2.shape[0], -1, x2.shape[-2], x2.shape[-1])


        x8 = self.conv_x8(x2)
        x2 = self.conv_x2(x1)
        x0 = self.conv_x0(x0)

        # Conv pre GQA 

        x8_ = self.conv8(x8) 
        x2_ = F.interpolate(self.conv2(x2), scale_factor=0.5, mode='bilinear', align_corners=False) 
        x0_ = F.interpolate(self.conv0(x0), scale_factor=0.25, mode='bilinear', align_corners=False) 
        x0_shape = x0_.shape
        
        # Multi Scale Grouped Query Attention Block 
        x8 = x8_.view(x0_shape[0],-1,embedding_dim) # V
        x2_ = x2_.view(x0_shape[0],-1,embedding_dim) # K
        x0_ = x0_.view(x0_shape[0],-1,embedding_dim) # Q
        x8 = self.attn(x0_,x2_,x8)[0].view(x0_shape) 
        
        ''' --- End of Encoder Part --- '''
        
        ## Apply Quantization to the Encoder output
        x8 = self._pre_vq_conv(x8)
        x8 , _loss_commit = self.vq(x8) # During training, retrieve the _loss_commit to optimize the model.

        
        '''   --   Decoder  Part  --   '''

        x = self.up8(x8) 
        x = self.up4(torch.cat([x2, x], dim=1))
        x = self.up2(torch.cat([x0, x], dim=1))

        ''' --- End of Decoder Part --- '''

        loss_commit = self._commitment_cost * _loss_commit 

        
        return self.final(x) ,loss_commit
    




class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.

class SEModule_small(nn.Module):
    def __init__(self, channel):
        super(SEModule_small, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y


class conv_dy(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, padding):
        super(conv_dy, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.dim = int(math.sqrt(inplanes))
        squeeze = max(inplanes, self.dim ** 2) // 16

        self.q = nn.Conv2d(inplanes, self.dim, 1, stride, 0, bias=False)

        self.p = nn.Conv2d(self.dim, planes, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(inplanes, squeeze, bias=False),
            SEModule_small(squeeze),
        )
        self.fc_phi = nn.Linear(squeeze, self.dim ** 2, bias=False)
        self.fc_scale = nn.Linear(squeeze, planes, bias=False)
        self.hs = Hsigmoid()

    def forward(self, x):
        r = self.conv(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        phi = self.fc_phi(y).view(b, self.dim, self.dim)
        scale = self.hs(self.fc_scale(y)).view(b, -1, 1, 1)
        r = scale.expand_as(r) * r

        out = self.bn1(self.q(x))
        _, _, h, w = out.size()

        out = out.view(b, self.dim, -1)
        out = self.bn2(torch.matmul(phi, out)) + out
        out = out.view(b, -1, h, w)
        out = self.p(out) + r
        return out


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            conv_dy(ch_out, ch_out, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

