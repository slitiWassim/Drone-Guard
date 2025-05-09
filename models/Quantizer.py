import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize ,ResidualVQ ,GroupedResidualVQ


class Quantizer(nn.Module):

    '''

    VQ has been successfully used by Deepmind and OpenAI for high quality generation of images (VQ-VAE-2) and music (Jukebox).    
    
    In this project, we utilized a pre-existing Vector Quantization implementation from a GitHub repository :
    https://github.com/lucidrains/vector-quantize-pytorch.git
    
    
    We integrated it into our model to improve the accuracy and quality of future image reconstruction.        
    
    
    '''

    def __init__(self,embedding_dim,Quantizer_name='VectorQuantize',commitment_weight=1.,codebook_size=128,decay=0.8,num_quantizers=4):
        super(Quantizer, self).__init__()

        if(Quantizer_name=='ResidualVQ'):
            
            self.vq = ResidualVQ(
                     dim = embedding_dim,
                     codebook_size = codebook_size,         # codebook size
                     decay = decay,                         # the exponential moving average decay, lower means the dictionary will change faster
                     num_quantizers = num_quantizers,       # specify number of quantizers
                     commitment_weight = commitment_weight, # the weight on the commitment loss
                     stochastic_sample_codes = True,
                     sample_codebook_temp = 0.1,            # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
                     shared_codebook = True,                # whether to share the codebooks for all quantizers or not
                     accept_image_fmap = True              
                       )
            

        elif(Quantizer_name=='GroupedResidualVQ'):

            self.vq = GroupedResidualVQ(
                     dim = embedding_dim,
                     codebook_size = codebook_size,         # codebook size
                     decay = decay,                         # the exponential moving average decay, lower means the dictionary will change faster
                     num_quantizers = num_quantizers,       # specify number of quantizers
                     commitment_weight = commitment_weight, # the weight on the commitment loss
                     stochastic_sample_codes = True,
                     sample_codebook_temp = 0.1,            # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
                     shared_codebook = True,                # whether to share the codebooks for all quantizers or not
                     accept_image_fmap = True               
                       )
            
        else :
            self.vq = VectorQuantize(
                     dim = embedding_dim,
                     codebook_size = codebook_size,         # codebook size
                     decay = decay ,                        # the exponential moving average decay, lower means the dictionary will change faster
                     commitment_weight = commitment_weight, # the weight on the commitment loss
                     accept_image_fmap = True
                       )


    def forward(self,inputs,return_loss=True):

        quantized,_, commit_loss = self.vq(inputs)
        if return_loss:
            return quantized ,torch.mean(commit_loss)
        return quantized


    


