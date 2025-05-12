
import torch 
def decode_input(input, train=True):
    video = input['video']
    video_name = input['video_name']

    if train:
        inputs = video[:-1]
        target = video[-1]
        return inputs, target
        # return video, video_name
    else:   # TODO: bo sung cho test
        return video, video_name
        
def To_Batch(x,batch,frame):
  batch_data=[]
  for b in range(batch) :
    data=[]
    for i in range(frame):
      data.append(x[i][b])
    batch_data.append(data)  
  return batch_data 


def To_Frame(x,batch,frame):
  output=[]
  for i in range(frame):
    zeros=torch.zeros([batch,x[0][0].shape[0],x[0][0].shape[1],x[0][0].shape[2]])
    for b in range(batch):

      zeros[b]=x[b][i]
    output.append(zeros)
  return output 


  