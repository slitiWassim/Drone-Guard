import math
import numpy as np
from sklearn import metrics
import os

def psnr_park(mse):
    return 10 * math.log10(1 / mse)


def anomaly_score(psnr, max_psnr, min_psnr):
    return (psnr - min_psnr) / (max_psnr - min_psnr)


def calculate_auc(config, psnr_list, mat):
    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df  # number of frames to process

    scores = np.array([], dtype=np.float64)
    labels = np.array([], dtype=np.int64)

    for i in range(len(psnr_list)):
        score = anomaly_score(psnr_list[i], np.max(psnr_list[i]), np.min(psnr_list[i]))

        scores = np.concatenate((scores, score), axis=0)
        labels = np.concatenate((labels, mat[i][fp:]), axis=0)
    assert scores.shape == labels.shape, f'Ground truth has {labels.shape[0]} frames, BUT got {scores.shape[0]} detected frames!'
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores,pos_label=0)
    auc = metrics.auc(fpr, tpr)

    return auc, fpr, tpr


def get_labels(DATASET):
    frame_path='./Datasets/labels/frame_labels_'+DATASET+'.npy' 
    folder='./Datasets/'+DATASET+'/testing/frames'
    label=np.load(frame_path)
    nb=0
    mat=[]
    folder_list=os.listdir(folder)
    folder_list.sort()
    for dr in folder_list:
      length=len(os.listdir(folder+'/'+dr))
      mat.append(label[0][nb:nb+length])
      nb+=length    

    return mat 

    

def get_drone_labels(DATASET):
    label_folder = './Datasets/'+DATASET+'/annotation'
    mat=[]
    label_list=os.listdir(label_folder)
    label_list.sort()
    

    for file in label_list:
      annotation = np.load(os.path.join(label_folder,file))
      mat.append(annotation)
       
    return mat 