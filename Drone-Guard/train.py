import os
import pprint
import argparse
import torch.backends.cudnn as cudnn
import sys
import numpy as np
import torch
import torch.nn as nn 
from test import inference
from config.defaults import _C as config, update_config
from utils import train_util, log_util, loss_util, optimizer_util, anomaly_util
import models as models
from models.model import DroneGuard as get_model
import Datasets


def parse_args():

    parser = argparse.ArgumentParser(description='Drone Guard')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        default='config/shanghaitech_wresnet.yaml', type=str)
    parser.add_argument('--model-file', help='model parameters',
                        default='pretrained/shanghaitech.pth', type=str)
    parser.add_argument('--resume-train', help='Resume Training',
                        default=False, type=bool) 
    parser.add_argument('--pseudo', help='Pseudo Annomalies Option',
                        default=False, type=bool)  
    parser.add_argument('--output-folder', help='Output Folder',
                        default=None)
    parser.add_argument('--test', help='Testing Model While Training',
                        default=False,type=bool)                                                                                                     
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        log_util.create_logger(config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    
    model = get_model(config)
    logger.info('Model: {}'.format(model.get_name()))

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    if(args.resume_train):
      state_dict = torch.load(args.model_file)
      if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
      else:
        model.module.load_state_dict(state_dict)
    losses = loss_util.MultiLossFunction(config=config).cuda()

    optimizer = optimizer_util.get_optimizer(config, model)

    scheduler = optimizer_util.get_scheduler(config, optimizer)

    train_dataset = eval('Datasets.get_data')(config)
    if(args.pseudo):
        train_dataset_jump=eval('Datasets.get_jump_data')(config)
        train_loader_jump = torch.utils.data.DataLoader(
            train_dataset_jump,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=True,
            drop_last=True
        )
    if(args.test):
        test_dataset = eval('Datasets.get_test_data')(config)
        BATCH_SIZE_PER_GPU_TEST=1
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE_PER_GPU_TEST* len(gpus),
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=True
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True
    )
    logger.info('Number videos: {}'.format(len(train_dataset)))
    pseudolosscounter = 1
    pseudolossepoch = 0
    last_epoch = config.TRAIN.BEGIN_EPOCH
    
    if(args.test):
        AUC_memory= 0.0
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        if(args.pseudo):
          train_pseudo(config, train_loader,train_loader_jump, model, losses, optimizer, epoch, logger,pseudolossepoch, pseudolosscounter,args)
        else:
          train(config, train_loader, model, losses, optimizer, epoch, logger,args)

        scheduler.step()

        if (epoch + 1) % config.SAVE_CHECKPOINT_FREQ == 0:
            logger.info('=> saving model state epoch_{}.pth to {}\n'.format(epoch+1, final_output_dir))
            torch.save(model.module.state_dict(), os.path.join(final_output_dir,
                                                               'epoch_{}.pth'.format(epoch + 1)))
        
        if (epoch % 10 ==0) and ( args.output_folder != None):
            torch.save(model.module.state_dict(), os.path.join(args.output_folder,
                                                               'epoch_train_{}.pth'.format(epoch + 1)))
        if(args.test)and(epoch > 15):
            if config.DATASET.TYPE == "ground":
               mat=anomaly_util.get_labels(config.DATASET.DATASET)
            else :
               mat=anomaly_util.get_drone_labels(config.DATASET.DATASET)

            psnr_list = inference(config, test_loader, model,args,quiet=True)
            assert len(psnr_list) == len(mat), f'Ground truth has {len(mat)} videos, BUT got {len(psnr_list)} detected videos!'
            auc, fpr, tpr = anomaly_util.calculate_auc(config, psnr_list, mat)
            if(auc>AUC_memory):
                logger.info('=> saving model state epoch_{}.pth as best AUC to {}\n'.format(epoch+1, final_output_dir))
                torch.save(model.module.state_dict(), os.path.join(final_output_dir,
                                                               'best_AUC.pth'.format(epoch + 1)))
                if( args.output_folder != None):
                    torch.save(model.module.state_dict(), os.path.join(args.output_folder,
                                                               'best_AUC.pth'.format(epoch + 1)))                                               
                AUC_memory=auc
            logger.info(f'AUC: {auc * 100:.1f}%  ')

    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)






def train(config, train_loader, model, loss_functions, optimizer, epoch, logger,args):
    loss_func_mse = nn.MSELoss(reduction='none')

    model.train()

    for i, data in enumerate(train_loader):
        # decode input
        
        inputs,target = train_util.decode_input(input=data, train=True)
        inputs = [input.cuda() for input in inputs]
        output,loss_commit = model.module.compute_loss(inputs)

        # compute loss
        target = target.cuda(non_blocking=True)
        inte_loss, grad_loss, msssim_loss, l2_loss = loss_functions(output, target)
        loss = inte_loss + grad_loss + msssim_loss + l2_loss + loss_commit

        # compute PSNR
        mse_imgs = torch.mean(loss_func_mse((output + 1) / 2, (target + 1) / 2)).item()
        psnr = anomaly_util.psnr_park(mse_imgs)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_lr = optimizer.param_groups[0]['lr']
        if (i + 1) % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Lr {lr:.6f}\t' \
                  '[inte {inte:.5f} + grad {grad:.4f} + msssim {msssim:.4f} + L2 {l2:.4f} + commit {loss_commit:.3f}]\t' \
                  'PSNR {psnr:.2f}'.format(epoch+1, i+1, len(train_loader),
                                             lr=cur_lr,
                                             inte=inte_loss, grad=grad_loss, msssim=msssim_loss, l2=l2_loss,loss_commit=loss_commit,
                                             psnr=psnr)
            logger.info(msg)

def train_pseudo(config, train_loader,train_loader_jump, model, loss_functions, optimizer, epoch, logger,pseudolossepoch, pseudolosscounter,args):
    loss_func_mse = nn.MSELoss(reduction='none')

    model.train()

    for i, (data,data_jump) in enumerate(zip(train_loader,train_loader_jump)):
        # decode input

        # images 
        inputs, target = train_util.decode_input(input=data, train=True)
        inputs_jump, target_jump = train_util.decode_input(input=data_jump, train=True)

        #print(len(target))
        #print(target.shape)
         
        #Batch  Data
        inputs_batch=train_util.To_Batch(inputs,config.TRAIN.BATCH_SIZE_PER_GPU * len(list(config.GPUS)),config.MODEL.ENCODED_FRAMES)
        inputs_batch_jump=train_util.To_Batch(inputs_jump,config.TRAIN.BATCH_SIZE_PER_GPU * len(list(config.GPUS)),config.MODEL.ENCODED_FRAMES)
        
       # target_batch=To_Batch(target,config.TRAIN.BATCH_SIZE_PER_GPU * len(list(config.GPUS)),config.MODEL.ENCODED_FRAMES)
       # target_batch_jump=To_Batch(target_jump,config.TRAIN.BATCH_SIZE_PER_GPU * len(list(config.GPUS)),config.MODEL.ENCODED_FRAMES)


        jump_pseudo_stat = []
        cls_labels = []
        for b in range(config.TRAIN.BATCH_SIZE_PER_GPU * len(list(config.GPUS))):
          total_pseudo_prob = 0
          rand_number = np.random.rand()
          pseudo_bool = False
          # skip frame pseudo anomaly
          #pseudo_anomaly_jump = total_pseudo_prob <= rand_number < total_pseudo_prob + args.pseudo_anomaly_jump
          #total_pseudo_prob += args.pseudo_anomaly_jump
          pseudo_anomaly_jump = total_pseudo_prob <= rand_number < total_pseudo_prob + config.DATASET.PSEUDO_ANOMALY_PROB
          total_pseudo_prob += config.DATASET.PSEUDO_ANOMALY_PROB
          if pseudo_anomaly_jump:
            inputs_batch[b] = inputs_batch_jump[b]
            target[b] = target_jump[b]
            jump_pseudo_stat.append(True)
            pseudo_bool = True
          else:
            jump_pseudo_stat.append(False)

          if pseudo_bool:
            cls_labels.append(0)
          else:
            cls_labels.append(1)
        cls_labels = torch.Tensor(cls_labels).unsqueeze(1).cuda()
        
        # Frame Data 
        inputs=train_util.To_Frame(inputs_batch,config.TRAIN.BATCH_SIZE_PER_GPU * len(list(config.GPUS)),config.MODEL.ENCODED_FRAMES)
        inputs = [input.cuda() for input in inputs]
        output,loss_commit_ = model.module.compute_loss(inputs)

        #target=To_Frame(target_batch,config.TRAIN.BATCH_SIZE_PER_GPU * len(list(config.GPUS)),config.MODEL.ENCODED_FRAMES)
        target = target.cuda(non_blocking=True)
       # inferance 
      
        

      # loss calculation 
        modified_inte_loss ,modified_grad_loss,modified_msssim_loss,modified_l2_loss,modified_loss_commit  = [],[],[],[],[]
        for b in range(config.TRAIN.BATCH_SIZE_PER_GPU * len(list(config.GPUS))):
          inte_loss_b, grad_loss_b, msssim_loss_b, l2_loss_b = loss_functions(output[b].unsqueeze(0), target[b].unsqueeze(0))
          if jump_pseudo_stat[b]:
            modified_inte_loss.append(-inte_loss_b)
            modified_grad_loss.append(-grad_loss_b)
            modified_msssim_loss.append(-msssim_loss_b)
            modified_l2_loss.append(-l2_loss_b)
            modified_loss_commit.append(-loss_commit_)
            pseud=inte_loss_b+grad_loss_b+msssim_loss_b+l2_loss_b+loss_commit_
            pseud /= -5
            pseudolossepoch += pseud.cpu().detach().item()
            pseudolosscounter += 1

          else:  # no pseudo anomaly
            modified_inte_loss.append(inte_loss_b)
            modified_grad_loss.append(grad_loss_b)
            modified_msssim_loss.append(msssim_loss_b)
            modified_l2_loss.append(l2_loss_b)
            modified_loss_commit.append(loss_commit_)
        
        

        inte_loss=torch.mean(torch.stack(modified_inte_loss))
        grad_loss=torch.mean(torch.stack(modified_grad_loss))
        msssim_loss=torch.mean(torch.stack(modified_msssim_loss))
        l2_loss=torch.mean(torch.stack(modified_l2_loss))
        loss_commit=torch.mean(torch.stack(modified_loss_commit))
        
        # compute loss
        loss =  inte_loss + grad_loss +  msssim_loss +  2*l2_loss  + loss_commit

        # compute PSNR

        mse_imgs = torch.mean(loss_func_mse((output + 1) / 2, (target + 1) / 2)).item()
        psnr = anomaly_util.psnr_park(mse_imgs)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_lr = optimizer.param_groups[0]['lr']
        if (i + 1) % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Lr {lr:.6f}\t' \
                  '[inte {inte:.5f} + grad {grad:.4f} + msssim {msssim:.4f} + L2 {l2:.4f} + commit {loss_commit:.3f} + pseudo {pseudo:.4f} ]\t' \
                  'PSNR {psnr:.2f}'.format(epoch+1, i+1, len(train_loader),
                                             lr=cur_lr,
                                             inte=inte_loss, grad=grad_loss, msssim=msssim_loss, l2=l2_loss,loss_commit=loss_commit,
                                             pseudo=pseudolossepoch/pseudolosscounter,
                                             psnr=psnr)
            logger.info(msg)



if __name__ == '__main__':
    main()
