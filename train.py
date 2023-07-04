# from architecture import *
from arch.NetMaker import model_bulider
from utils.utils import *
import torch
import scipy.io as scio
import time
import os
import numpy as np
from torch.autograd import Variable
import datetime
from option.option import opt,config
import torch.nn.functional as F
from tqdm import tqdm
# import pytorch_warmup as warmup

torch.cuda.set_device(int(opt.gpu_id))
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')


# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + str(opt.epoch_sam_num) + '/' + config['model_type']+ '-' + 'STAGE' + '-' + str(config['stage']) + '-' + date_time + '/result/'
model_path = opt.outf + str(opt.epoch_sam_num)  + '/' + config['model_type']+ '-' + 'STAGE' + '-' + str(config['stage']) + '-' + date_time + '/model/'

if opt.RESUME:
    model_path = opt.re_path[0]
    result_path = opt.re_path[1]

if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

logger = gen_log(model_path)
logger.info("\n trainSetting:{}\n model config:{}\n".format(opt, config))

# model
model = model_bulider(config).cuda()

# init mask
mask3d_batch_train, input_mask_train = init_mask(opt.mask_path, opt.input_mask, opt.batch_size)
mask3d_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 10)
if config['model_type'] in ['DDUDSU']:
    train_input_mask,train_mask_s = input_mask_train
    test_input_mask,test_mask_s = input_mask_test
# dataset
train_set = LoadTraining(opt.data_path)
test_data = LoadTest(opt.test_path)


# optimizing
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
if opt.scheduler=='MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler=='CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)
if config['model_type'] in ['DDUDSU']:
    lf = lossFuc().cuda()
    print("DDUDSU loss")
else:
    if opt.loss_type=='L2':
        mse = torch.nn.MSELoss().cuda()
        print("L2 loss")
    elif opt.loss_type=='ML2':
        mmse = mutilstage_loss().cuda()

def train(epoch, logger):
    epoch_loss = 0
    begin = time.time()
    batch_num = int(np.floor(opt.epoch_sam_num / opt.batch_size))
    train_logger = tqdm(range(batch_num))
    for i in train_logger:
        gt_batch = shuffle_crop(train_set, opt.batch_size)
        gt = Variable(gt_batch).cuda().float()
        input_meas = init_meas(gt, mask3d_batch_train, opt.input_setting)
        optimizer.zero_grad()
        if config['model_type'] in ['DDUDSU']:
            model_out,layers_sym = model(input_meas,train_input_mask,train_mask_s)
        else:
            model_out = model(input_meas,train_input_mask,train_mask_s)

        if config['model_type'] in ['DDUDSU']:
            loss = lf(model_out,layers_sym,gt)
        else:
            loss = torch.sqrt(mse(model_out, gt))

        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
        train_logger.set_description(desc='[epoch: %d][lr: %.6f][loss: %.6f][mean_loss: %.6f]'%(epoch,scheduler.get_last_lr()[0],loss,epoch_loss / (i+1)))
    end = time.time()
    logger.info("===> Epoch {} Complete: lr:{:.6f} Avg. Loss: {:.6f} time: {:.2f}".format(epoch,scheduler.get_last_lr()[0],epoch_loss / batch_num, (end - begin)))
    return 0

def test(epoch, logger):
    psnr_list, ssim_list = [], []
    test_gt = test_data.cuda().float()
    input_meas = init_meas(test_gt, mask3d_batch_test, opt.input_setting)
    model.eval()
    begin = time.time()
    with torch.no_grad():
        if config['model_type'] in ['DDUDSU']:
            model_out,layers_sym = model(input_meas,test_input_mask,test_mask_s)
        else:
            model_out = model(input_meas,test_input_mask,test_mask_s)
    end = time.time()
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'
                .format(epoch, psnr_mean, ssim_mean,(end - begin)))
    model.train()
    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean

def main():
    psnr_max = 0
    start_epoch = 0
    if opt.RESUME:
        path_checkpoint = os.path.join(model_path, 'mycheckpoint.pth')  #
        recheckpoint = torch.load(path_checkpoint)  # 

        model.load_state_dict(recheckpoint['net'])  # 

        optimizer.load_state_dict(recheckpoint['optimizer'])  # 
        start_epoch = recheckpoint['epoch']  # 
    for epoch in range(start_epoch+1, opt.max_epoch + 1):
        train(epoch, logger)
        (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(epoch, logger)
        scheduler.step()
        mycheckpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(mycheckpoint, os.path.join(model_path, 'mycheckpoint.pth'))
        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            if psnr_mean > 28:
                name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_max, ssim_mean) + '.mat'
                scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all})
                checkpoint(model, epoch, model_path, logger)

if __name__ == '__main__':
    main()