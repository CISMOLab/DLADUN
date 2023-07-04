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


torch.cuda.set_device(int(opt.gpu_id))
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')
model = model_bulider(config)
mask3d_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 10)
test_input_mask,test_mask_s = input_mask_test
test_data = LoadTest(opt.test_path)
model_path = './model_epoch_$(EPOCH).pth'
checkpoint = torch.load(model_path,map_location='cpu')
model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
                        strict=True)
# model.cuda()
test_gt = test_data.cuda().float()
input_meas = init_meas(test_gt,mask3d_batch_test, opt.input_setting)
model.eval()
with torch.no_grad():
    psnr_list, ssim_list = [], []
    model_out,layers_sym = model(input_meas.cpu(),test_input_mask.cpu())
    results =model_out
    end = time.time()
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(results[k, :, :, :].cuda(), test_gt[k, :, :, :].cuda())
        ssim_val = torch_ssim(results[k, :, :, :].cuda(), test_gt[k, :, :, :].cuda())
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    print('psnr = {:.2f}, ssim = {:.3f}'.format(psnr_mean, ssim_mean))