import os
import sys
import time
import numpy as np
import math
import argparse
import random
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import options.options as option

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from code.data.dataset import IR_Dataset
from models import create_model
from utils.util import opt_get, get_resume_paths
from data import create_dataloader



def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1024/1024)
    os.remove('temp.p')

def print_size_of_pretrained_model(model_path):
    print("Size (MB):", os.path.getsize(model_path)/1024/1024)  # return the model size in bytes
    
def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    model_path = opt_get(opt, ['model_path'], None)
    model.load_network(load_path=model_path, network=model.netG)
    return model, opt

def main():
    # Set up warnings
    import warnings
    warnings.filterwarnings(
        action='ignore',
        category=DeprecationWarning,
        module=r'.*'
    )
    warnings.filterwarnings(
        action='default',
        module=r'torch.ao.quantization'
    )

    # Specify random seed for repeatable results
    torch.manual_seed(2507)
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option YMAL file.',
                            default='/home/yufei/shulin/LLFlow_v1/code/confs/IR_smallNet.yml' if sys.platform != 'win32' else './confs/LOL_smallNet.yml')
    args = parser.parse_args()
    conf_path = '/home/yufei/shulin/LLFlow_v1/code/confs/IR_smallNet.yml'
    data_path = '/home/yufei/shulin/LLFlow_v1/IR-RGB-Dataset-resize'
    float_model_file = '/home/yufei/shulin/LLFlow_v1/experiments/IR_small_ori_v1/models/best_psnr_G.pth'
    # scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
    # scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'
    qat_model, opt = load_model(conf_path)
    qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    torch.ao.quantization.prepare_qat(qat_model, inplace=True)
    
    opt = option.parse(args.opt, is_train=True)

    
    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        resume_state_path, _ = get_resume_paths(opt)

        # distributed resuming: all load into default GPU
        if resume_state_path is None:
            resume_state = None
        else:
            device_id = torch.cuda.current_device()
            resume_state = torch.load(resume_state_path,
                                      map_location=lambda storage, loc: storage.cuda(device_id))
            option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None
    
    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    
    
    if opt['dataset'] == 'IR_Dataset':
        dataset_cls = IR_Dataset
    else:
        raise NotImplementedError()
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = dataset_cls(opt=dataset_opt, train=True, all_opt=opt)
            train_loader = create_dataloader(True, train_set, dataset_opt, opt, None)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
        elif phase == 'val':
            val_set = dataset_cls(opt=dataset_opt, train=False, all_opt=opt)
            val_loader = create_dataloader(False, val_set, dataset_opt, opt, None)
    total_iters = int(opt['train']['niter'])
    total_epochs = int(math.ceil(total_iters / train_size))
    
    
    #### create model
    current_step = 0 if resume_state is None else resume_state['iter']
    model = create_model(opt, current_step)
    print("Parameters of full network %.4f and encoder %.4f"%(sum([m.numel() for m in model.netG.parameters()])/1e6, sum([m.numel() for m in model.netG.RRDB.parameters()])/1e6))
    
    
    if resume_state:
        # logger.info('Resuming training from epoch: {}, iter: {}.'.format(
        #     resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0
        
    qat_model,opt = load_model(conf_path)
    qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    torch.ao.quantization.prepare_qat(qat_model, inplace=True)
    
    
    criterion = nn.CrossEntropyLoss()
    float_model = load_model(float_model_file).to('cpu')
    optimizer = torch.optim.SGD(float_model.parameters(), lr = 0.0001)
    float_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    torch.ao.quantization.prepare_qat(float_model, inplace=True)
    
    # Next, we'll "fuse modules"; this can both make the model faster by saving on memory access
    # while also improving numerical accuracy. While this can be used with any model, this is
    # especially common with quantized models.

    print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)
    float_model.eval()

    # Fuses modules
    float_model.fuse_model()

    # Note fusion of Conv+BN+Relu and Conv+Relu
    print('\n Inverted Residual Block: After fusion\n\n',float_model.features[1].conv)

    num_eval_batches = 1000

    print("Size of baseline model")
    print_size_of_model(float_model)

    top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
    torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)



if __name__=='__main__':
    main()