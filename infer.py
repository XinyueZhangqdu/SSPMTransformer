"""
Rethinking Portrait Matting with Privacy Preserving
Inferernce file.

Copyright (c) 2022, Sihan Ma (sima7436@uni.sydney.edu.au) and Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting.git
Paper link: https://arxiv.org/abs/2203.16828

"""

import torch
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
from torchvision import transforms

from config import *
from util import *
from network import build_model
import os

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--arch', type=str, required =  False,default= 'vitae',  help='model architecture')
    parser.add_argument('--dataset', type=str, required=False, choices=['P3M10K', 'RWP', 'SAMPLES'], help='dataset to test')
    parser.add_argument('--test_set', type=str, choices=['RWP', 'P3M_500_P', 'P3M_500_NP'], help='the validation set to test')
    parser.add_argument('--model_path', type=str, default = 'P3M-Net_ViTAE-S_trained_on_P3M-10k.pth', help='path of checkpoint')
    parser.add_argument('--test_choice', type=str, required =False, choices=['HYBRID', 'RESIZE'],  help='how to test')
    parser.add_argument('--test_result_dir', type=str,default = '../SPDDataset/',help='path to save results of datasets')
    args, _ = parser.parse_known_args()

    print('Model architecture: {}'.format(args.arch))
    print('Model path: {}'.format(args.model_path))
    print('Test dataset: {}, set: {}'.format(args.dataset, args.test_set))
    print('Test choice: {}'.format(args.test_choice))
    if args.dataset != 'SAMPLES':
        print('Save results to {}'.format(args.test_result_dir))
    else:
        print('Save alpha results to {}'.format(SAMPLES_RESULT_ALPHA_PATH))
        print('Save color results to {}'.format(SAMPLES_RESULT_COLOR_PATH))

    return args

def inference_once(args, model, scale_img, scale_trimap=None):
    if torch.cuda.device_count() > 0:
        tensor_img = torch.from_numpy(scale_img.astype(np.float32)[:, :, :]).permute(2, 0, 1).cuda()
    else:
        tensor_img = torch.from_numpy(scale_img.astype(np.float32)[:, :, :]).permute(2, 0, 1)
    input_t = tensor_img
    input_t = input_t/255.0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    input_t = normalize(input_t)
    input_t = input_t.unsqueeze(0)
    pred_global, pred_local, pred_fusion = model(input_t)[:3]
    pred_global = pred_global.data.cpu().numpy()
    pred_global = gen_trimap_from_segmap_e2e(pred_global)
    pred_local = pred_local.data.cpu().numpy()[0,0,:,:]
    pred_fusion = pred_fusion.data.cpu().numpy()[0,0,:,:]
    return pred_global, pred_local, pred_fusion

def inference_img_sspm(args, model, img):
    h, w, c = img.shape
    new_h = min(MAX_SIZE_H, h - (h % 32))
    new_w = min(MAX_SIZE_W, w - (w % 32))

    # for P3M-Net(Swin-T) model on very small images in RWP

    #pred_global, pred_local, pred_fusion = inference_once(args, model, scale_img)
    #return pred_fusion
    '''
    if args.test_choice=='HYBRID':
        global_ratio = 1/2
        local_ratio = 1
        resize_h = int(h*global_ratio)
        resize_w = int(w*global_ratio)
        new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
        new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
        scale_img = resize(img,(new_h,new_w))*255.0
        pred_coutour_1, pred_retouching_1, pred_fusion_1 = inference_once(args, model, scale_img)
        # torch.cuda.empty_cache()
        pred_coutour_1 = resize(pred_coutour_1,(h,w))*255.0
        resize_h = int(h*local_ratio)
        resize_w = int(w*local_ratio)
        new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
        new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
        scale_img = resize(img,(new_h,new_w))*255.0
        pred_coutour_2, pred_retouching_2, pred_fusion_2 = inference_once(args, model, scale_img)
        # torch.cuda.empty_cache()
        pred_retouching_2 = resize(pred_retouching_2,(h,w))
        pred_fusion = get_masked_local_from_global_test(pred_coutour_1, pred_retouching_2)
        return pred_fusion
    elif args.test_choice=='RESIZE':
    '''
    resize_h = int(h/2)
    resize_w = int(w/2)
    new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
    new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
    scale_img = resize(img,(new_h,new_w))*255.0
    pred_global, pred_local, pred_fusion = inference_once(args, model, scale_img)
    pred_local = resize(pred_local,(h,w))
    pred_global = resize(pred_global,(h,w))*255.0
    pred_fusion = resize(pred_fusion,(h,w))
    return pred_fusion
    '''
    else:
        raise NotImplementedError
    '''
def test_dataset(args, model):
    if torch.cuda.device_count() > 0:
        torch.cuda.empty_cache()
    else:
        print('NO GPU AVAILABLE')
        return

    ############################
    # Some initial setting for paths
    ############################
    ORIGINAL_PATH = DATASET_PATHS_DICT[args.dataset][args.test_set]['ORIGINAL_PATH']

    ############################
    # Start testing
    ############################
    result_dir = args.test_result_dir
    refresh_folder(result_dir)

    model.eval()
    img_list = listdir_nohidden(ORIGINAL_PATH)

    for img_name in tqdm(img_list):
        img_path = ORIGINAL_PATH+img_name
        img = np.array(Image.open(img_path))
        img = img[:,:,:3] if img.ndim>2 else img

        with torch.no_grad():
            predict = inference_img_sspm(args, model, img)
            save_test_result(os.path.join(result_dir, extract_pure_name(img_name)+'.png'),predict)

def test_samples(args, model):
    model.eval()
    #SAMPLES_ORIGINAL_PATH = 'H:/ViTAE-Transformer-Matting-main/samples/original/'
    SAMPLES_ORIGINAL_PATH =  "H:/compare/ViTAE-Transformer-Matting-main/testImg/"
    SAMPLES_RESULT_ALPHA_PATH = 'H:/compare/ViTAE-Transformer-Matting-main/result/'
    img_list = os.listdir(SAMPLES_ORIGINAL_PATH)
    #img_list = listdir_nohidden(SAMPLES_ORIGINAL_PATH)
    #refresh_folder(SAMPLES_RESULT_ALPHA_PATH)
    #refresh_folder(SAMPLES_RESULT_COLOR_PATH)
    for img_name in tqdm(img_list):
        img_path = SAMPLES_ORIGINAL_PATH+img_name
        try:
            img = np.array(Image.open(img_path))[:,:,:3]

        except Exception as e:
            img = np.array(Image.open(img_path))
            h, w = img.shape
            img1 = np.zeros((h,w,3))
            img1[:,:,0] = img
            img1[:,:,1] = img
            img1[:,:,2] = img
            img = img1
        h, w, c = img.shape

        if min(h, w)>SHORTER_PATH_LIMITATION:
            if h>=w:
                new_w = SHORTER_PATH_LIMITATION
                new_h = int(SHORTER_PATH_LIMITATION*h/w)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                new_h = SHORTER_PATH_LIMITATION
                new_w = int(SHORTER_PATH_LIMITATION*w/h)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        with torch.no_grad():
            if torch.cuda.device_count() > 0:
                torch.cuda.empty_cache()
            predict = inference_img_sspm(args, model, img)

        composite = generate_composite_img(img, predict)
        cv2.imwrite(os.path.join(SAMPLES_RESULT_COLOR_PATH, extract_pure_name(img_name)+'.png'),composite)
        predict = predict*255.0
        predict = cv2.resize(predict, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(SAMPLES_RESULT_ALPHA_PATH, extract_pure_name(img_name)+'.png'),predict.astype(np.uint8))

def load_model_and_deploy(args):
    ### build model
    model = build_model(args.arch, pretrained=False)
    print("build successfully")
    ### load ckpt
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model = model.cuda()

    ### Test

    test_samples(args, model)


if __name__ == '__main__':
    args = get_args()
    load_model_and_deploy(args)
