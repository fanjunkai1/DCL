import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from datasets import DenseFogDataset
from layers import disp_to_depth
from options import MonodepthOptions
from tqdm import tqdm
import time
import argparse
from networks import ResnetEncoder, DepthDecoder, ResnetGenerator


def parse_args():

    parser = argparse.ArgumentParser(
        description='Simple testing funtion for DCL models.')

    parser.add_argument('--image_path', type=str,
                    help='path to a test image or folder of images', 
                    required=True)

    parser.add_argument('--model_name', type=str,
                    help='name of a pretrained model to use',
                    default = "DCL")
    
    parser.add_argument("--min_depth",
                    type=float,
                    help="minimum depth",
                    default=0.1)
    
    parser.add_argument("--max_depth",
                    type=float,
                    help="maximum depth",
                    default=100.0)
    
    parser.add_argument("--num_workers",
                    type=int,
                    help="number of dataloader workers",
                    default=4)

    parser.add_argument("--load_weights_folder",
                    type=str,
                    help="name of model to load",
                    default='./models/DCL')
    
    parser.add_argument("--dataset",
                    type=str,
                    help="dataset to train on",
                    default="gopro",
                    choices=["densefog", "lightfog"])           

    parser.add_argument("--num_layers",
                    type=int,
                    help="number of resnet layers",
                    default=18,
                    choices=[18, 34, 50, 101, 152])
    
    parser.add_argument("--save_pred_disps",
                    help="if set saves predicted disparities",
                    action="store_true")
    
    parser.add_argument("--post_process",
                    help="if set will perform the flipping post processing "
                            "from the original monodepth paper",
                    action="store_true")
    
    return parser.parse_args()

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(args):
    """Evaluates a pretrained model using a specified test set
    """
    # dataset and loader
    if args.dataset == 'densefog':
        dataset = DenseFogDataset(args.image_path,
                                  'splits/dense_fog/dense_fog.txt',
                                  'strongest')
    elif args.dataset == 'lightfog':
        dataset = DenseFogDataset(args.image_path,
                                  'splits/dense_fog/light_fog.txt', 
                                  'strongest')
    else:
        raise NotImplementedError()

    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False)

    # depth range
    MIN_DEPTH = dataset.min_depth
    MAX_DEPTH = dataset.max_depth

    assert os.path.isdir(args.load_weights_folder), \
        "Cannot find a folder at {}".format(args.load_weights_folder)

    print("-> Loading weights from {}".format(args.load_weights_folder))

    # load depth estimation model

    print("-> Loading pretrained depth estimation network")
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    encoder = ResnetEncoder(args.num_layers, False)
    depth_decoder = DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.to(device)
    encoder.eval()
    depth_decoder.to(device)
    depth_decoder.eval()

    # load video dehazing model
    print("-> Loading pretrained dehazing network")

    dehaze_path = os.path.join(args.load_weights_folder, "dehaze_network.pth")
    dehaze_network = ResnetGenerator(
        input_nc=3,
        ngf=64,
        n_blocks=9)
    loaded_dehaze_dict = torch.load(dehaze_path, map_location=device)
    dehaze_network.load_state_dict(loaded_dehaze_dict)
    dehaze_network.to(device)
    dehaze_network.eval()
    

    pred_disps = []
    pred_metrics = []

    print("-> Computing predictions and evaluating...")

    time_total = 0

    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader)):
            input_color = data[0].to(device)
            gt_depth = data[1].numpy()

            if args.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            start_time = time.time()

            output = depth_decoder(encoder(input_color))
            dehaze_output = dehaze_network(input_color)

            time_total += time.time() - start_time


            pred_disp, _ = disp_to_depth(output[("disp", 0)], args.min_depth, args.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            if args.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            # save predictions
            pred_disps.append(pred_disp)

            # evaluation
            pred_depth = 1.0 / pred_disp

            mask = gt_depth > 0

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            # median scaling
            ratio = np.median(gt_depth) / np.median(pred_depth)
            pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            pred_metrics.append(compute_errors(gt_depth, pred_depth))

            # save visualization

            output_dir = f'outputs/{args.dataset}_disp'
            os.makedirs(output_dir, exist_ok=True)
            plt.imsave(os.path.join(output_dir, f'{idx:04}_rgb.png'), data[0][0, :, :, :].permute(1, 2, 0).numpy())
            plt.imsave(os.path.join(output_dir, f'{idx:04}_disp.png'), pred_disp[0], cmap='magma')
            plt.imsave(os.path.join(output_dir, f'{idx:04}_gt.png'), data[1][0, :, :].numpy(), cmap='magma')

            # save dehaze results
            dehaze_resize_np = dehaze_output.squeeze().cpu().numpy()
            dehaze_resize_np = (np.transpose(dehaze_resize_np, (1, 2, 0)) + 1) / 2.0

            # output
            output_dir = f'outputs/{args.dataset}_dehazed'
            os.makedirs(output_dir, exist_ok=True)
            plt.imsave(os.path.join(output_dir, f'{idx:04}.png'), dehaze_resize_np)

    # save
    if args.save_pred_disps:
        pred_disps = np.concatenate(pred_disps)
        output_path = os.path.join(
            args.load_weights_folder, "disps_{}.npy".format(args.dataset))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    print("-> Evaluating")

    mean_errors = np.array(pred_metrics, dtype=np.float32).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")
    print('avgtime:{:.3f}'.format(time_total/572))

    print('-> Done!')


if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda')
    evaluate(args)
