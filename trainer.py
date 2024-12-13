# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vgg16
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
# from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.models_D = {}
        self.parameters_to_train = []
        self.parameters_to_train_D = []

        self.dehaze_pool = ImagePool(50)
        self.disp_pool = ImagePool(50)

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.num_pose_frames = 2

        # depth network
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, 
            self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # pose network
        self.models["pose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)
        self.models["pose_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())

        self.models["pose"] = networks.PoseDecoder(
            self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)
        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())

        # dehazing network
        self.models["dehaze_network"] = networks.ResnetGenerator(
            input_nc=3,
            ngf = 64, 
            n_blocks = 9)
        self.models["dehaze_network"].to(self.device)
        self.parameters_to_train += list(self.models["dehaze_network"].parameters())

        # beta network
        self.models["beta"] = networks.BetaDecoder(
            self.models["encoder"].num_ch_enc)
        self.models["beta"].to(self.device)
        self.parameters_to_train += list(self.models["beta"].parameters())

        # define discriminator for J-Net
        self.models_D["discriminator_J"] = networks.HFDiscriminator(
            input_nc=3,
            ndf=64, 
            n_layers=1,
            norm = self.opt.norm)
        self.models_D["discriminator_J"].to(self.device)
        self.parameters_to_train_D += list(self.models_D["discriminator_J"].parameters())

        # define discriminator for D-Net
        self.models_D["discriminator_D"] = networks.NLayerDiscriminator(
            input_nc=1,
            ndf=64, 
            n_layers=2,
            norm = self.opt.norm)
        self.models_D["discriminator_D"].to(self.device)
        self.parameters_to_train_D += list(self.models_D["discriminator_D"].parameters())

        # define GAN loss.
        self.criterionGAN = networks.GANLoss(self.opt.gan_mode).to(self.device)

        # definte rec loss
        self.criterionRec = torch.nn.L1Loss()

        # contextual loss
        # layers = {
        #     "conv_1_2": 1.0,
        #     "conv_2_2": 1.0,
        #     "conv_3_2": 1.0
        # }
        layers = {
                "relu_1_2": 1.0,
                "relu_2_2": 1.0,
                "relu_3_3": 1.0,
                "relu_4_3": 1.0,
                "relu_5_3": 1.0
            }
        self.contextual_loss = networks.Contextual_Loss(layers, max_1d_size=64, device=self.device).to(self.device)

        # --- Define the perceptual loss network --- #
        vgg_model = vgg16(pretrained=True).features[:31]
        vgg_model = vgg_model.to(self.device)
        for param in vgg_model.parameters():
            param.requires_grad = False

        self.loss_network = networks.LossNetwork(vgg_model)
        self.loss_network.eval()

        # optimizer and scheduler
        self.model_optimizer = optim.Adam(self.parameters_to_train, 
                                          self.opt.learning_rate)
        
        self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(
                                          self.model_optimizer, 
                                          milestones=[15], gamma=0.1)
        
        self.model_optimizer_D = optim.Adam(self.parameters_to_train_D, 
                                          self.opt.learning_rate)
        
        self.model_lr_scheduler_D = optim.lr_scheduler.MultiStepLR(
                                          self.model_optimizer_D, 
                                          milestones=[15], gamma=0.1)
        
        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)
        
        datasets_dict = {"gopro": datasets.GoProDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        # dataset and dataloader 
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)


        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        # SSIM Loss 
        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}

        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_requires_grad(self, nets, requires_grad=False):
        """
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for model in net:
                    for param in net[model].parameters():
                        if param ==  'LH.weight' or param == 'HL.weight' or param == 'HH.weight':
                            param.requires_grad = False
                        param.requires_grad = requires_grad


    def set_train(self):
        """
        Convert all models to training mode

        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """
        Convert all models to testing/evaluation mode

        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """
        Run the entire training pipeline

        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """
        Run a single epoch of training and validation

        """
        # self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.set_requires_grad(self.models_D, False)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            self.set_requires_grad(self.models_D, True)
            self.model_optimizer_D.zero_grad()
            self.backward_D_dehaze()
            self.backward_D_depth()
            self.model_optimizer_D.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 400
            late_phase = self.step % 400 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, 
                              duration, 
                              losses["loss"].cpu().data, 
                              losses["loss/G_dehaze"].cpu().data, 
                              losses["loss/D_dehaze"].cpu().data,
                              losses["loss/G_depth"].cpu().data,
                              losses["loss/D_depth"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                
                self.log("train", inputs, outputs, losses)
                self.val()
            
            self.step += 1

        self.model_lr_scheduler.step()
        self.model_lr_scheduler_D.step()
    
    def process_batch(self, inputs):
        """
        Pass a minibatch through the network and generate images and losses

        """
        haze = []
        outputs = {}

        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        
        self.ref_disp = inputs["ref_disp"]

        # we only feed the image with frame_id 0 through the depth encoder
        features = self.models["encoder"](inputs["haze_aug", 0, 0])
        outputs = self.models["depth"](features)

        outputs[("beta", 0, 0)] = self.models["beta"](features)

        ################################

        for f_i in self.opt.frame_ids:
            haze.append(inputs["haze", f_i, 0])

        haze_input = torch.stack(haze, dim=1)
        b, t, c, h, w = haze_input.shape

        self.dehaze_output = (self.models["dehaze_network"](haze_input.view(b*t, c, h, w))+1)/2

        dehaze_output = self.dehaze_output.view(b, t, c, h, w)

        outputs[("dehaze", 0, 0)] = dehaze_output[:, 0, :, :, :]
        outputs[("dehaze", -1, 0)] = dehaze_output[:, 1, :, :, :]
        outputs[("dehaze", 1, 0)] = dehaze_output[:, 2, :, :, :]

        for f_i in self.opt.frame_ids:
            for scale in self.opt.scales:
                if scale == '0':
                    continue
                else:
                    s = 2 ** int(scale)
                    outputs["dehaze", f_i, int(scale)] = F.interpolate(
                                    outputs["dehaze", f_i, 0],
                                    [self.opt.height // s, self.opt.width // s],
                                    mode="bilinear", 
                                    align_corners=False)
                    
        ################################

        outputs.update(self.predict_poses(outputs))
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_all_losses(inputs, outputs)

        return outputs, losses
    

    def predict_poses(self, outputs):
        """
        Predict poses between input frames for monocular sequences.

        """

        # In this setting, we compute the pose to each source frame via a
        # separate forward pass through the pose network.

        pose_feats = {f_i: outputs["dehaze", f_i, 0] for f_i in self.opt.frame_ids}

        for f_i in self.opt.frame_ids[1:]:

            # To maintain ordering we always pass frames in temporal order
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

            axisangle, translation = self.models["pose"](pose_inputs)
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation

            # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        
        return outputs
    

    def val(self):
        """
        Validate the model on a single minibatch

        """
        self.set_eval()

        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)
            
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses
        
        self.set_train()

    
    def generate_images_pred(self, inputs, outputs):
        """
        Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.

        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            scaled_disp, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth
            outputs[("scaled_disp", 0, 0)] = scaled_disp

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                depth, inputs[("inv_K", source_scale)])

                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
                
                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("dehaze_sample", frame_id, scale)] = F.grid_sample(
                    outputs[("dehaze", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)
                
                outputs[("dehaze_output_identity", frame_id, scale)] = \
                    outputs[("dehaze", frame_id, source_scale)]
                    
    def compute_reprojection_loss(self, pred, target):
        """
        Computes reprojection loss between a batch of predicted and target images

        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    
    
    def compute_SMDE_losses(self, outputs, losses):
        """
        Compute the reprojection and smoothness losses for a minibatch

        """
        smde_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            source_scale = 0

            disp = outputs[("disp", scale)]
            self.source_scale_disp_pre = outputs[("disp", 0)]
            
            # print("haha",disp.shape)
            color = outputs[("dehaze", 0, scale)]
            target = outputs[("dehaze", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("dehaze_sample", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_loss = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("dehaze", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))
                
            identity_reprojection_loss = torch.cat(identity_reprojection_losses, 1)

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            to_optimise, idxs = torch.min(combined, dim=1)

            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()
            
            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            # print(norm_disp.shape, color.shape)
            smooth_loss = get_smooth_loss(norm_disp, color)  #issue

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            smde_loss += loss
            losses["loss/{}".format(scale)] = loss

        smde_loss /= self.num_scales

        return smde_loss, losses
    

    def compute_rec_losses(self, inputs, outputs):

        # get A_infinite
        dark_img = networks.dehaze.get_dark_channel(inputs["haze", 0, 0])
        A_infinite = networks.dehaze.get_atmospheric_light(inputs["haze", 0, 0], dark_img)

        # get rec haze
        # mean_beta =  outputs[("beta", 0, 0)].mean(2, True).mean(3, True)
        # norm_beta = outputs[("beta", 0, 0)] / (mean_beta + 1e-7)

        # mean_depth = outputs[("depth", 0, 0)].mean(2, True).mean(3, True)
        # norm_depth = outputs[("depth", 0, 0)] / (mean_depth + 1e-7)

        # trans_map = torch.exp(- norm_beta * outputs[("depth", 0, 0)]).clamp(min=0, max=1)
        trans_map = torch.exp(- outputs[("beta", 0, 0)] * outputs[("scaled_disp", 0, 0)]
                              * outputs[("depth", 0, 0)]).clamp(min=0, max=1)
        rec_haze = (outputs[("dehaze", 0, 0)]) * trans_map + A_infinite * (1 - trans_map)
        # rec_haze = (rec_haze - 0.5) / 0.5   # rec_haze scale [-1,1]
        outputs["rec_haze", 0, 0] = rec_haze

        # compute reconstruction loss by using atmospheric scatter model
        L1_loss = self.criterionRec(rec_haze, inputs["haze", 0, 0])
        ssim_loss = self.ssim(rec_haze, inputs["haze", 0, 0]).mean()
        perceptual_loss = self.loss_network(rec_haze, inputs["haze", 0, 0])
        rec_loss = (L1_loss + ssim_loss + perceptual_loss) * 0.1

        return rec_loss
    
    def compute_contextual_losses(self, inputs):

        # get rec dehaze
        # rec_dehaze = reverse_fog(inputs["haze", 0, 0], trans_map, A_infinite)

        # compute contextual loss
        self.ref_clear = torch.stack([inputs["ref_clear", 0, 0],
                                      inputs["ref_clear", -1, 0], 
                                      inputs["ref_clear", 1, 0]],
                                      dim=1).view(-1, 3, self.opt.height, self.opt.width)
        
        loss_contextual = self.contextual_loss(self.dehaze_output, self.ref_clear) * 0.2
        # loss_contextual2 = self.contextual_loss(rec_dehaze, inputs["ref_clear", 0, 0])
        # loss_contextual = (loss_contextual1 + loss_contextual2) * 0.5

        return loss_contextual
    
    def backward_D_basic(self, discriminator, real, fake, weight):

        """
        Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.

        """
        # Real
        pred_real = discriminator(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = discriminator(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * weight
        loss_D.backward()
        return loss_D
    
    def backward_D_dehaze(self):
        """Calculate GAN loss for discriminator D_J"""

        dehaze_output = self.dehaze_pool.query(self.dehaze_output)
        self.loss_D_dehaze = self.backward_D_basic(self.models_D["discriminator_J"], self.ref_clear, dehaze_output, 4.0e-3)
        self.losses["loss/D_dehaze"] = self.loss_D_dehaze

    def backward_D_depth(self):
        """Calculate GAN loss for discriminator D_D"""

        pre_disp = self.disp_pool.query(self.source_scale_disp_pre)
        # norm estimate disp
        mean_pre_disp = pre_disp.mean(2, True).mean(3, True)
        norm_pre_disp = pre_disp / (mean_pre_disp + 1e-7)
        # norm reference disp
        mean_ref_disp = self.ref_disp.mean(2, True).mean(3, True)
        norm_ref_disp = self.ref_disp / (mean_ref_disp + 1e-7)
        
        self.loss_D_depth = self.backward_D_basic(self.models_D["discriminator_D"], norm_ref_disp, norm_pre_disp, 1.0e-3)
        self.losses["loss/D_depth"] = self.loss_D_depth

    def compute_all_losses(self, inputs, outputs):

        self.losses = {}
        SMDE_loss, self.losses = self.compute_SMDE_losses(outputs, self.losses)
        self.losses["loss/smde_loss"] = SMDE_loss

        rec_loss = self.compute_rec_losses(inputs, outputs)
        self.losses["loss/rec_loss"] = rec_loss

        contextual_loss = self.compute_contextual_losses(inputs)
        self.losses["loss/contextual_loss"] = contextual_loss

        # norm estimate disp
        mean_pre_disp = outputs[("disp", 0)].mean(2, True).mean(3, True)
        norm_pre_disp = outputs[("disp", 0)] / (mean_pre_disp + 1e-7)

        dehaze_loss = self.criterionGAN(self.models_D["discriminator_J"](self.dehaze_output), True) * 4.0e-3
        self.losses["loss/G_dehaze"] = dehaze_loss

        depth_loss = self.criterionGAN(self.models_D["discriminator_D"](norm_pre_disp), True) * 1.0e-3
        self.losses["loss/G_depth"] = depth_loss

        self.losses["loss"] = SMDE_loss + rec_loss + contextual_loss + dehaze_loss + depth_loss

        return self.losses


    def compute_depth_losses(self, inputs, outputs, losses):
        """
        Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance

        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())
    
    def log_time(self, batch_idx, duration, loss, G_dehaze, D_dehaze, G_depth, D_depth):
        """
        Print a logging statement to the terminal

        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | loss_G_dehaze: {:.5f} | loss_D_dehaze: {:.5f} | loss_G_depth: {:.5f} | loss_D_depth: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss, G_dehaze, D_dehaze, G_depth, D_depth,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
        
    def log(self, mode, inputs, outputs, losses):
        """
        Write an event to the tensorboard events file

        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "haze_{}_{}/{}".format(frame_id, s, j),
                        inputs[("haze", frame_id, s)][j].data, self.step)

                    #################################
                    writer.add_image(
                        "dehaze_{}_{}/{}".format(frame_id, s, j),
                        outputs[("dehaze", frame_id, s)][j].data, self.step)
                    #################################
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "haze_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("dehaze_sample", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                writer.add_image(
                    "automask_{}/{}".format(s, j),
                    outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

            ###################################
            for frame_id in self.opt.frame_ids:
                writer.add_image(
                    "ref_clear_{}_{}/{}".format(frame_id, 0, j),
                    inputs[("ref_clear", frame_id, 0)][j].data, self.step)
            ###################################

            writer.add_image(
                    "rec_haze_{}_{}/{}".format(0, 0, j),
                    outputs[("rec_haze", 0, 0)][j].data, self.step)
            
            writer.add_image(
                    "beta_{}_{}/{}".format(0, 0, j),
                    outputs[("beta", 0, 0)][j].data, self.step)
            
            writer.add_image(
                    "ref_disp_{}/{}".format(0, j),
                    normalize_image(inputs[("ref_disp")][j]), self.step)

    def save_opts(self):
        """
        Save options to disk so we know what we ran this experiment with

        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """
        Save model weights to disk

        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """
        Load model(s) from disk

        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

