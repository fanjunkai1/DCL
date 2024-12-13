from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

# from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class GoProDataset(MonoDataset):
    """
    Superclass for different types of GoProHazy dataset loaders
    
    """
    def __init__(self, *args, **kwargs):
        super(GoProDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        # self.K = np.array([[0.58, 0, 0.5, 0],
        #                    [0, 1.92, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1600, 512)

        self.K = self.load_intrinsic()
    
    def load_intrinsic(self):
        """
        Load and parse intrinsic matrix
        :return:
        """
        src_k = np.load(os.path.join(self.data_path, 'intrinsic.npy')).astype(np.float32)
        fx, cx = src_k[0, 0], src_k[0, 2]
        fy, cy = src_k[1, 1], src_k[1, 2]

        # fx = fx / self.full_res_shape[0]
        # fy = fy / self.full_res_shape[1]

        intrinsic = np.array([
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        # print(intrinsic)
        return intrinsic

    def check_ref_disp(self):
        line = self.filenames[0].split()
        scene_name = line[0].replace("hazy", "disp")
        ref_disp_index = int(line[2])

        disp_filename = os.path.join(self.data_path,
                     scene_name,
                     "{:05d}_disp.jpeg".format(ref_disp_index))
        
        return os.path.isfile(disp_filename)
    
    def get_image_path(self, folder, frame_index, frame_type):

        f_str = "{:05d}.jpg".format(frame_index)
     
        if frame_type == 'ref_clear':
            folder = folder.replace("hazy", "clear")

        image_path = os.path.join(self.data_path, folder, f_str)

        return image_path

    # def get_refImg_path(self, folder, refer_index):

    #     # get reference clear image
    #     c_f_str = "{:05d}.jpg".format(refer_index)
    #     refImg_folder = folder.replace("hazy", "clear")
    #     refImg_path = os.path.join(self.data_path, refImg_folder, c_f_str)

    #     return refImg_path
    
    def get_color(self, folder, frame_index, do_flip, frame_type):
        color = self.loader(self.get_image_path(folder, frame_index, frame_type))

        if do_flip:
            self.K[0, 2] = self.full_res_shape[0] - self.K[0, 2]
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    # def get_ref_color(self, folder, refer_index, do_flip):
    #     ref_color = self.loader(self.get_refImg_path(folder, refer_index))

    #     if do_flip:
    #         self.K[0, 2] = self.full_res_shape[0] - self.K[0, 2]
    #         ref_color = ref_color.transpose(pil.FLIP_LEFT_RIGHT)
        
    #     return ref_color

    def get_ref_disp(self, folder, frame_index, do_flip):

        f_str = "{:05d}_disp.jpeg".format(frame_index)
        disp_folder = folder.replace("hazy", "disp")
        disp_path = os.path.join(self.data_path,
                                  disp_folder,
                                  f_str)
        
        ref_disp = pil.open(disp_path)
        # ref_disp = ref_disp.resize(self.full_res_shape, pil.NEAREST)
        ref_disp = np.array(ref_disp).astype(np.float32) / 255

        if do_flip:
            ref_disp = np.fliplr(ref_disp)

        return ref_disp

        
