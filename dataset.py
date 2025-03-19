import os
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
from imageio import get_writer

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import center_crop, resize

def image2tensor(image_file, width=512, height=320):
    image = Image.fromarray(image_file).convert('RGB')
    image = image.resize((width, height))
    image_tensor = transforms.ToTensor()(image)
    image_tensor = torch.from_numpy(np.ascontiguousarray(image_tensor)).float()
    # normalize
    image_tensor = image_tensor * 2. - 1.
    
    return image_tensor

def pad_last_frame(tensor, num_frames):
    # T, H, W, C
    if len(tensor) < num_frames:
        pad_length = num_frames - len(tensor)
        # Use the last frame to pad instead of zero
        last_frame = tensor[-1]
        pad_tensor = last_frame.unsqueeze(0).expand(pad_length, *tensor.shape[1:])
        padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
        return padded_tensor
    else:
        return tensor[:num_frames]

def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=transforms.InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = transforms.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr

class VideoDataset(Dataset):
    def __init__(self, root, video_size, fps, max_num_frames, skip_frms_num=0):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(VideoDataset, self).__init__()

        self.root = root
        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num

    def __getitem__(self, index):

        import decord
        from decord import VideoReader
        decord.bridge.set_bridge("torch")

        video_path = self.root
        vr = VideoReader(uri=video_path, height=-1, width=-1)
        actual_fps = vr.get_avg_fps()
        ori_vlen = len(vr)

        if ori_vlen / actual_fps * self.fps > self.max_num_frames:
            num_frames = self.max_num_frames
            start = int(self.skip_frms_num)
            end = int(start + num_frames / self.fps * actual_fps)
            end_safty = min(int(start + num_frames / self.fps * actual_fps), int(ori_vlen))
            indices = np.arange(start, end, (end - start) // num_frames).astype(int)
            temp_frms = vr.get_batch(np.arange(start, end_safty))
            assert temp_frms is not None
            tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
            tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]
        else:
            if ori_vlen > self.max_num_frames:
                num_frames = self.max_num_frames
                start = int(self.skip_frms_num)
                end = int(ori_vlen - self.skip_frms_num)
                indices = np.arange(start, end, max((end - start) // num_frames, 1)).astype(int)
                temp_frms = vr.get_batch(np.arange(start, end))
                assert temp_frms is not None
                tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
                tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]
            else:

                def nearest_smaller_4k_plus_1(n):
                    remainder = n % 4
                    if remainder == 0:
                        return n - 3
                    else:
                        return n - remainder + 1

                start = int(self.skip_frms_num)
                end = int(ori_vlen - self.skip_frms_num)
                num_frames = nearest_smaller_4k_plus_1(end - start)  # 3D VAE requires the number of frames to be 4k+1
                end = int(start + num_frames)
                temp_frms = vr.get_batch(np.arange(start, end))
                assert temp_frms is not None
                tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms

        # the len of indices may be less than num_frames, due to round error
        tensor_frms = pad_last_frame(
            tensor_frms, self.max_num_frames
        )  
        tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        tensor_frms = resize_for_rectangle_crop(tensor_frms, self.video_size, reshape_mode="center")
        tensor_frms = (tensor_frms - 127.5) / 127.5

        assert tensor_frms.size(0) == self.max_num_frames, f'{video_path}: {tensor_frms.shape}'

        # caption
        caption = "The video features a figure standing against a minimalistic, softly lit background that is predominantly blue."

        item = {
            "video": tensor_frms,
            "prompt": caption,
        }
        return item

    def __len__(self):
        return 1
    
class ValidationDataset(Dataset):
    def __init__(self, root):   
        self.root = root

    def __len__(self):
        return 1

    def __getitem__(self, index):

        image_path = "test.png"
        caption = "The video features a figure standing against a minimalistic, softly lit background that is predominantly blue."
        
        return {"image": image_path, 'prompt': caption}