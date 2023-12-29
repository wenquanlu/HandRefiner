


# STEP 1: Import the necessary modules.
from __future__ import absolute_import, division, print_function
import sys
from config import handrefiner_root
import os

def load():
    paths = [handrefiner_root, os.path.join(handrefiner_root, 'MeshGraphormer'), os.path.join(handrefiner_root, 'preprocessor')]
    for p in paths:
        sys.path.insert(0, p)

load()

import argparse
import json
import torch
import numpy as np
import cv2

from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import config

import cv2
import einops
import numpy as np
import torch
import random
from pathlib import Path
from preprocessor.meshgraphormer import MeshGraphormerMediapipe
import ast

transform = transforms.Compose([           
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

def parse_args():
    parser = argparse.ArgumentParser()

    # input directory containing images to be rectified
    parser.add_argument('--input_dir', type=str, default="")

    # input image
    parser.add_argument('--input_img', type=str, default="")

    # output directory where the rectified images will be saved to
    parser.add_argument('--out_dir', type=str, default="")

    # file where the mpjpe values will be logged to
    parser.add_argument('--log_json', type=str, default="")

    # control strength for ControlNet
    parser.add_argument('--strength', type=float, default=1.0)

    # directory where the depth maps will be saved to. Leaving it empty will disable this function
    parser.add_argument('--depth_dir', type=str, default="")

    # directory where the masks will be saved to. Leaving it empty will disable this function
    parser.add_argument('--mask_dir', type=str, default="")

    # whether evaluate the mpjpe error in fixed control strength mode
    parser.add_argument('--eval', type=ast.literal_eval, default=False)

    # whether use finetuned ControlNet trained on synthetic images as introduced in the paper
    parser.add_argument('--finetuned', type=ast.literal_eval, default=True)

    # path to the SD + ControlNet weights
    parser.add_argument('--weights', type=str, default="")

    # batch size
    parser.add_argument('--num_samples', type=int, default=1)

    # prompt file for multi-image rectification
    # see manual.md for file format
    parser.add_argument('--prompt_file', type=str, default="")

    # prompt for single image rectification
    parser.add_argument('--prompt', type=str, default="")

    # number of generation iteration for each image to be rectified
    # in general, for each input image, n_iter x num_samples number of rectified images will be produced
    parser.add_argument('--n_iter', type=int, default=1)

    # adaptive control strength as introduced in paper (we tend to use fixed control strength as default)
    parser.add_argument('--adaptive_control', type=ast.literal_eval, default=False)

    # padding controls the size of masks around the hand
    parser.add_argument('--padding_bbox', type=int, default=30)
    
    # set seed
    parser.add_argument('--seed', type=int, default=-1)
    args = parser.parse_args()
    return args

args = parse_args()

if (args.prompt_file != "" and args.prompt != "") or (args.prompt_file == "" and args.prompt == ""):
    raise Exception("Please specify one and only one of the --prompt and --prompt_file")
if (args.input_dir != "" and args.input_img != "") or (args.input_dir == "" and args.input_img == ""):
    raise Exception("Please specify one and only one of the --input_dir and --input_img")

model = create_model("control_depth_inpaint.yaml").cpu()
if args.finetuned:
    model.load_state_dict(load_state_dict(args.weights, location='cuda'), strict=False)
else:
    model.load_state_dict(
        load_state_dict("models/sd-v1-5-inpainting.ckpt", location="cuda"), strict=False
    )
    model.load_state_dict(
        load_state_dict("models/control_v11f1p_sd15_depth.pth", location="cuda"),
        strict=False,
    )

model = model.to("cuda")

meshgraphormer = MeshGraphormerMediapipe()

if args.log_json != "":
    f_mpjpe = open(args.log_json, 'w')


# prompt needs to be same for all pictures in the same batch
if args.input_img != "":
    assert args.prompt_file == "", "prompt file should not be used for single image rectification"
    inputs = [args.input_img]
else:
    if args.prompt_file != "":
        f_prompt = open(args.prompt_file)
        inputs = f_prompt.readlines()
    else:
        inputs = os.listdir(args.input_dir)

for file_info in inputs:
    if args.prompt_file != "":
        file_info = json.loads(file_info)
        file_name = file_info["img"]
        prompt = file_info["txt"]
    else:
        file_name = file_info
        prompt = args.prompt

    image_file = os.path.join(args.input_dir, file_name)

    file_name_raw = Path(file_name).stem

    # STEP 3: Load the input image.
    image = np.array(Image.open(image_file))

    raw_image = image
    H, W, C = raw_image.shape
    gen_count = 0
    for iteration in range(args.n_iter):

        depthmap, mask, info = meshgraphormer.get_depth(args.input_dir, file_name, args.padding_bbox)

        if args.depth_dir != "":
            cv2.imwrite(os.path.join(args.depth_dir, file_name_raw + "_depth.jpg"), depthmap)
        if args.mask_dir != "":
            cv2.imwrite(os.path.join(args.mask_dir, file_name_raw + "_mask.jpg"), mask)
        
        control = depthmap

        ddim_sampler = DDIMSampler(model)
        num_samples = args.num_samples
        ddim_steps = 50
        guess_mode = False
        strength = args.strength
        scale = 9.0
        seed = args.seed

        label = file_name[:2]
        a_prompt = "realistic, best quality, extremely detailed"
        n_prompt = "fake 3D rendered image, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, blue"

        source = raw_image

        source = (source.astype(np.float32) / 127.5) - 1.0
        source = source.transpose([2, 0, 1])  # source is c h w

        mask = mask.astype(np.float32) / 255.0
        mask = mask[None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        hint = control.astype(np.float32) / 255.0

        masked_image = source * (mask < 0.5)  # masked image is c h w

        mask = torch.stack([torch.tensor(mask) for _ in range(num_samples)], dim=0).to("cuda")
        mask = torch.nn.functional.interpolate(mask, size=(64, 64))

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        masked_image = torch.stack(
            [torch.tensor(masked_image) for _ in range(num_samples)], dim=0
        ).to("cuda")

        # this should be b,c,h,w
        masked_image = model.get_first_stage_encoding(model.encode_first_stage(masked_image))

        x = torch.stack([torch.tensor(source) for _ in range(num_samples)], dim=0).to("cuda")
        z = model.get_first_stage_encoding(model.encode_first_stage(x))

        cats = torch.cat([mask, masked_image], dim=1)

        hint = hint[
            None,
        ].repeat(3, axis=0)

        hint = torch.stack([torch.tensor(hint) for _ in range(num_samples)], dim=0).to("cuda")

        cond = {
            "c_concat": [cats],
            "c_control": [hint],
            "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)],
        }
        un_cond = {
            "c_concat": [cats],
            "c_control": [hint],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
        }
        

        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        if not args.adaptive_control:
            seed_everything(seed)
            model.control_scales = (
                [strength * (0.825 ** float(12 - i)) for i in range(13)]
                if guess_mode
                else ([strength] * 13)
            )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = ddim_sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond,
                x0=z,
                mask=mask
            )
            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            x_samples = model.decode_first_stage(samples)
            # print(x_samples.shape)
            x_samples = (
                (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
                .cpu()
                .numpy()
                .clip(0, 255)
                .astype(np.uint8)
            )

            if args.eval: # currently only works for batch size of 1 
                assert args.num_samples == 1, "MPJPE evaluation currently only works for batch size of 1"
                mpjpe = meshgraphormer.eval_mpjpe(x_samples[0], info)
                print(mpjpe)
                if args.log_json != "":
                    mpjpe_info = {"img": image_file, "strength": strength, "mpjpje": mpjpe}
                    f_mpjpe.write(json.dumps(mpjpe_info))
                    f_mpjpe.write("\n")
            for i in range(args.num_samples):
                cv2.imwrite(
                    os.path.join(args.out_dir, "{}_{}.jpg".format(file_name_raw, gen_count)), cv2.cvtColor(x_samples[i], cv2.COLOR_RGB2BGR)
                )
                gen_count += 1
        else:
            assert args.num_samples == 1, "Adaptive thresholding currently only works for batch size of 1"
            strengths = [1.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            ref_mpjpe = None
            chosen_strength = None
            final_mpjpe = None
            chosen_sample = None
            count = 0
            for strength in strengths:
                seed_everything(seed)
                model.control_scales = (
                    [strength * (0.825 ** float(12 - i)) for i in range(13)]
                    if guess_mode
                    else ([strength] * 13)
                )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
                samples, intermediates = ddim_sampler.sample(
                    ddim_steps,
                    num_samples,
                    shape,
                    cond,
                    verbose=False,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=un_cond,
                    x0=z,
                    mask=mask
                )
                if config.save_memory:
                    model.low_vram_shift(is_diffusing=False)

                x_samples = model.decode_first_stage(samples)

                x_samples = (
                    (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
                    .cpu()
                    .numpy()
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                mpjpe = meshgraphormer.eval_mpjpe(x_samples[0], info)
                if count == 0:
                    ref_mpjpe = mpjpe
                    chosen_sample = x_samples[0]
                elif mpjpe < ref_mpjpe * 1.15:
                    chosen_strength = strength
                    final_mpjpe = mpjpe
                    chosen_sample = x_samples[0]
                    break
                elif strength == 0.9:
                    final_mpjpe = ref_mpjpe
                    chosen_strength = 1.0
                count += 1
            
            if args.log_json != "":    
                mpjpe_info = {"img": image_file, "strength": chosen_strength, "mpjpje": final_mpjpe}
                f_mpjpe.write(json.dumps(mpjpe_info))
                f_mpjpe.write("\n")

            cv2.imwrite(
            os.path.join(args.out_dir, "{}_{}.jpg".format(file_name_raw, gen_count)), cv2.cvtColor(x_samples[0], cv2.COLOR_RGB2BGR)
            )
            gen_count += 1


