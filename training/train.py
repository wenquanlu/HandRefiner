from ldm.data.control_synthcompositedata import Control_composite_Hand_synth_data
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from einops import rearrange
from PIL import Image
import numpy as np
import os
from cldm.logger import ImageLogger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--devices", default="0", type=str, help="comma delimited list of devices")
parser.add_argument("--batchsize", default=4, type=int)
parser.add_argument("--acc_grad", default=4, type=int)
parser.add_argument("--max_epochs", default=3, type=int)
args = parser.parse_args()
args.devices = [int(n) for n in args.devices.split(",")]

def get_state_dict(d):
    return d.get('state_dict', d)
def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict

learning_rate = 1e-5

model = create_model("control_depth_inpaint.yaml")

#### load the SD inpainting weights
states = load_state_dict("./sd-v1-5-inpainting.ckpt", location='cpu')
model.load_state_dict(states, strict=False)

control_states = load_state_dict("./models/control_v11f1p_sd15_depth.pth")
model.load_state_dict(control_states, strict=False)


model.learning_rate = learning_rate
model.sd_locked = True
model.only_mid_control = False

dataset = Control_composite_Hand_synth_data()

checkpoint_callback = ModelCheckpoint(save_top_k=-1, monitor="epoch")

#### start of the training expectation: the model should behave the same to standalone depth controlnet + inpainting SD
dataloader = DataLoader(dataset, num_workers=8, batch_size=args.batchsize, shuffle=True)
trainer = pl.Trainer(precision=32, max_epochs=args.max_epochs, accelerator="gpu", devices=args.devices, accumulate_grad_batches=args.acc_grad, callbacks=[ImageLogger(), checkpoint_callback], strategy='ddp')
trainer.fit(model, dataloader)