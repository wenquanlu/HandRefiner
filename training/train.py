import json
import cv2
import numpy as np
from PIL import Image
import random

from torch.utils.data import Dataset

DATA_PATH_1 = "../RHD/RHD_published_v2/"
DATA_PATH_2 = "../synthesisai/"

abbrev_dict = {"RHD": DATA_PATH_1, 
                "synthesisai": DATA_PATH_2}

class Control_composite_Hand_synth_data(Dataset):
    def __init__(self):
        self.data = []
        with open('../RHD/RHD_published_v2/embedded_rgb_caption.json', 'rt') as f1:
            for line in f1:
                item = json.loads(line)
                item['dataset'] = 'RHD'
                self.data.append(item)     
        with open('../synthesisai/embedded_rgb_caption.json', 'rt') as f2:
            for line in f2:
                item = json.loads(line)
                item['dataset'] = 'synthesisai'
                self.data.append(item)     
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        source_filename = item['jpg']
        prompt = item['txt']   
        dataset = item['dataset']
        datapath = abbrev_dict[dataset]
        if random.random() < 0.5:
            prompt = ""
        source = cv2.imread(datapath + "image/" + source_filename)
        source = (source.astype(np.float32) / 127.5) - 1.0
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

        mask = np.array(Image.open(datapath + "mask/" + source_filename).convert("L"))
        mask = mask.astype(np.float32)/255.0
        mask = mask[None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = np.transpose(mask, [1, 2, 0])

        hint = cv2.imread(datapath + "pose/" + source_filename)
        hint = cv2.cvtColor(hint, cv2.COLOR_BGR2RGB)

        hint = hint.astype(np.float32) / 255.0

        masked_image = source * (mask < 0.5)
        return dict(jpg=source, txt=prompt, hint=hint, mask=mask, masked_image=masked_image)