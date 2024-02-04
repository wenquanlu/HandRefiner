## Training Script - train.py
The training script should be placed at the same level of the cldm folder.
Some paths needed to be manually set:

<li>L40: path to SD1.5
<li>L43: path to depth controlnet weight

## Data Loader - control_synthcompositedata.py
The loader should be placed in ldm/data/

Some paths needed to be mannally set:

dataset needs to be structured as:
```bash
|- dataset1
|   |- image
|   |- mask
|   |- pose
|   |- prompt.json
```
Some paths needed to be manually set:
<li>L9: path to dataset 1
<li>L10: path to dataset 2
<li>L18: path to dataset 1 prompt json file
<li>L23: path to dataset 2 prompt json file

Each prompt json file are structured as:
```json
{"jpg": "image name", "txt": "text prompt", "dataset": "dataset identifier (RHD|synthesisai)"}
```
