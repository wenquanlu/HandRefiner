# MeshGraphormer Instructions for HandRefiner

## Installation

### Requirements



Install the MeshGraphormer to HandRefiner/MeshGraphormer

```bash
git clone --recursive https://github.com/microsoft/MeshGraphormer.git
cd MeshGraphormer
pip install ./manopth/.
```


## Download
Make sure you are on 'HandRefiner/MeshGraphormer' directory for the following steps
1. Create folder that store pretrained models.
    ```bash
    mkdir -p models  # pre-trained models
    ```

2. Download pretrained models, and some code modifications.

    ```bash
    cp ../scripts/download_models.sh scripts/download_models.sh
    cp ../scripts/_gcnn.py src/modeling/_gcnn.py
    cp ../scripts/_mano.py src/modeling/_mano.py
    cp ../scripts/config.py src/modeling/data/config.py
    bash scripts/download_models.sh
    ```

    The resulting data structure should follow the hierarchy as below. 
    ```
    MeshGraphormer 
    |-- models  
    |   |-- graphormer_release
    |   |   |-- graphormer_hand_state_dict.bin
    |   |-- hrnet
    |   |   |-- hrnetv2_w64_imagenet_pretrained.pth
    |   |   |-- cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
    |-- src 
    |-- datasets 
    |-- predictions 
    |-- README.md 
    |-- ... 
    |-- ... 
    ```

3. Download MANO model from their official websites

    - Download `MANO_RIGHT.pkl` from [MANO](https://mano.is.tue.mpg.de/), and place it at `MeshGraphormer/src/modeling/data`.

    Please put the downloaded files under the `MeshGraphormer/src/modeling/data` directory. The data structure should follow the hierarchy below. 
    ```
    MeshGraphormer  
    |-- src  
    |   |-- modeling
    |   |   |-- data
    |   |   |   |-- MANO_RIGHT.pkl
    |-- models
    |-- datasets
    |-- predictions
    |-- README.md 
    |-- ... 
    |-- ... 
    ```
4. exit the MeshGraphormer directory when finished
    ```bash
    cd ..
    ```