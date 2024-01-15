## Installation Instructions

1. Clone HandRefiner to your local repository
2. Install <a href="https://github.com/microsoft/MeshGraphormer">MeshGraphormer</a> to <b>HandRefiner/MeshGraphormer</b> following instructions in [meshgraphormer.md](meshgraphormer.md). (If encountrer any error, you can also refer to original documentations in the Meshgraphormer).
    Please also comply to Mesh Graphormer's license when using it in this project.
3. Make sure you are on the 'HandRefiner/' directory for the following steps, refer to [requirements.txt](../requirements.txt) for packages required for the project. 
4. Install Mediapipe:
    ```bash
    pip install -q mediapipe==0.10.0
    cd preprocessor
    wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    ```
    ```
5. Download weights, there are two sets of weights can be used:
    - Inpaint Stable Diffusion weights [sd-v1-5-inpainting.ckpt](https://huggingface.co/runwayml/stable-diffusion-inpainting/blob/main/sd-v1-5-inpainting.ckpt) and Depth controlnet weights [control_v11f1p_sd15_depth.pth](https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11f1p_sd15_depth.pth). Put sd-v1-5-inpainting.ckpt and control_v11f1p_sd15_depth.pth in HandRefiner/models/ folder. To use these weights, set --finetuned flag to False when executing the HandRefiner. 
    - Finetuned weights [inpaint_depth_control.ckpt](https://drive.google.com/file/d/1eD2Lnfk0KZols68mVahcVfNx3GnYdHxo/view?usp=sharing) as introduced in the paper. Put inpaint_depth_control.ckpt in the HandRefiner/models/ folder. A control strength of 0.4 - 0.8 is recommended for the finetuned weights, we use 0.55 in the evaluation of paper. Alternatively, adaptive control strength can be used by setting --adaptive_control flag to True, though the inference time is much longer.

    Finetuned weights are more adaptable to complex gestures, and their inpainting is more harmonious. You can also attempt to use original weights while the failure rate could be higher.

6. Test if installation succeeds:

    For single image rectification:
    ```bash
    python handrefiner.py --input_img test/1.jpg --out_dir output --strength 0.55 --weights models/inpaint_depth_control.ckpt --prompt "a man facing the camera, making a hand gesture, indoor" --seed 1
    ```
    For multiple image rectifications:
    ```bash
    python handrefiner.py --input_dir test --out_dir output --strength 0.55 --weights models/inpaint_depth_control.ckpt --prompt_file test/test.json --seed 1
    ```

