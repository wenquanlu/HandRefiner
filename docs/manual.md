## Manual
Arguments for executing HandRefiner.py:
<ul>
<li>--input_dir

input directory containing images to be rectified

<li>--input_img

input image to be rectified
<li>--out_dir

output directory where the rectified images will be saved to

<li>--log_json

file where the mpjpe values will be logged to
<li>--strength

control strength for ControlNet

<li>--depth_dir

directory where the depth maps will be saved to. Leaving it empty will disable this function
<li>--mask_dir

directory where the masks will be saved to. Leaving it empty will disable this function
<li>--eval (True/False)

whether evaluate the mpjpe error in fixed control strength mode, currently only works for batch size of 1.
<li>--finetuned (True/False)

whether use finetuned ControlNet trained on synthetic images as introduced in the paper
<li>--weights

path to the SD + ControlNet weights
<li>--num_samples

batch size
<li>--prompt_file

prompt file for multi-image rectification
Format for prompt file: 
```
{"img": filename, "txt": prompt}
```
Example:
```json
{"img": "img1.jpg", "txt": "a woman making a hand gesture"}
{"img": "img2.jpg", "txt": "a man making a hand gesture"}
{"img": "img3.jpg", "txt": "a man making a thumbs up gesture"}
```

<li>--prompt

prompt for single image rectification
<li>--n_iter

number of generation iteration for each image to be rectified. In general, for each input image, n_iter x num_samples number of rectified images will be produced
<li>--adaptive_control (True/False)

adaptive control strength as introduced in paper, currently only works for batch size of 1. We tend to use fixed control strength as default. 
<li>--padding_bbox

padding controls the size of masks around the hand

<li>--seed

set seed to maintain reproducibility
</ul>
