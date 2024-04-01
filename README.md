<h1 align="center"> HandRefiner: Refining Malformed Hands in Generated Images by Diffusion-based Conditional Inpainting </h1>
<p align="center">
<a href="[https://arxiv.org/abs/2305.02034](https://arxiv.org/abs/2311.17957)"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>

# News

**2023.12.1**
The paper is post on arxiv! 

**2023.12.29**
First code commit released.

**2024.1.7**
The preprocessor and the finetuned model have been ported to [ComfyUI controlnet](https://github.com/Fannovel16/comfyui_controlnet_aux). The preprocessor has been ported to [sd webui controlnet](https://github.com/Mikubill/sd-webui-controlnet). Thanks for all your great work!

**2024.1.15**
⚠️ When using finetuned ControlNet from this repository or [control_sd15_inpaint_depth_hand](https://huggingface.co/hr16/ControlNet-HandRefiner-pruned), I noticed many still use control strength/control weight of 1 which can result in loss of texture. As stated in the paper, we recommend using a smaller control strength (e.g. 0.4 - 0.8).

# Introduction

This is the official repository of the paper <a href="https://arxiv.org/abs/2311.17957"> HandRefiner: Refining Malformed Hands in Generated Images by Diffusion-based Conditional Inpainting </a>

<figure>
<img src="Figs/banner.png">
<figcaption align = "center"><b>Figure 1: Stable Diffusion (first two rows) and SDXL (last row) generate malformed hands (left in each pair), e.g., incorrect
number of fingers or irregular shapes, which can be effectively rectified by our HandRefiner (right in each pair). 
 </b></figcaption>
</figure>

<p>

<p align="left"> 
In this study, we introduce a lightweight post-processing solution called <b>HandRefiner</b> to correct malformed hands in generated images. HandRefiner employs a conditional inpainting
approach to rectify malformed hands while leaving other
parts of the image untouched. We leverage the hand mesh
reconstruction model that consistently adheres to the correct number of fingers and hand shape, while also being
capable of fitting the desired hand pose in the generated
image. Given a generated failed image due to malformed
hands, we utilize ControlNet modules to re-inject such correct hand information. Additionally, we uncover a phase
transition phenomenon within ControlNet as we vary the
control strength. It enables us to take advantage of more
readily available synthetic data without suffering from the
domain gap between realistic and synthetic hands.

# Visual Results
<figure>
<img src="Figs/github_results.png">
</figure>

# Installation
Check [installation.md](docs/installation.md) for installation instructions.

# Manual
Check [manual.md](docs/manual.md) for an explanation of commands to execute the HandRefiner.

# Get Started
For single image rectification:
```bash
python handrefiner.py --input_img test/1.jpg --out_dir output --strength 0.55 --weights models/inpaint_depth_control.ckpt --prompt "a man facing the camera, making a hand gesture, indoor" --seed 1
```
For multiple image rectifications:
```bash
python handrefiner.py --input_dir test --out_dir output --strength 0.55 --weights models/inpaint_depth_control.ckpt --prompt_file test/test.json --seed 1
```



# Important Q&A
<ul>
<li> <b>What kind of images can be rectified?</b></li>

Like any method, this method also has its limits. If the original hands are so bad that are inrecognisable from human eyes, then it is pretty much impossible for neural networks to fit a reasonable mesh. Also, due to the fitting nature of the method, we do not rectify the hand size. So if you have a giant malformed hand in the original image, you will still get a giant hand back in the rectified image. Thus malformed hands with hand-like shape and appropriate size can be rectified. 

<li> <b>Can we use it on SDXL images?</b>

In the paper, the SDXL images are resized to 512x512 before the rectification, because the base model used in this project is sd1.5.
Solution for SDXL:
However, it is certainly not difficult to implement it in SDXL, and I believe many implementations already have the functionality of using inpainting SDXL combined with depth controlnet to inpaint the image.
So what you can do is get the depth map and masks from the pipeline of this repository, then pipe them to the whatever implementation for SDXL you use for inpainting the image.
A caveat is that I have not tested this before, and as mentioned in the paper, since depth controlnet is not fine-tuned on the hand mesh data, it may have a high rate of failed inpainting. In that case, you can use the technique mentioned in the paper, using available synthetic data to fine-tune the depth sdxl controlnet, for example, using these two datasets here [[1]](https://synthesis.ai/static-gestures-dataset/)[[2]](https://synthesis.ai/animated-gestures-dataset/), then you can adjust control strength to get the desired texture and appearance.

<li> <b>What if the generation failed?</b>

The first thing is to check the depth map, if the depth map is bad, you can consider using a different mesh reconstruction model to reconstruct the mesh. 

Second things is to check if the masks of hands fully cover the malformed hands, some malformed hand can have very long fingers so it may not be covered by the detected masks, to fix this
1. Consider using a greater padding by adjusting the pad parameter in the argument
2. Provide a hand-drawn mask

If all of the previous steps are ok, then you may need to regenerate several times or try different control strengths. <- changing the seed can be very helpful.

<li> <b>Since the small hands is a limitation mentioned in the paper, what is the appropriate hand size for the SD v1.5 weight?</b>

Generally, hands with size at least 60px &#215; 60px is recommended for the current weights. To make it applicable for small hands, consider scale up the image using some super-resolution methods.

<li> <b>How to contribute to this project?</b>

In the last decade, the CV community has produced dozens of highly accurate mesh reconstruction models, in this project we use the recent SOTA model Mesh Graphormer on the FreiHAND benchmark. However, it is very welcome to contribute to this project by porting other models here, I have written a template parent class for models under preprocessor folder.

<li> <b>Can I use it for Anime hands or other styles?</b>

As long as the hand detection model and the mesh reconstruction model are able to detect the hands and reconstruct meshes, it should work for other styles. However, from my understanding, these models are not trained on cartoon or anime images, so there is a great chance that the mesh reconstruction stage may fail. 

</ul>

## Comments
- Our codebase builds heavily on [stable-diffusion](https://github.com/CompVis/stable-diffusion), [ControlNet](https://github.com/lllyasviel/ControlNet) and [MeshGraphormer](https://github.com/microsoft/MeshGraphormer).

## Citation

If you find HandRefiner helpful, please consider giving this repo a star :star: and citing:

```
@article{lu2023handrefiner,
   title={HandRefiner: Refining Malformed Hands in Generated Images by Diffusion-based Conditional Inpainting},
   author={Wenquan Lu and Yufei Xu and Jing Zhang and Chaoyue Wang and Dacheng Tao},
   journal={arXiv preprint arXiv:2311.17957},
   year={2023}
}
```
