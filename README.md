<h1 align="center"> HandRefiner: Refining Malformed Hands in Generated Images by Diffusion-based Conditional Inpainting </h1>
<p align="center">

# News

**2023.11.28**



# Introduction

This is the official repository of the paper <a href=""> HandRefiner: Refining Malformed Hands in Generated Images by Diffusion-based Conditional Inpainting </a>

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
