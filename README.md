# 🏆 [CVPR 2026] PromptStereo: Zero-Shot Stereo Matching via Structure and Motion Prompts 🏆

Xianqi Wang, Hao Yang, Hangtian Wang, Junda Cheng, Gangwei Xu, Min Lin, Xin Yang

Huazhong University of Science and Technology, Optics Valley Laboratory

<a href='https://arxiv.org/abs/2603.01650'><img src='https://img.shields.io/badge/arXiv-2501.08654-b31b1b?logo=arxiv' alt='arxiv'></a>

![PromptStereo](PromptStereo.png)

## 🔄 Update

* **04/01/2026:** Update the evaluation code.

## ⚙️ Environment

```
conda create -n promptstereo python=3.12
conda activate promptstereo

pip install tqdm numpy wandb opt_einsum hydra-core
pip install imageio scipy torch torchvision opencv-python matplotlib
pip install xformers accelerate scikit-image
```

## 📂 Required Data

Data for evaluation:

* [KITTI 2012](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)

* [KITTI 2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

* [Middlebury](https://vision.middlebury.edu/stereo/submit3)

* [ETH3D](https://www.eth3d.net/datasets)

* [DrivingStereo](https://drivingstereo-dataset.github.io)

* [Booster](https://amsacta.unibo.it/id/eprint/6876)

## 🎁 Pre-Trained Model

| Model | Link |
| :-: | :-: |
| Depth-Anything-V2-Large | [Download 🤗](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) |
| PromptStereo-Unlimited-192 | [Download 🤗](https://huggingface.co/Windsrain/PromptStereo/tree/main) |

## 📊 Evaluation

```
accelerate launch evaluate_stereo.py
```

Default settings use bf16 precision with faster speed but a very little performance degration, you can set accelerator.mixed_precision to null to obtain entire performance.

## 🔔 Notification

Complete demo, training, and fine-tuning code will be released soon. More versions of models such as adapting to large disparity will also be released soon.

## 🙏 Acknowledgement

This project is based on [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), and [MonSter](https://github.com/Junda24/MonSter). We thank the original authors for their excellent works.