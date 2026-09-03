<h1 align="center">LatentSync</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.09262)
[![arXiv](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/ByteDance/LatentSync-1.6)
[![arXiv](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/fffiloni/LatentSync)

</div>

## 🔥 Updates

- `2025/06/11`: We released **LatentSync 1.6**, which is trained on 512 $\times$ 512 resolution videos to mitigate the blurriness problem. Watch the demo [here](docs/changelog_v1.6.md).

- `2025/03/14`: We released **LatentSync 1.5**, which **(1)** improves temporal consistency via adding temporal layer, **(2)** improves performance on Chinese videos and **(3)** reduces the VRAM requirement of the stage2 training to **20 GB** through a series of optimizations. Learn more details [here](docs/changelog_v1.5.md).

## 📖 Introduction

We present *LatentSync*, an end-to-end lip-sync method based on audio-conditioned latent diffusion models without any intermediate motion representation, diverging from previous diffusion-based lip-sync methods based on pixel-space diffusion or two-stage generation. Our framework can leverage the powerful capabilities of Stable Diffusion to directly model complex audio-visual correlations.

## 🏗️ Framework

<p align="center">
<img src="docs/framework.png" width=100%>
<p>

LatentSync uses the [Whisper](https://github.com/openai/whisper) to convert melspectrogram into audio embeddings, which are then integrated into the U-Net via cross-attention layers. The reference and masked frames are channel-wise concatenated with noised latents as the input of U-Net. In the training process, we use a one-step method to get estimated clean latents from predicted noises, which are then decoded to obtain the estimated clean frames. The TREPA, [LPIPS](https://arxiv.org/abs/1801.03924) and [SyncNet](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf) losses are added in the pixel space.

## 🎬 Demo

<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
        <td width="50%"><b>Original video</b></td>
        <td width="50%"><b>Lip-synced video</b></td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/b778e3c3-ba25-455d-bdf3-d89db0aa75f4 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/ac791682-1541-4e6a-aa11-edd9427b977e controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/6d4f4afd-6547-428d-8484-09dc53a19ecf controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/b4723d08-c1d4-4237-8251-09c43eb77a6a controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/fb4dc4c1-cc98-43dd-a211-1ff8f843fcfa controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/7c6ca513-d068-4aa9-8a82-4dfd9063ac4e controls preload></video>
    </td>
  </tr>
  <tr>
    <td width=300px>
      <video src=https://github.com/user-attachments/assets/0756acef-2f43-4b66-90ba-6dc1d1216904 controls preload></video>
    </td>
    <td width=300px>
      <video src=https://github.com/user-attachments/assets/663ff13d-d716-4a35-8faa-9dcfe955e6a5 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/0f7f9845-68b2-4165-bd08-c7bbe01a0e52 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/c34fe89d-0c09-4de3-8601-3d01229a69e3 controls preload></video>
    </td>
  </tr>
</table>

(Photorealistic videos are filmed by contracted models, and anime videos are from [VASA-1](https://www.microsoft.com/en-us/research/project/vasa-1/))

## 📑 Open-source Plan

- [x] Inference code and checkpoints
- [x] Data processing pipeline
- [x] Training code

## 🔧 Setting up the Environment

Install the required packages and download the checkpoints via:

```bash
source setup_env.sh
```

If the download is successful, the checkpoints should appear as follows:

```
./checkpoints/
|-- latentsync_unet.pt
|-- whisper
|   `-- tiny.pt
```

Or you can download `latentsync_unet.pt` and `tiny.pt` manually from our [HuggingFace repo](https://huggingface.co/ByteDance/LatentSync-1.6)

### GPU face preprocessing

The standard `opencv-python` and `mediapipe` wheels run YuNet and FaceMesh on the CPU. With a
CUDA device, LatentSync now uses ONNX Runtime CUDA automatically for both the MIT-licensed,
dynamic-input YuNet 2026may detector and the Apache-2.0 MediaPipe Face Landmarker v2 model
(478 points), using the parity-tested ONNX port and ROI transform from
[`yakhyo/mediapipe-face-mesh-onnx`](https://github.com/yakhyo/mediapipe-face-mesh-onnx).
The pinned models are downloaded by the setup script or automatically on first use, and their
SHA-256 hashes are checked. CUDA is used from the first frame; there is no CPU-only initialization
step.

OpenCV describes 2026may as a dynamic-shape re-export of 2023mar, not a newly trained detector;
the benefit here is reliable variable-resolution ONNX Runtime input rather than changed accuracy.

For single-presenter input up to 4K, detection adapts to the source resolution and downsizes only
when the longest edge exceeds 640 pixels. Detection boxes and landmarks are mapped back onto the
original-resolution frame before LatentSync crops and aligns the face. Override the detection
resolution only when smaller faces need more detail:

```bash
# Windows PowerShell
$env:LATENTSYNC_FACE_DETECTION_SIZE = "960"

# Linux/macOS
export LATENTSYNC_FACE_DETECTION_SIZE=960
```

For single-presenter footage, YuNet runs on the first frame and then every 5 frames by default.
The landmark model still runs on every frame and updates the face ROI, so lip and head motion are
not sampled at the lower rate. If tracking loses confidence, YuNet runs again immediately. Change
the periodic detection interval when needed:

```bash
# Windows PowerShell (1 restores detection on every frame)
$env:LATENTSYNC_FACE_DETECTION_INTERVAL = "5"

# Linux/macOS
export LATENTSYNC_FACE_DETECTION_INTERVAL=5
```

At startup, an unavailable CUDA provider produces an explicit warning before falling back to the
CPU MediaPipe implementation. If both `onnxruntime` and `onnxruntime-gpu` are installed, uninstall
the CPU-only `onnxruntime` package so that `CUDAExecutionProvider` is available.

### Face alignment scale

By default, the MediaPipe triangle is contracted to `0.935` of its original size around its
centroid. This keeps the anchor centre unchanged and makes the aligned face about 7% larger,
matching the scale of the LatentSync training crops more closely. The calibrated value is fixed
for preprocessing, Gradio, regular inference, and batch inference so every entry point uses the
same crop geometry.

## 🚀 Inference

Minimum VRAM for inference:

- **8 GB** with LatentSync 1.5
- **18 GB** with LatentSync 1.6

There are two ways to perform inference:

### 1. Gradio App

Run the Gradio app for inference:

```bash
python gradio_app.py
```

### 2. Command Line Interface

Run the script for inference:

```bash
./inference.sh
```

You can try adjusting the following inference parameters to achieve better results:

- `inference_steps` [20-50]: A higher value improves visual quality but slows down the generation speed.
- `guidance_scale` [1.0-3.0]: A higher value improves lip-sync accuracy but may cause the video distortion or jitter.

## 🔄 Data Processing Pipeline

The complete data processing pipeline includes the following steps:

1. Remove the broken video files.
2. Resample the video FPS to 25, and resample the audio to 16000 Hz.
3. Scene detect via [PySceneDetect](https://github.com/Breakthrough/PySceneDetect).
4. Split each video into 5-10 second segments.
5. Affine transform the faces according to the landmarks detected by MediaPipe Face Mesh, then resize to 256 $\times$ 256.

First, put your videos in the `my_data/raw` directory.

Then run the script to execute the data processing pipeline:

```bash
./data_processing_pipeline.sh
```

The processed videos will be saved in the `affine_transformed` directory. Each step will generate a new directory to prevent the need to redo the entire pipeline in case the process is interrupted by an unexpected error.

Then generate file list for training:

```bash
python tools/write_fileslist.py
```

## 🏋️‍♂️ Training U-Net

Before training, you should process the data as described above. We released a pretrained SyncNet with 94% accuracy on both VoxCeleb2 and HDTF datasets for the supervision of U-Net training. You can execute the following command to download this SyncNet checkpoint:

```bash
huggingface-cli download ByteDance/LatentSync-1.6 stable_syncnet.pt --local-dir checkpoints
```

If all the preparations are complete, you can train the U-Net with the following script:

```bash
./train_unet.sh
```

We prepared several UNet configuration files in the ``configs/unet`` directory, each corresponding to a specific training setup:

- `stage1.yaml`: Stage1 training, requires **23 GB** VRAM.
- `stage2.yaml`: Stage2 training with optimal performance, requires **30 GB** VRAM.
- `stage2_efficient.yaml`: Efficient Stage 2 training, requires **20 GB** VRAM. It may lead to slight degradation in visual quality and temporal consistency compared with `stage2.yaml`, suitable for users with consumer-grade GPUs, such as the RTX 3090.
- `stage1_512.yaml`: Stage1 training on 512 $\times$ 512 resolution videos, requires **30 GB** VRAM.
- `stage2_512.yaml`: Stage2 training on 512 $\times$ 512 resolution videos, requires **55 GB** VRAM.

Also remember to change the parameters in U-Net config file to specify the data directory, checkpoint save path, and other training hyperparameters. For convenience, we prepared a script for writing a data files list. Run the following command:

```bash
python -m tools.write_fileslist
```

## 🏋️‍♂️ Training SyncNet

In case you want to train SyncNet on your own datasets, you can run the following script. The data processing pipeline for SyncNet is the same as U-Net. 

```bash
./train_syncnet.sh
```

After `validations_steps` training, the loss charts will be saved in `train_output_dir`. They contain both the training and validation loss. If you want to customize the architecture of SyncNet for different image resolutions and input frame lengths, please follow the [guide](docs/syncnet_arch.md).

## 📊 Evaluation

You can evaluate the accuracy of SyncNet on a dataset by running the following script:

```bash
./eval/eval_syncnet_acc.sh
```

Note that our released SyncNet is trained on data processed through our data processing pipeline, including affine transformation. Therefore, before evaluation, the test data must first be processed using the provided pipeline.

## ⚖️ Licensing

The code in this repository is released under the Apache License 2.0. **Model weights are licensed separately** — `latentsync_unet.pt` is distributed under the [CreativeML Open RAIL++-M License](https://huggingface.co/ByteDance/LatentSync-1.6), which permits commercial use but attaches use-based restrictions that must be passed on to downstream recipients.

A complete AI Bill of Materials — every code module and weight file with its source, license, byte count and SHA-256 — is in [`docs/AIBOM.html`](docs/AIBOM.html). Open it in a browser; it is a single self-contained file.

Read it before redistributing weights, shipping an SDK, or handing fine-tuned checkpoints to a customer.

## 🙏 Acknowledgement

- Our code is built on [AnimateDiff](https://github.com/guoyww/AnimateDiff). 
- Some code are borrowed from [MuseTalk](https://github.com/TMElyralab/MuseTalk) (MIT).

Thanks for their generous contributions to the open-source community!

## 📖 Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@article{li2024latentsync,
  title={LatentSync: Taming Audio-Conditioned Latent Diffusion Models for Lip Sync with SyncNet Supervision},
  author={Li, Chunyu and Zhang, Chao and Xu, Weikai and Lin, Jingyu and Xie, Jinghui and Feng, Weiguo and Peng, Bingyue and Chen, Cunjian and Xing, Weiwei},
  journal={arXiv preprint arXiv:2412.09262},
  year={2024}
}
```
