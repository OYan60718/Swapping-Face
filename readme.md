# FaceSwap Project

## Overview

This project implements a GAN-based face swapping pipeline, the model is adapted from **SimSwap** [1]. Given a *source* image (or video frame) and a *donor* identity image, the model transfers the donor's facial identity onto every face detected in the source while preserving the original pose, expression, and background. Training and evaluation are performed on the **LFW** dataset.

![SimSwap overview](/figs/simswap_overview.png)

- **Generator**: The target image is encoded into feature maps. An identity vector is extracted from the source image via a frozen **ArcFace** [2] identity extractor, which is a face recognition model for measuring face identity similarity. The vector is then injected into the bottleneck through 9 × ID-Blocks, which modulate the feature maps to carry the source identity while retaining the target's spatial attributes. The decoder upsamples the modulated features back to the full-resolution swapped image.

- **Training objective**: An **ID loss** (ArcFace cosine distance) enforces that the swapped face matches the source identity. A **GAN loss** from a discriminator and a **Weak Feature Matching (FM) loss** (matching intermediate discriminator features) jointly encourage attribute preservation and photo-realism.

---
## Setup
The project is built with Docker container, all paths in `train_config.json` are Docker-internal paths (`/workspace/…`). Build and run from the **project root** (the directory containing `FaceSwap/` and `insightface_func/`):

```bash
# Build image
cd FaceSwap
docker build -t faceswap .
cd ..
```

Download the LFW dataset to the `data` folder: https://www.kaggle.com/datasets/atulanandjha/lfwpeople?resource=download

---
## Training

```bash
# Run training (mount data, checkpoints, and ArcFace weights)
docker run --rm -it --gpus all -e CUDA_VISIBLE_DEVICES=0 -p 6002:6002 -v <root_dir>:/workspace  faceswap:latest   python FaceSwap/train.py --config FaceSwap/train_config.json
```

TensorBoard is started automatically on port `6002` by default. Access it at `http://localhost:6002`.

### Loss function

The generator losses are listed in the table below:

| Term | Description | Weight (λ) |
|---|---|---|
| `L_adv` | Adversarial loss | 1 |
| `L_id` | ArcFace cosine distance between swapped and donor embeddings | 30 |
| `L_feat` | L1 feature-matching on discriminator outputs | 10 |
| `L_rec` | L1 reconstruction loss with the same-identity but different images, applied every other step) | 10 |

The losses and key hyperparameters are adapted from SimSwap [1]. However, a smaller number of training steps is applied due to smaller dataset, and a smaller batch size is used due to limited GPU resources. In addtion, I use half of the original learning rate when training from pretrained models with a decay scheduler to avoid overfitting. 

### Checkpoint saving

Model checkpoints are saved every `model_freq` steps under `checkpoints_dir/<name>/`:

```
{step}_net_G.pth        # Generator ckpt
{step}_net_D.pth        # Discriminator ckpt
{step}_optim_G.pth      # Generator optimiser state
{step}_optim_D.pth      # Discriminator optimiser state
```

To resume training, set `continue_train: true` and `load_pretrain` to the checkpoint directory in `train_config.json`.

ONNX models are exported after training completes when `export_onnx: true` in the config. Two files are written to `onnx_output_dir`:

```
generator.onnx    # inputs: source image, id embedding from acface
                  # output: swapped_image
arcface.onnx      # input: donnor image
                  # output: id embedding
```

### Dataset Update
The training script supports update of dataset, such as adding or removing data from the dataset. To enable this, set the `refresh_freq` in `train_config.json`, which defines the frequency of checking for updates in dataset during training. Remove this setting from the config file to disable it. When new data is added/removed from the existing dataset, the dataloader update the dataset every `refresh_freq` steps and replit the dataset to `train`, `test`, and `val` sets, existing data split remains unchanged, and new data is split and added to these sets based on the split ratio, if too few images are added (smaller than 10 ids), they are added to training set only.

---
## Evaluation

### Metrics

Evaluation is implemented in `evaluate.py` and reports three metrics on the test split of LFW, the metrics are used by ArcFace [2] or SimSwap [1], as well as other face swapping literature:

| Metric | Description | Direction |
|---|---|---|
| **ID cosine similarity** | Cosine similarity between the swapped image's identity embedding and the donor's embedding. | High |
| **ID retrieval accuracy** | Top-1 accuracy of a nearest-neighbour identity search over the full gallery, determined by whether the ArcFace model retrieves the correct donor identity from the swapped images. | High |
| **Posture error** | Mean yaw, pitch, roll L1 error between the background image and the swapped image. | Low |

### Running evaluation

Set `eval_only: true` in `train_config.json`, and point `load_eval` to the checkpoint directory, `total_step` to the desired checkpoint step, then run the same training command as before to run evaluation only.

Results are printed to stdout and logged to TensorBoard under the `eval/` prefix. Qualitative results are saved to `checkpoints/<name>/eval_results/step_<N>/`.

---

## Results

### Loss curves

Training and validation losses over 50000 steps, training from a pretrained generator checkpoints provided by the authors of SimSwap.

![Loss history](/figs/loss_history.png)

Loss curves indicate that most of the loss functions, especially the validation losses, are decreasing during the training process. However, this decline is not obvious mainly due to using a pretrained model weight.

### Evaluation metrics at step 50000

ID cos similarity | ID retrieval | Posture error (deg)
|---|---|---|
| 0.711 | 98.0% | 3.687 |

Values are printed to stdout during the final evaluation pass. The quantitative results are comparable with the results provided in SimSwap's paper, which indicates that this training code is valid.

### Example Qualitative Results
Examples qualitative results show successful face swapping, while the performance can be improved by leveraging larger datasets with multiple images for each ID or using larger batch size as suggested by the authors (16 image per batch vs 3 image per batch).  Top row: source, middel row: donor, bottom row: swapped.

![results1](/figs/test_1.jpg)

![results2](/figs/test_2.jpg)


---

## Inference

Inference uses the exported ONNX models. Both **image** and **video** inputs are supported, which are detected automatically from the file extension. Sample video and images are provided in `samples` folder, images are sourced from the internet, do not distribute, video is sourced from Kaggle at https://www.kaggle.com/datasets/simongraves/deepfake-dataset?select=video.

Limitation: the current method uses the SCRFD face detector [3] from the Insightface library (https://github.com/deepinsight/insightface) to detect the positions of the face from the images/video frames and to predict the facial keypoints for alignment purposes. The implementation only supports single face detection at the moment, while multiple face detection can also be implemented with other functions in this library.

```bash
# Image inference
docker run --rm -it --gpus all -e CUDA_VISIBLE_DEVICES=0 -p 6002:6002 -v <root_dir>:/workspace  faceswap:latest   python FaceSwap/inference.py --source samples/<source>.png --donor samples/<donor>.png

# Video inference
docker run --rm -it --gpus all -e CUDA_VISIBLE_DEVICES=0 -p 6002:6002 -v <root_dir>:/workspace  faceswap:latest   python FaceSwap/inference.py --source samples/<source>.mp4 --donor samples/<donor>.png
```

The default path to the ONNX models are the saved path from the training code, if saved in different locations, please provide them via parser `--onnx-dir`.

The default setting uses GPU for realtime performance, an inference speed is printed to stdout after the inference in using video option. In my testing, the inference speed is reported as 10-20 FPS using a NVIDIA RTX 4070 GPU, supporting real-time inference. The tested CPU inference (automatic when CUDA not available) speed is around 2.5 FPS, which is expected.

Sample Outputs are illustrated below and can be found in `samples/samples_results` folder.

Images:
![image_inf](/figs/inference_results.png)

Video:
![image_inf](/figs/inference_video.gif)

---

## References

[1] Chen, R., Chen, X., Ni, B., & Ge, Y. (2020). **SimSwap: An efficient framework for high fidelity face swapping.** In *Proceedings of the 28th ACM International Conference on Multimedia* (pp. 2003–2011). ACM.

[2] Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). **ArcFace: Additive angular margin loss for deep face recognition.** In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 4690–4699).

[3] Guo, J., Deng, J., Lattas, A., & Zafeiriou, S. (2022). **Sample and computation redistribution for efficient face detection.** In *Proceedings of the International Conference on Learning Representations*.
