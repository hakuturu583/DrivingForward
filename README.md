<p align="center">
  <h1 align="center">DrivingForward: Feed-forward 3D Gaussian Splatting for Driving Scene Reconstruction from Flexible Surround-view Input</h1>
  <p align="center">
    <a href="https://fangzhou2000.github.io/">Qijian Tian</a><sup>1</sup>
    &nbsp;·&nbsp;
    <a href="https://tanxincs.github.io/">Xin Tan</a><sup>2</sup>
    &nbsp;·&nbsp;
    <a href="https://scholar.google.com/citations?user=RN1QMPgAAAAJ">Yuan Xie</a><sup>2</sup>
    &nbsp;·&nbsp;
    <a href="https://dmcv.sjtu.edu.cn/people/">Lizhuang Ma</a><sup>1</sup>
  </p>
  <p align="center">
    <sup>1</sup>Shanghai Jiao Tong University
    <br>
    <sup>2</sup>East China Normal University
  </p>
  <h3 align="center">AAAI 2025</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2409.12753">Paper</a> | <a href="https://fangzhou2000.github.io/projects/drivingforward/">Project Page</a> | <a href="https://drive.google.com/drive/folders/1IASOPK1RQeP-nLQvJUn7WQUtb_fwGlVS">Pretrained Models</a> </h3>
</p>

## Introduction

We propose a feed-forward Gaussian Splatting model that reconstructs driving scenes from flexible sparse surround-view input.

<img src=".\assets\framework.png">

Given sparse surround-view input from vehicle-mounted cameras, our model learns
scale-aware localization for Gaussian primitives from the small overlap of spatial and temporal context views. A Gaussian
network predicts other parameters from each image individually. This feed-forward pipeline enables the real-time reconstruction
of driving scenes and the independent prediction from single-frame images supports flexible input modes. At the inference stage,
we include only the depth network and the Gaussian network, as shown in the lower part of the figure.

## Installation

To get started, clone this project, create a conda virtual environment using Python 3.8, and install the requirements:

```bash
git clone https://github.com/fangzhou2000/DrivingForward
git submodule update --init --recursive
cd DrivingForward
conda create -n DrivingForward python=3.8
conda activate DrivingForward
pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 --index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
cd models/gaussian/gaussian-splatting
pip install submodules/diff-gaussian-rasterization
cd ../../..
```

### Docker

If you use the Docker setup in `docker/`, you can put the following in a `.env` file at the repository root to avoid passing UID/GID every time:

```
UID=1000
GID=1000
```

To build and launch the container with Docker Compose:

```
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm drivingforward
```

The compose file mounts the nuScenes dataset from `/mnt/sata_ssd/nuscenes_full/v1.0` to `input_data/nuscenes/v1.0` (read-only) and writes outputs to `./output`.

### Sequence Rendering

To render a continuous sequence in eval order and save images, use `render_sequence.py`:

```
python -W ignore render_sequence.py \
  --weight_path ./weights_SF \
  --novel_view_mode SF \
  --output_dir ./results/sequence \
  --save_gt
```

Options:
```
--max_frames N   # render only N frames (0 = all)
--stride N       # render every Nth frame
```

## Datasets

### nuScenes 
* Download [nuScenes](https://www.nuscenes.org/nuscenes) official dataset
* Place the dataset in `input_data/nuscenes/`

Data should be as follows:
```
├── input_data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
│   │   ├── v1.0-trainval
```

## Running the Code

### Evaluation

Get the [pretrained models](https://drive.google.com/drive/folders/1IASOPK1RQeP-nLQvJUn7WQUtb_fwGlVS), save them to the root directory of the project, and unzip them.

For SF mode, run the following:
```
python -W ignore eval.py --weight_path ./weights_SF --novel_view_mode SF
```


For MF mode, run the following:
```
python -W ignore eval.py --weight_path ./weights_MF --novel_view_mode MF
```

### Training

For SF mode, run the following:
```
python -W ignore train.py --novel_view_mode SF
```

For MF mode, run the following:
```
python -W ignore train.py --novel_view_mode MF
```

## BibTeX
```
@inproceedings{tian2025drivingforward,
      title={DrivingForward: Feed-forward 3D Gaussian Splatting for Driving Scene Reconstruction from Flexible Surround-view Input}, 
      author={Qijian Tian and Xin Tan and Yuan Xie and Lizhuang Ma},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      year={2025}
}
```

## Acknowledgements

The project is partially based on some awesome repos: [MVSplat](https://github.com/donydchen/mvsplat), [GPS-Gaussian](https://github.com/aipixel/GPS-Gaussian), and [VFDepth](https://github.com/42dot/VFDepth). Many thanks to these projects for their excellent contributions!
