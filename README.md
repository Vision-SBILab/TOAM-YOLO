# TOAM-YOLO

Official PyTorch implementation of **"TOAM-YOLO: A Tiny Object-Aware Multi-Expert YOLO Framework for Diverse Domains"**

This repository is built upon the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework and contains the modifications proposed in our paper, including the TOA-MoE module, the high-resolution P2 detection pathway, the BiFPN-style feature fusion network, DCNv3 integration, and CARAFE upsampling. The repository also includes the model definitions, training scripts, and dataset configurations used in our experiments.

---

## License

This repository is released under the **AGPL-3.0** license, following the licensing terms of the underlying [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework. See the `LICENSE` file for details.

---

## Installation

Create a Python environment:

```bash
conda create -n toam-yolo python=3.12
conda activate toam-yolo
```

Install PyTorch (choose the appropriate CUDA version from the official PyTorch website if necessary), followed by the project dependencies:

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

Build the DCNv3 extension:

```bash
cd ultralytics/ops_dcnv3
bash make.sh
cd ../..
```

Install FlashAttention (required by the Area Attention blocks inherited from YOLOv12):

```bash
pip install flash-attn --no-build-isolation
```

---

## Dataset Setup

Prepare a standard YOLO-format dataset:

```text
datasets/
└── your_dataset/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

The experiments in the paper use the following publicly available datasets:

- [SeaPerson](https://cove.thecvf.com/datasets/695)
- [TinyPerson](http://vision.ucas.ac.cn/sources)
- [VisDrone](https://github.com/VisDrone/VisDrone-Dataset)
- [BCCD](https://github.com/Shenggan/BCCD_Dataset)
- [CBC](https://github.com/MahmudulAlam/Complete-Blood-Cell-Count-Dataset)

Download the datasets and update the corresponding dataset YAML files before training.

---

## Training

Training is performed using `main.py`, which wraps the Ultralytics training pipeline with the configurations used in our paper.

Example:

```bash
python main.py \
    --model_yaml configs/toam-yolo.yaml \
    --data_yaml datasets/your_dataset/data.yaml \
    --run_name toam_yolo_experiment \
    --epochs 500 \
    --batch_size 16 \
    --project runs/train \
    --results_csv results.csv
```

---

## Inference

Run inference using a trained checkpoint:

```python
from ultralytics import YOLO

model = YOLO("runs/train/toam_yolo_experiment/weights/best.pt")

model.predict(
    source="path/to/images_or_video",
    imgsz=640,
    save=True
)
```

---

## Reproducing Paper Experiments

The repository includes the configuration files used in the paper.

To run multiple model and dataset combinations automatically:

1. Edit the `MODELS` and `DATASETS` variables in `main.bash`.
2. Run:

```bash
bash main.bash
```

---

## Citation

## Citation

If you find this work useful in your research, please consider citing our paper.

```bibtex
@article{
sharma2026toamyolo,
title={{TOAM}-{YOLO}: A Tiny Object-Aware Multi-Expert {YOLO} Framework for Diverse Domains},
author={Vaibhav Sharma and Arnesh Batra and Arush Gumber and Ritu Gupta and Anubha Gupta},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=2lIE1tgmRN},
note={}
}
```

---

## Acknowledgements

This work builds upon the excellent [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework. We thank the Ultralytics team for making their implementation publicly available.
