# TOAM-YOLO

Official implementation of "TOAM-YOLO: A Tiny Object-Aware Multi-Expert YOLO Framework for Diverse Domains" (TMLR 2026).

Built upon [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0), this repository includes the TOAM-YOLO model definition, training script, and dataset configurations used for experiments in the paper.

## License
Released under AGPL-3.0, built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). See `LICENSE`.

## Installation

Create a Python environment:

```bash
conda create -n toam-yolo python=3.12
conda activate toam-yolo
```

Install dependencies from the repository requirements file:

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

Build the local DCNv3 extension:

```bash
cd ultralytics/ops_dcnv3
bash make.sh
cd ../..
```

Install FlashAttention for TOA-MoE area attention:

```bash
pip install flash-attn --no-build-isolation
```

If you need a different PyTorch CUDA build, install the matching PyTorch command from the official PyTorch setup page before installing `requirements.txt`.

## Usage

Prepare a YOLO-format dataset with the following structure:

```text
datasets/your_dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## Dataset Setup

Download the datasets and place them following the structure above:

- SeaPerson: https://cove.thecvf.com/datasets/695
- TinyPerson: http://vision.ucas.ac.cn/sources
- VisDrone: https://github.com/VisDrone/VisDrone-Dataset
- BCCD: https://github.com/Shenggan/BCCD_Dataset
- CBC: https://github.com/MahmudulAlam/Complete-Blood-Cell-Count-Dataset

Train TOAM-YOLO from the repository root:

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

Evaluate or run inference with a trained checkpoint:

```bash
python - <<'PY'
from ultralytics import YOLO

model = YOLO("runs/train/toam_yolo_experiment/weights/best.pt")
model.predict(source="path/to/images_or_video", imgsz=640, save=True)
PY
```

Run multiple model/dataset combinations using `main.bash`:

1. Edit `MODELS` and `DATASETS` in `main.bash`.
2. Run:

```bash
bash main.bash
```

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex

```
