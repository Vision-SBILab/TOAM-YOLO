#!/bin/bash

# TOAM-YOLO Experimental Training Script
# 
# This script automates training experiments for TOAM-YOLO framework across
# multiple datasets and configurations for comprehensive evaluation.
#
# Reference: TOAM-YOLO paper - Experimental setup and evaluation protocol
#
# Usage: 
#   1. Configure model and dataset paths below
#   2. Set experimental parameters (epochs, batch size, device)
#   3. Run: bash main.bash
#
# Output:
#   - Training runs saved to PROJECT_DIR
#   - Results aggregated in RESULTS_CSV
#   - Run summaries with hyperparameters saved per experiment

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE PATHS FOR YOUR SETUP
# =============================================================================

# Model configurations (add paths to your model YAML files)
MODELS=(
    # "configs/toam-yolo.yaml"
    # "configs/toe-yolo.yaml"
    # "yolov12s.yaml"
    # "yolov12n.yaml"
)

# Dataset configurations (add paths to your dataset YAML files)
DATASETS=(
    # datasets/SeaPerson/data.yaml
    # datasets/tinyperson/data.yaml
    # datasets/VisDrone/data.yaml
    # datasets/bccd/data.yaml
    # datasets/cbc/data.yaml
)

# =============================================================================
# EXPERIMENTAL PARAMETERS
# =============================================================================

EPOCHS=100                            # Training epochs
BATCH_SIZE=16                        # Batch size (adjust based on GPU memory)
DEVICE='0'                             # GPU device ID (use '0,1' for multi-GPU)
RESULTS_CSV="toam_yolo_results.csv"    # Results aggregation file
PROJECT_DIR="experiments/toam_yolo"    # Base directory for all runs
PYTHON_SCRIPT="main.py"                # Training script
RUN_DATE=$(date +"%m%d")               # Date suffix for run identification

# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

echo "Starting TOAM-YOLO experimental evaluation"
echo "Models: ${#MODELS[@]}, Datasets: ${#DATASETS[@]}"
echo "Total experiments: $((${#MODELS[@]} * ${#DATASETS[@]}))"
echo "----------------------------------------"

# Loop through all model-dataset combinations
for model in "${MODELS[@]}"; do
    model_name=$(basename "$model" .yaml)
    for data in "${DATASETS[@]}"; do
        dataset_name=$(basename "$data" .yaml)
        run_name="${model_name}_${dataset_name}_bs${BATCH_SIZE}_${RUN_DATE}"
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        SUMMARY_FILE="experiment_logs/run_summary_${run_name}_${TIMESTAMP}.txt"

        echo "Running experiment: Model=${model_name}, Dataset=${dataset_name}, Epochs=${EPOCHS}"

        # Create experiment log directory
        mkdir -p experiment_logs

        # Create comprehensive run summary for reproducibility
        {
            echo "TOAM-YOLO Experiment Summary"
            echo "Run Name: $run_name"
            echo "Timestamp: $TIMESTAMP"
            echo ""
            echo "Experimental Configuration:"
            echo "--------------------------"
            echo "Model Configuration: $model"
            echo "Dataset Configuration: $data"
            echo "Training Epochs: $EPOCHS"
            echo "Batch Size: $BATCH_SIZE"
            echo "GPU Device: $DEVICE"
            echo "Results File: $RESULTS_CSV"
            echo "Project Directory: $PROJECT_DIR"
            echo ""
            echo "Training Script Contents:"
            echo "------------------------"
            cat "$PYTHON_SCRIPT"
            echo ""
            echo "System Information:"
            echo "-------------------"
            nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
            echo "Python: $(python --version)"
            echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
        } > "$SUMMARY_FILE"

        # Execute training experiment
        CUDA_VISIBLE_DEVICES=$DEVICE python "$PYTHON_SCRIPT" \
            --model_yaml "$model" \
            --data_yaml "$data" \
            --run_name "$run_name" \
            --project "$PROJECT_DIR" \
            --results_csv "$RESULTS_CSV" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE"

        echo "Completed experiment: $run_name"
        echo "Summary saved to: $SUMMARY_FILE"
        echo "----------------------------------------"
    done
done

echo "All experiments completed successfully."
echo "Results available in: $RESULTS_CSV"
echo "Individual run summaries in: experiment_logs/"