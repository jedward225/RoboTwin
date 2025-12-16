#!/bin/bash
# Pi0 (InternData) Policy Evaluation Script for RoboTwin
# ======================================================
#
# Usage:
#   bash eval.sh <task_name> <task_config> <ckpt_setting> [seed] [gpu_id] [arm_mode]
#
# Examples:
#   # Evaluate on place_can_basket task with left arm
#   bash eval.sh place_can_basket demo_clean /path/to/checkpoint 0 0 left
#
#   # Evaluate with specific checkpoint directory
#   bash eval.sh place_can_basket demo_clean intern_ball_overfit 0 0 left
#
# Arguments:
#   task_name    - RoboTwin task name (e.g., place_can_basket)
#   task_config  - Config file name (e.g., demo_clean, demo_randomized)
#   ckpt_setting - Checkpoint path or experiment name
#   seed         - Random seed (default: 0)
#   gpu_id       - GPU ID (default: 0)
#   arm_mode     - 'left', 'right', or 'both' (default: left)

set -e

# Parse arguments
policy_name=pi0_intern
task_name=${1:?"Error: task_name required"}
task_config=${2:?"Error: task_config required"}
ckpt_setting=${3:?"Error: ckpt_setting required"}
seed=${4:-0}
gpu_id=${5:-0}
arm_mode=${6:-left}

# Set GPU
export CUDA_VISIBLE_DEVICES=${gpu_id}

# Get script directory and navigate to RoboTwin root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
REPO_ROOT="$(dirname "$(dirname "$ROBOTWIN_ROOT")")"

cd "$ROBOTWIN_ROOT"

echo "=============================================="
echo "Pi0 (InternData) Policy Evaluation"
echo "=============================================="
echo "Task:        ${task_name}"
echo "Config:      ${task_config}"
echo "Checkpoint:  ${ckpt_setting}"
echo "Seed:        ${seed}"
echo "GPU:         ${gpu_id}"
echo "Arm Mode:    ${arm_mode}"
echo "RoboTwin:    ${ROBOTWIN_ROOT}"
echo "Repository:  ${REPO_ROOT}"
echo "=============================================="

# Add repository to Python path
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"

# Run evaluation
python script/eval_policy.py \
    --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --arm_mode ${arm_mode}

echo "=============================================="
echo "Evaluation completed!"
echo "Results saved to: eval_result/${task_name}/${policy_name}/"
echo "=============================================="
