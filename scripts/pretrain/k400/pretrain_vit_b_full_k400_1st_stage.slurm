#!/bin/bash
#SBATCH --nodes=4
#SBATCH --gpus-per-node=2
#SBATCH --time=0:59:00
#SBATCH --constraint=v100
#SBATCH --mem=100GB
#SBATCH --cpus-per-gpu=6
#SBATCH --array=0-100%1

module purge

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))
echo $MASTER_ADDR
echo $MASTER_PORT

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=300

# path to to pretraining file
DATA_PATH='/ibex/user/thokerfm/datasets/kinetics_from_ivi/labels/train.csv'

# path to folder containing object categories
OBJECTS_PATH='/ibex/user/thokerfm/datasets/micheal_objects/cleaned/images'

# path to output ckpt dierectory
OUTPUT_DIR='expirements/VIT_BASE/k400_smile_pretraining_first_stage'


#Notes

# For 600 epoch training we run the first stage training for 150 epochs (effectively equals to 300 epochs, as we use multiple sampling).

# For 1200 epoch training we run the first stage training for 400 epochs (effectively equals to 800 epochs, as we use multiple sampling).

# We use early stopping and use the exact following hyper parameters 

source activate fgvssl

JOB_NAME=$1
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
CPUS_PER_TASK=${CPUS_PER_TASK:-6}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

srun -p --job-name=${JOB_NAME} \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --kill-on-bad-exit=1 \
        ${SRUN_ARGS} \
        python -u run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tubelet \
        --sub_mask_type 'tube+traj_mask' \
        --target_type clip \
        --distillation_teacher clip_b  \
        --mask_ratio 0.8 \
        --traj_unmask_ratio 0.2 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 32 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 1 \
        --epochs 801 \
        --add_tubelets \
        --use_objects \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --objects_path ${OBJECTS_PATH}
        --multiple_sampling  \
