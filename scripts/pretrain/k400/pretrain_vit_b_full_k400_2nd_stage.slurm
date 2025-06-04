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

# path to pretraining file
DATA_PATH='/ibex/user/thokerfm/datasets/kinetics_from_ivi/labels/train.csv'

#Notes
# For 600 epoch training we load the 150th ckpt from first stage (checkpoint-150.pth)

# We stop the second stage training after 150  epochs (effectively equals to 300 epochs more, as we use multiple sampling) = Total 300 + 300 epochs.

# For 1200 epoch training we the 400th ckpt from first stage (checkpoint-150.pth)
# we stop the second stage training after 200  epochs (effectively equals to 400 epochs more, as we use multiple sampling) = Total 800 + 400 epochs.


#e.g.
# path to the ckpt from first pretraining stage
FIRST_STAGE_CKPT='expirements/VIT_BASE/k400_smile_pretraining_first_stage/checkpoint-150.pth'

# path to output ckpt dierectory
OUTPUT_DIR='expirements/VIT_BASE/k400_smile_pretraining_2nd_stage'


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
        --mask_type tube \
        --target_type clip \
        --distillation_teacher clip_b  \
        --mask_ratio 0.8 \
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
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --multiple_sampling  \
        --first_stage_path ${FIRST_STAGE_CKPT} \
