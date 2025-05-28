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

DATA_PATH='/ibex/user/thokerfm/datasets/kinetics_from_ivi/labels/train.csv'

OUTPUT_DIR='/ibex/user/thokerfm/expirements_pretrain_smile/VIT_BASE/IVI_data_full_k400_clip_motion_mask_80_traj_20_multiple_sampling'
OUTPUT_DIR='./test_run'

OBJECTS_PATH='/ibex/user/jianl0b/Dataset/Fida_file_1/video_images/micheal_objects/cleaned/images'

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
