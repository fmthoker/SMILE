DATA_PATH='/home/thokerfm/20bn-something-something-v2/something-something-v2-annotations/'


export MODEL_PATH="/path_to_pretrained_checkpoint_folder/checkpoint-800.pth"
export OUTPUT_DIR='../expirements_finetune/VIT_BASE/finetune_ssv2/eval_lr_1e-3_4gpus_update_freq_1'

OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=4 \
        --master_port 12320  run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --nb_classes 174 \
        --data_set SSV2 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 8 \
        --num_sample 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --val_freq 2 \
        --num_frames 16 \
        --opt adamw \
        --lr 1e-3 \
        --layer_decay 0.7 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs 40 \
        --dist_eval \
        --test_num_segment 2 \
        --test_num_crop 3 \
        --num_workers 8 \
        --update_freq 1 \
        --enable_deepspeed    # use if configured
