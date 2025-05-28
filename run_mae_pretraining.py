import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
from timm.models import create_model
from optim_factory import create_optimizer
from datasets import build_pretraining_dataset
from engine_for_pretraining import train_one_epoch
from utils_mae import NativeScalerWithGradNormCount as NativeScaler
import utils_mae as utils
import modeling_pretrain
from timm.models.vision_transformer import vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224
from modeling_pretrain import FeatureExtractor


def get_args():
    parser = argparse.ArgumentParser('VideoMAE pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)

    # Model parameters
    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')

    parser.add_argument('--mask_type', default='tube', choices=['random', 'tube', 'tubelet'],
                        type=str, help='masked strategy of video tokens/patches')
    
    parser.add_argument('--sub_mask_type', default='tube+picked_frame_visible', choices=['tube', 'tube+picked_frame_visible', 'tube+traj_mask'],
                        type=str, help='sub masked strategy of tubelet masking')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')

    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')

    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
                        
    parser.add_argument('--normlize_target', default=True, type=bool,
                        help='normalized the target patch pixels')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Dataset parameters
    parser.add_argument('--data_path', default='/path/to/list_kinetics-400', type=str,
                        help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Tubelet params
    parser.add_argument('--add_tubelets', action='store_true')
    parser.set_defaults(add_tubelets=False)
    parser.add_argument('--use_objects', action='store_true') 
    parser.set_defaults(use_objects=False)
    parser.add_argument('--objects_path', type=str, default=None)
    parser.add_argument('--motion_type', type=str, default='gaussian')
    parser.add_argument('--scales', type=str, default='[32, 48, 56, 64, 96, 128]')
    parser.add_argument('--loc_velocity', type=int, default=12)
    parser.add_argument('--visible_frames', type=str, default=None) # not used
    parser.add_argument('--traj_unmask_ratio', type=float, default=0.1)

    #dino params
    parser.add_argument('--target_type', default='pixel', choices=['pixel', 'dino_v1', 'clip'], type=str, help='define target type for loss')
    parser.add_argument('--distillation_teacher', default="clip_b", type=str, choices=['dino_s', 'dino_b', 'clip_b'], help='distillation teacher model')

    # multiple sampling
    parser.add_argument('--multiple_sampling', action='store_true')
    # for 2nd stage training
    parser.add_argument('--first_stage_path', type=str, default=None)

    return parser.parse_args()



def get_teacher_student_models(args):
    print(f"Creating model: {args.model}")
    if args.target_type=='pixel':
        dec_dim = 1536
    elif 'dino' in args.target_type or 'clip' in args.target_type:
        if args.distillation_teacher == 'dino_s':
            dec_dim = 384
        elif args.distillation_teacher == 'dino_b' or args.distillation_teacher == 'clip_b':

            dec_dim = 768

    student_model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        decoder_depth=args.decoder_depth,
        use_checkpoint=args.use_checkpoint,
        decoder_num_classes=dec_dim,
    )

    if args.target_type == 'dino_v1':

        # load dino 
        if args.distillation_teacher == 'dino_s':
            pretraining = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            teacher_model = vit_small_patch16_224(pretrained=False)
        elif args.distillation_teacher == 'dino_b':
            pretraining = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            teacher_model = vit_base_patch16_224(pretrained=False)

        msg =teacher_model.load_state_dict(pretraining.state_dict(), strict=False)
        teacher_model = FeatureExtractor(teacher_model, args.input_size, 16)
        print(msg)
        teacher_model.eval()

    elif args.target_type == 'clip':

         # load clip 
         from utils_viclip.config import Config
         from utils_viclip.config_utils import setup_viclip
         from tasks.shared_utils import setup_model
         from models_viclip.viclip import ViCLIP

         config = setup_viclip('configs/config.py')
         model_cls = eval(config.model.get('model_cls', 'ViCLIP'))
         teacher_model = setup_model(
             config,
             model_cls=model_cls,
             has_decoder=False,
             pretrain=False,
             find_unused_parameters=False,
         )
         teacher_model.eval()
    else:
        teacher_model = None


    return student_model, teacher_model

def load_first_stage(model,args):
    if args.first_stage_path is not None:
        checkpoint = torch.load(args.first_stage_path, map_location='cpu')
        print("loading first stage from ",args.first_stage_path)
        checkpoint_model = checkpoint['model']
        utils.load_state_dict(model, checkpoint_model)


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    student_model, teacher_model = get_teacher_student_models(args)

    patch_size = student_model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1]) # [8, 14, 14]
    print(f"Window Size = {args.window_size}")
    args.patch_size = patch_size


    # Start from pretrained first stage model
    if args.first_stage_path is not None:
           load_first_stage(student_model,args)

    # get dataset
    dataset_train = build_pretraining_dataset(args)


    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank

    total_batch_size = args.batch_size * num_tasks
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))


    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size if not args.multiple_sampling else int(args.batch_size/2),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=utils.seed_worker
    )

    student_model.to(device)
    if teacher_model is not None:
        teacher_model.to(device)
    model_without_ddp = student_model
    n_parameters = sum(p.numel() for p in student_model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = student_model.module

    optimizer = create_optimizer(
        args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=student_model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    torch.cuda.empty_cache()
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            student_model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            patch_size=patch_size[0],
            normlize_target=args.normlize_target,
            teacher_model = teacher_model,
            target_type=args.target_type,
            multiple_sampling=args.multiple_sampling,
        )
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=student_model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        #if (epoch + 1) % 2 == 0:
            #exit(0)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
