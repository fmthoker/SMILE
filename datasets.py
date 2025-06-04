import os
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator, TubeletMaskingGenerator
from kinetics import VideoClsDataset, VideoMAE
from ssv2 import SSVideoClsDataset
import synthetic_tubelets as synthetic_tubelets
import ast
import random

class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        self.add_tubelets = args.add_tubelets
        self.mask_type = args.mask_type

        # original transform without adding tubelets
        self.transform_original = transforms.Compose([
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])

        # tubelet transform
        if args.add_tubelets:
            scales = ast.literal_eval(args.scales)
            
            self.tubelets = synthetic_tubelets.PatchMask(
                    use_objects=args.use_objects,
                    objects_path=args.objects_path,
                    region_sampler=dict(
                        scales=scales,
                        ratios=[0.5, 0.67, 0.75, 1.0, 1.33, 1.50, 2.0],
                        scale_jitter=0.18,
                        num_rois=2,
                    ),
                    key_frame_probs=[0.5, 0.3, 0.2],
                    loc_velocity=12,
                    rot_velocity=6,
                    size_velocity=0.025,
                    label_prob=1.0,
                    motion_type=args.motion_type,
                    patch_transformation='rotation',)

            
            self.transform1 = transforms.Compose([
                self.train_augmentation,
                self.tubelets,
            ])
            self.transform2 = transforms.Compose([Stack(roll=False),
                ToTorchFormatTensor(div=True),
                normalize,
            ])
        else:
            self.transform = self.transform_original

        self.original_masked_position_generator = TubeMaskingGenerator(
            args.window_size, args.mask_ratio
        )
        
        if args.mask_type == 'tube':
            self.masked_position_generator = self.original_masked_position_generator
        elif args.mask_type == 'tubelet':
            self.masked_position_generator = TubeletMaskingGenerator(
                args.window_size, args.mask_ratio, args.visible_frames, args.sub_mask_type
            )
        else:
            raise NotImplemented

    
    def __call__(self, images):
        process_data, _, traj_rois = self.ComposedTransform(images)

        if self.mask_type == 'tubelet' and traj_rois is not None:
            return process_data, self.masked_position_generator(traj_rois)
        else:
            return process_data, self.masked_position_generator()

    def ComposedTransform(self, images):
        traj_rois = None
        
        if self.add_tubelets:
            data = self.transform1(images)
            process_data, traj_rois = data[:-1], data[-1]
            process_data, _ = self.transform2(process_data)
        else:
            process_data, _ = self.transform(images)

        return process_data, _, traj_rois

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        try:
            self.transform
        except:
            repr += "  transform = %s,\n" % (str(self.transform1) + str(self.transform2))
        else:
            repr += "  transform = %s,\n" % str(self.transform)
            
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False)
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args):
    if args.data_set == 'Kinetics-400' or args.data_set == "Mini-Kinetics":
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            if 'Mini' in args.data_set:
                anno_path = os.path.join(args.data_path, 'train_mini_kinetics.csv')
            else:
                anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            if 'Mini' in args.data_set:
                anno_path = os.path.join(args.data_path, 'test_mini_kinetics.csv')
            else:
                anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            if 'Mini' in args.data_set:
                anno_path = os.path.join(args.data_path, 'val_mini_kinetics.csv')
            else:
                anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        if 'Mini' in args.data_set:
            nb_classes = 200
        else:
            nb_classes = 400
    
    elif args.data_set == 'SSV2' or args.data_set == 'SSV2-Mini':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            if 'Mini' in args.data_set:
                 anno_path = os.path.join(args.data_path, 'train_mini.csv')
            else:
                 anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 174

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
