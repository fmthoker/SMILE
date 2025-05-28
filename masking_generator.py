import numpy as np
import random
import ast

class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask


class TubeletMaskingGenerator:
    def __init__(self, input_size, mask_ratio, visible_frames, mask_type="tube", traj_unmask_ratio=0.1):
        self.tube_masking_generator = TubeMaskingGenerator(input_size, mask_ratio)
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame
        self.patch_size = 16
        self.traj_unmask_ratio = traj_unmask_ratio
        if visible_frames is not None:
            visible_list = ast.literal_eval(visible_frames)
            self.visible_frames = [int(element) for element in visible_list]
        else:
            self.visible_frames = None

        self.mask_type = mask_type
    
    def _balance_num_masks(self, combined_mask, 
                        unmasked_object_patches_index, 
                        unmasked_non_object_patches_index, 
                        masked_object_patches_index, 
                        tube_masked_index=None, 
                        tube_unmasked_index=None):
        current_masks = np.sum(combined_mask)
        num_diff = np.abs(self.total_masks - current_masks)

        if tube_masked_index is None or tube_unmasked_index is None:
            # tubelet masking without tube mask
            # if too many masked patches, we unmask some patches
            if current_masks > self.total_masks:
                picked_index = masked_object_patches_index[np.random.choice(masked_object_patches_index.size, size=int(num_diff), replace=False)]
                combined_mask[picked_index] = 0.
            # if too few masked patches, we first try to mask non-object patches, if not enough, then we mask protected patches
            elif current_masks < self.total_masks:
                if num_diff <= len(unmasked_non_object_patches_index):
                    picked_index = unmasked_non_object_patches_index[np.random.choice(unmasked_non_object_patches_index.size, size=int(num_diff), replace=False)]
                    combined_mask[picked_index] = 1.
                else:
                    combined_mask[unmasked_non_object_patches_index] = 1.
                    picked_index = unmasked_object_patches_index[np.random.choice(unmasked_object_patches_index.size, size=int(num_diff - len(unmasked_non_object_patches_index)), replace=False)]
                    combined_mask[picked_index] = 1.
        else:
            # if too many masked patches, we first try to unmask tube masked patches, if not enough, then we unmask object patches
            tube_masked_non_object_index = np.array(list(set(tube_masked_index) - set(masked_object_patches_index) - set(unmasked_object_patches_index)))
            if current_masks > self.total_masks:
                if num_diff <= len(tube_masked_non_object_index):
                    picked_index = tube_masked_non_object_index[np.random.choice(tube_masked_non_object_index.size, size=int(num_diff), replace=False)]
                    combined_mask[picked_index] = 0.
                else:
                    combined_mask[tube_masked_non_object_index] = 0.
                    picked_index = masked_object_patches_index[np.random.choice(masked_object_patches_index.size, size=int(num_diff - len(tube_masked_non_object_index)), replace=False)]
                    combined_mask[picked_index] = 0.
            # if too few masked patches, we first try to mask non-object patches, if not enough, then we mask protected patches
            elif current_masks < self.total_masks:
                tube_unmasked_non_object_index = np.array(list(set(tube_unmasked_index) - set(masked_object_patches_index) - set(unmasked_object_patches_index)))
                if num_diff <= len(tube_unmasked_non_object_index):
                    picked_index = tube_unmasked_non_object_index[np.random.choice(tube_unmasked_non_object_index.size, size=int(num_diff), replace=False)]
                    combined_mask[picked_index] = 1.
                else:
                    combined_mask[tube_unmasked_non_object_index] = 1.
                    picked_index = unmasked_object_patches_index[np.random.choice(unmasked_object_patches_index.size, size=int(num_diff - len(tube_unmasked_non_object_index)), replace=False)]
                    combined_mask[picked_index] = 1.
        
        balanced_mask = combined_mask
        return balanced_mask

    def __repr__(self):
            repr_str = "Maks: total patches {}, mask patches {}".format(
                self.total_patches, self.total_masks
            )
            return repr_str

    # 1 in mask array means masked, 0 means unmasked
    def __call__(self, traj_rois):
        # generate original VideoMAE tube mask and intialize the tube mask index
        tube_mask = self.tube_masking_generator()
        tube_masked_index = None
        tube_unmasked_index = None

        # initialize mask
        num_tubelet, num_frame, box = traj_rois.shape
        assert num_frame % 2 == 0 and self.frames == (num_frame // 2)
        combined_mask = np.zeros((num_frame // 2, self.height, self.width))
        # assume patch size is (2, 16, 16) so mask shape should be (8, 14, 14)
        # we combine the traj_rois of two consecutive frames to one large traj_rois

        # pick one tubelet that is not masked
        if self.visible_frames is None:
            picked_frame = np.random.randint(0, (num_frame // 2))
            picked_list = [picked_frame]
        else:
            picked_list = self.visible_frames
            
        # combined mask 1 means object patches that should be masked, 2 means object patches that should not be masked, 0 means non-object patches
        for roi_idx, roi in enumerate(traj_rois):
            for i in range(num_frame // 2):
                min_x = min( (roi[2 * i][0], roi[2 * i + 1][0]) )
                max_x = max( (roi[2 * i][2], roi[2 * i + 1][2]) )
                min_y = min( (roi[2 * i][1], roi[2 * i + 1][1]) )
                max_y = max( (roi[2 * i][3], roi[2 * i + 1][3]) )

                patch_index_x_min = max( int(np.floor(min_x / self.patch_size)), 0)
                patch_index_x_max = min( int(np.ceil(max_x / self.patch_size)) + 1, 14)
                patch_index_y_min = max( int(np.floor(min_y / self.patch_size)), 0)
                patch_index_y_max = min( int(np.ceil(max_y / self.patch_size)) + 1, 14)
                
                if i in picked_list:
                    combined_mask[i][patch_index_y_min:patch_index_y_max, patch_index_x_min:patch_index_x_max] = 2.
                else:
                    combined_mask[i][patch_index_y_min:patch_index_y_max, patch_index_x_min:patch_index_x_max] = 1.

        combined_mask = combined_mask.flatten()
        masked_object_patches_index = np.where(combined_mask == 1.)[0]
        unmasked_non_object_patches_index = np.where(combined_mask == 0.)[0]
        unmasked_object_patches_index = np.where(combined_mask == 2.)[0]
        combined_mask[unmasked_object_patches_index] = 0.
        
        tube_masked_index = np.where(tube_mask == 1.)[0]
        tube_unmasked_index = np.where(tube_mask == 0.)[0]

        # combine tubelet mask and tube mask
        combined_mask = np.bitwise_or(combined_mask.astype(bool), tube_mask.astype(bool)).astype(np.float32)
        
        if self.mask_type == "tube+picked_frame_visible":
            # unmasked the protected patches
            combined_mask[unmasked_object_patches_index] = 0.

        elif self.mask_type == "tube+traj_mask":
            # get index of unmasked traj patches
            traj_unmask_ratio = self.traj_unmask_ratio
            traj_patches_index = np.array(list(set(masked_object_patches_index) | set(unmasked_object_patches_index)))
            unmasked_traj_patches_index = traj_patches_index[np.random.choice(traj_patches_index.size, size=int(traj_unmask_ratio * len(traj_patches_index)), replace=False)]

            # mask the whole traj
            combined_mask[traj_patches_index] = 1.
            # unmask those selected patches
            combined_mask[unmasked_traj_patches_index] = 0.

            # update indexes
            unmasked_object_patches_index = unmasked_traj_patches_index
            masked_object_patches_index = np.array(list(set(traj_patches_index) - set(unmasked_traj_patches_index)))


        # balance masked patch number
        mask = self._balance_num_masks(combined_mask, 
                                       unmasked_object_patches_index, 
                                       unmasked_non_object_patches_index,
                                       masked_object_patches_index,
                                       tube_masked_index, 
                                       tube_unmasked_index)
        

        assert np.sum(mask) == self.total_masks
        return mask

    
