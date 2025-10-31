"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time
from collections import deque
from vla import load_vla
import numpy as np
import tensorflow as tf
import torch
from PIL import Image

class CronusServer:
    def __init__(
        self,
        cfg, 
    ) -> None:
        self.cfg = cfg
        self.action_ensemble = False
        self.adaptive_ensemble_alpha = 0.0
        self.action_ensemble_horizon = 0
        self.vla = load_vla(
            cfg.pretrained_checkpoint,
            load_for_training=False, 
            action_model_type=cfg.action_model_type,
            future_action_window_size=cfg.future_action_window_size,
            action_dim=cfg.action_dim,
        )
        self.horizon = self.vla.past_action_window_size

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        self.cognition_features_history = deque(maxlen=self.horizon)
        # if self.action_ensemble:
        #     self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        # else:
        #     self.action_ensembler = None
        self.action_ensembler = None
        self.num_image_history = 0
        self.num_cognition_features_history = 0
        self.step_index = 0
        self.execute_horizon = cfg.execute_horizon
        self.buffer_actions = None
        if self.cfg.use_bf16:
            self.vla.vlm = self.vla.vlm.to(torch.bfloat16)
        self.vla = self.vla.to("cuda").eval()

    def _add_image_to_history(self, image: Image.Image) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def _add_cognition_features_to_history(self, cognition_feature) -> None:
        self.cognition_features_history.append(cognition_feature)
        self.num_cognition_features_history = min(self.num_cognition_features_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        self.cognition_features_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.num_cognition_features_history = 0
        self.step_index = 0
        self.buffer_actions = None

    def crop_and_resize(self, image, crop_scale, batch_size):
        """
        Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
        to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
        distribution shift at test time.

        Args:
            image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
                values between [0,1].
            crop_scale: The area of the center crop with respect to the original image.
            batch_size: Batch size.
        """
        # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
        assert image.shape.ndims == 3 or image.shape.ndims == 4
        expanded_dims = False
        if image.shape.ndims == 3:
            image = tf.expand_dims(image, axis=0)
            expanded_dims = True

        # Get height and width of crop
        new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
        new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

        # Get bounding box representing crop
        height_offsets = (1 - new_heights) / 2
        width_offsets = (1 - new_widths) / 2
        bounding_boxes = tf.stack(
            [
                height_offsets,
                width_offsets,
                height_offsets + new_heights,
                width_offsets + new_widths,
            ],
            axis=1,
        )

        # Crop and then resize back up
        image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

        # Convert back to 3D Tensor (H, W, C)
        if expanded_dims:
            image = image[0]

        return image


    def get_cronusvla_action(self, vla, cfg, base_vla_name, obs, task_label, unnorm_key, center_crop=True):
        """Generates an action with the VLA policy."""
        all_images = [Image.fromarray(obs["full_image"]).convert("RGB")]
        if cfg.use_wrist_image:
            all_images.extend([Image.fromarray(obs[k]).convert("RGB") for k in obs.keys() if "wrist" in k])

        # (If trained with image augmentations) Center crop image and then resize back up to original size.
        # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
        #            the original height and width by sqrt(0.9) -- not 0.9!
        if center_crop:
            processed_images = []
            for image in all_images:
                batch_size = 1
                crop_scale = 0.9

                # Convert to TF Tensor and record original data type (should be tf.uint8)
                image = tf.convert_to_tensor(np.array(image))
                orig_dtype = image.dtype

                # Convert to data type tf.float32 and values between [0,1]
                image = tf.image.convert_image_dtype(image, tf.float32)

                # Crop and then resize back to original size
                image = self.crop_and_resize(image, crop_scale, batch_size)

                # Convert back to original data type
                image = tf.clip_by_value(image, 0, 1)
                image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

                # Convert back to PIL Image
                image: Image.Image = Image.fromarray(image.numpy())
                image = image.convert("RGB")
                processed_images.append(image)
            assert len(processed_images) == len(all_images)

        if task_label is not None:
            if task_label.lower() != self.task_description:
                self.reset(task_label.lower())
                
        self._add_image_to_history(Image.fromarray(obs["full_image"]).convert("RGB"))

        execute_i = self.step_index % self.execute_horizon
        if not cfg.use_wrist_image:
            processed_images = processed_images[0]
        actions, normalized_actions, cognition_features_current = vla.predict_action(image=processed_images, 
                                                                        instruction=self.task_description,
                                                                        unnorm_key=unnorm_key,
                                                                        do_sample=False, 
                                                                        cfg_scale=cfg.cfg_scale,
                                                                        use_ddim=cfg.use_ddim,
                                                                        num_ddim_steps=cfg.num_ddim_steps,
                                                                        cognition_features_history = self.cognition_features_history,
                                                                        num_cognition_features_history = self.num_cognition_features_history
                                                                        )
        if execute_i == 0:
            self.buffer_actions = actions

        self._add_cognition_features_to_history(cognition_features_current)
        # if self.action_ensemble:
        #     action = self.action_ensembler.ensemble_action(action)[None]

        self.step_index += 1

        return self.buffer_actions[execute_i]


