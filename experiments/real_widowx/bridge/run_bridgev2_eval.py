"""
run_bridgev2_eval.py

Runs a model in a real-world Bridge V2 environment.

Usage:
    # OpenVLA:
    python experiments/robot/bridge/run_bridgev2_eval.py --model_family openvla --pretrained_checkpoint openvla/openvla-7b
"""

import os
import sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.getcwd())
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union, Optional
import numpy as np
from PIL import Image
import draccus
import argparse
import cv2 as cv
import json
import math
import torch
from vla import load_vla
from vla.adaptive_ensemble import AdaptiveEnsembler
# Append current directory so that interpreter can find experiments.robot
from experiments.real_widowx.bridge.bridgev2_utils import (
    get_next_task_label,
    get_preprocessed_image,
    get_widowx_env,
    refresh_obs,
    save_rollout_data,
    save_rollout_video,
)
from experiments.real_widowx.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
)
from collections import deque


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "cronusvla"                               # Model family
    pretrained_checkpoint: Union[str, Path] = ""                # Pretrained checkpoint path
    load_in_8bit: bool = False                                  # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                                  # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = False                                   # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # WidowX environment-specific parameters
    #################################################################################################################
    host_ip: str = "localhost"
    port: int = 5000

    # Note: Setting initial orientation with a 30 degree offset, which makes the robot appear more natural
    init_ee_pos: List[float] = field(default_factory=lambda: [0.3, -0.09, 0.26])
    init_ee_quat: List[float] = field(default_factory=lambda: [0, -0.259, 0, -0.966])
    bounds: List[List[float]] = field(default_factory=lambda: [
            [0.1, -0.20, -0.01, -1.57, 0],
            [0.45, 0.25, 0.30, 1.57, 0],
        ]
    )


    blocking: bool = False                                      # Whether to use blocking control
    max_episodes: int = 50                                      # Max number of episodes to run
    max_steps: int = 300                                         # Max number of timesteps per episode
    control_frequency: float = 5                               # WidowX control frequency
    #################################################################################################################
    # Utils
    #################################################################################################################
    save_data: bool = False                                     # Whether to save rollout data (images, actions, etc.)
    camera_topics: List[Dict[str, str]] = field(default_factory=lambda: [{"name": "/D435/color/image_raw", "info_name": "/D435/color/camera_info"}])
    unnorm_key: str = 'bridge_dataset'
    image_size: list = field(default_factory=lambda: [224, 224])
    action_model_type: str = "DiT-B"
    future_action_window_size: int = 15
    cfg_scale: float = 1.5
    use_bf16: bool = True
    action_dim: int = 7
    action_ensemble: bool = True
    action_ensemble_horizon: int = 2
    adaptive_ensemble_alpha: float = 0.1
    action_chunking: bool = False
    action_chunking_window: int = None
    # fmt: on

class CronusVLAService:
    def __init__(
        self,
        saved_model_path: str = None,
        unnorm_key: str = None,
        image_size: list[int] = [224, 224],
        action_model_type: str = "DiT-B",  # choose from ['DiT-Small', 'DiT-Base', 'DiT-Large'] to match the model weight
        future_action_window_size: int = 15,
        cfg_scale: float = 1.5,
        num_ddim_steps: int = 10, 
        use_ddim: bool = True,
        use_bf16: bool = True,
        action_dim: int = 7,
        action_ensemble: bool = True,
        adaptive_ensemble_alpha: float = 0.1,
        action_ensemble_horizon: int = 2,
        action_chunking: bool = False,
        action_chunking_window: Optional[int] = None,
        args=None
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        assert not (action_chunking and action_ensemble), "Now 'action_chunking' and 'action_ensemble' cannot both be True."  

        self.unnorm_key = unnorm_key

        print(f"*** unnorm_key: {unnorm_key} ***")
        self.vla = load_vla(
          saved_model_path,
          load_for_training=False, 
          action_model_type=action_model_type,
          future_action_window_size=future_action_window_size,
          action_dim=action_dim,
        )
        if use_bf16:
            self.vla.vlm = self.vla.vlm.to(torch.bfloat16)
        self.vla = self.vla.to("cuda").eval()
        self.cfg_scale = cfg_scale
        self.horizon = self.vla.past_action_window_size
        self.image_size = image_size
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.action_chunking = action_chunking
        self.action_chunking_window = action_chunking_window
        self.cognition_features_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_cognition_features_history = 0
        self.args = args
        self.reset()

    def _add_cognition_features_to_history(self, cognition_feature) -> None:
        self.cognition_features_history.append(cognition_feature)
        self.num_cognition_features_history = min(self.num_cognition_features_history + 1, self.horizon)

    def reset(self) -> None:
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.cognition_features_history.clear()
        self.num_cognition_features_history = 0

    def step(
        self, image: Image.Image, 
        task_description: Optional[str] = None, 
        reset: bool = False, 
        *args, **kwargs,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: Path to the image file
            task_description: Optional[str], task description
        Output:
            action: list[float], the ensembled 7-DoFs action of End-effector and gripper

        """

        # image: Image.Image = Image.open(image)

        # [IMPORTANT!]: Please process the input images here in exactly the same way as the images
        # were processed during finetuning to ensure alignment between inference and training.
        # Make sure, as much as possible, that the gripper is visible in the processed images.
        resized_image = resize_image(image, size=self.image_size)
        unnormed_actions, normalized_actions, cognition_features_current = self.vla.predict_action(
            image=resized_image, 
            instruction=task_description, 
            unnorm_key=self.unnorm_key, 
            do_sample=False, 
            cfg_scale=self.cfg_scale, 
            use_ddim=self.use_ddim, 
            num_ddim_steps=self.num_ddim_steps,
            cognition_features_history = self.cognition_features_history,
            num_cognition_features_history = self.num_cognition_features_history
            )
        self._add_cognition_features_to_history(cognition_features_current)

        if self.action_ensemble:
            unnormed_actions = self.action_ensembler.ensemble_action(unnormed_actions)
            # Translate the value of the gripper's open/close state to 0 or 1.
            # Please adjust this line according to the control mode of different grippers.
            unnormed_actions[6] = unnormed_actions[6] >= 0.95
            action = np.array(unnormed_actions) #.tolist()
        elif self.action_chunking:
            # [IMPORTANT!]: Please modify the code here to output multiple actions at once.
            # The code below only outputs the first action in the chunking.
            # The chunking window size can be adjusted by modifying the 'action_chunking_window' parameter.
            if self.action_chunking_window is not None:
                chunked_actions = []
                for i in range(0, self.action_chunking_window):
                    unnormed_actions[i][6] = unnormed_actions[i][6] >= 0.95
                    chunked_actions.append(unnormed_actions[i].tolist())
                action = chunked_actions
            else:
                raise ValueError("Please specify the 'action_chunking_window' when using action chunking.")
        else:
            # Output the first action in the chunking. Can be modified to output multiple actions at once.
            unnormed_actions = unnormed_actions[0]
            unnormed_actions[6] = unnormed_actions[6] >= 0.95
            action = unnormed_actions.tolist()

        print(f"Instruction: {task_description}")
        # print(f"action: {action}")
        # print(f"Model path: {self.args.saved_model_path} at port {self.args.port}")

        if reset:
            self.reset()
            
        return action

# [IMPORTANT!]: Please modify the image processing code here to ensure that the input images  
# are handled in exactly the same way as during the finetuning phase.
# Make sure, as much as possible, that the gripper is visible in the processed images.
def resize_image(image: Image, size=(224, 224), shift_to_left=0):
    w, h = image.size
    assert h <= w, "Height should be less than width"
    left_margin = (w - h) // 2 - shift_to_left
    left_margin = min(max(left_margin, 0), w - h)
    image = image.crop((left_margin, 0, left_margin + h, h))

    image = image.resize(size, resample=Image.LANCZOS)
    
    # image = scale_and_resize(image, target_size=(224, 224), scale=0.9, margin_w_ratio=0.5, margin_h_ratio=0.5)
    return image

# Here the image is first center cropped and then resized back to its original size 
# because random crop data augmentation was used during finetuning.
def scale_and_resize(image : Image, target_size=(224, 224), scale=0.9, margin_w_ratio=0.5, margin_h_ratio=0.5):
    w, h = image.size
    new_w = int(w * math.sqrt(scale))
    new_h = int(h * math.sqrt(scale))
    margin_w_max = w - new_w
    margin_h_max = h - new_h
    margin_w = int(margin_w_max * margin_w_ratio)
    margin_h = int(margin_h_max * margin_h_ratio)
    image = image.crop((margin_w, margin_h, margin_w + new_w, margin_h + new_h))
    image = image.resize(target_size, resample=Image.LANCZOS)
    return image


@draccus.wrap()
def eval_model_in_bridge_env(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    assert not cfg.center_crop, "`center_crop` should be disabled for Bridge evaluations!"
    # [OpenVLA] Set action un-normalization key
    # cfg.unnorm_key = "bridge_orig"
    # Load model
    model = CronusVLAService(
        saved_model_path=cfg.pretrained_checkpoint,
        unnorm_key=cfg.unnorm_key,
        image_size=cfg.image_size,
        action_model_type=cfg.action_model_type,
        future_action_window_size=cfg.future_action_window_size,
        cfg_scale=cfg.cfg_scale,
        use_bf16=cfg.use_bf16,
        action_dim=cfg.action_dim,
        action_ensemble=cfg.action_ensemble,
        adaptive_ensemble_alpha=cfg.adaptive_ensemble_alpha,
        action_ensemble_horizon=cfg.action_ensemble_horizon,
        action_chunking=cfg.action_chunking,
        action_chunking_window=cfg.action_chunking_window,
        args=cfg
    )

    assert cfg.model_family == "cronusvla"
    # Initialize the WidowX environment
    env = get_widowx_env(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    task_label = ""
    episode_idx = 0
    while episode_idx < cfg.max_episodes:
        # Get task description from user
        task_label = get_next_task_label(task_label)

        # Reset environment
        obs, _ = env.reset()

        # Setup
        t = 0
        step_duration = 1.0 / cfg.control_frequency
        replay_images = []
        if cfg.save_data:
            rollout_images = []
            rollout_states = []
            rollout_actions = []

        # Start episode
        input(f"Press Enter to start episode {episode_idx+1}...")
        print("Starting episode... Press Ctrl-C to terminate episode early!")
        last_tstamp = time.time()
        model.reset()
        while t < cfg.max_steps:
            try:
                curr_tstamp = time.time()
                if curr_tstamp > last_tstamp + step_duration:
                    print(f"t: {t}")
                    print(f"Previous step elapsed time (sec): {curr_tstamp - last_tstamp:.2f}")
                    last_tstamp = time.time()

                    # Refresh the camera image and proprioceptive state
                    obs = refresh_obs(obs, env)

                    # Save full (not preprocessed) image for replay video
                    replay_images.append(obs["full_image"])

                    # Get preprocessed image
                    # obs["full_image"] = get_preprocessed_image(obs, resize_size)

                    # Query model to get action
                    image = Image.fromarray(obs["full_image"])
                    image = image.convert("RGB")
                    action = model.step(image, task_label)
                    # [If saving rollout data] Save preprocessed image, robot state, and action
                    if cfg.save_data:
                        rollout_images.append(obs["full_image"])
                        rollout_states.append(obs["proprio"])
                        rollout_actions.append(action)

                    # Execute action
                    print("action:", action)
                    obs, _, _, _, _ = env.step(action)
                    t += 1

            except (KeyboardInterrupt, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    print("\nCaught KeyboardInterrupt: Terminating episode early.")
                else:
                    print(f"\nCaught exception: {e}")
                break

        # Save a replay video of the episode
        save_rollout_video(replay_images, episode_idx)

        # [If saving rollout data] Save rollout data
        if cfg.save_data:
            save_rollout_data(replay_images, rollout_images, rollout_states, rollout_actions, idx=episode_idx)

        # Redo episode or continue
        if input("Enter 'r' if you want to redo the episode, or press Enter to continue: ") != "r":
            episode_idx += 1


if __name__ == "__main__":
    eval_model_in_bridge_env()
