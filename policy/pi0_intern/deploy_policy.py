"""
Pi0 Policy Adapter for RoboTwin Evaluation

This adapter allows evaluating pi0 models (trained on InternData) in RoboTwin.

IMPORTANT NOTE:
- InternData uses single-arm Franka robot: action_dim = 8 (7 joints + 1 gripper)
- RoboTwin uses dual-arm robot: action_dim = 14 (left_arm:7 + left_gripper:1 + right_arm:7 + right_gripper:1)

Adaptation Strategy:
- For single-arm tasks: Use only left arm, right arm stays at initial position
- For dual-arm tasks: Need to retrain model with dual-arm data

Usage:
    cd /home/jjliu/HistoryAwareVLA/third_party/RoboTwin
    bash policy/pi0_intern/eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id>
"""

import sys
import os
import numpy as np
import torch
import cv2

# Add paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOTWIN_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
REPO_ROOT = os.path.dirname(os.path.dirname(ROBOTWIN_ROOT))

sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models import model as _model


class Pi0InternPolicy:
    """
    Pi0 policy wrapper for RoboTwin evaluation.

    Handles:
    - Image preprocessing (resize, normalize)
    - State/action dimension mapping
    - Action chunk execution
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda:0',
        action_horizon: int = 10,
        action_dim: int = 32,
        max_token_len: int = 48,
        image_size: int = 224,
        arm_mode: str = 'left',  # 'left', 'right', or 'both'
        embodiment: str = 'dual_arm',  # 'franka' for single-arm, 'dual_arm' for aloha
    ):
        self.device = torch.device(device)
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.max_token_len = max_token_len
        self.image_size = image_size
        self.arm_mode = arm_mode
        self.embodiment = embodiment

        # InternData action mapping (8 dim)
        # [joint_0, joint_1, ..., joint_6, gripper_openness]
        self.intern_action_dim = 8

        # RoboTwin action mapping
        if embodiment == 'franka':
            # Single-arm Franka: 8 dim [arm:7, gripper:1]
            self.robotwin_action_dim = 8
        else:
            # Dual-arm (aloha): 14 dim [left_arm:7, left_gripper:1, right_arm:7, right_gripper:1]
            self.robotwin_action_dim = 14

        # Initialize tokenizer
        self.tokenizer = PaligemmaTokenizer(max_len=max_token_len)

        # Load model
        self._load_model(checkpoint_path)

        # State buffers
        self.obs_history = []
        self.action_buffer = []
        self.current_instruction = None

    def _load_model(self, checkpoint_path: str):
        """Load pi0 model from checkpoint."""
        import openpi.models.pi0_config as pi0_config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        import safetensors.torch

        # Create model config
        model_config = pi0_config.Pi0Config(
            dtype="bfloat16",
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            max_token_len=self.max_token_len,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pi05=False,
        )

        # Initialize model
        self.model = PI0Pytorch(model_config).to(self.device)

        # Load weights
        if os.path.isdir(checkpoint_path):
            # Checkpoint directory format
            model_path = os.path.join(checkpoint_path, "model.safetensors")
        else:
            model_path = checkpoint_path

        if os.path.exists(model_path):
            safetensors.torch.load_model(self.model, model_path, device=str(self.device))
            print(f"Loaded model from: {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model.eval()

    def _preprocess_image(self, rgb: np.ndarray) -> np.ndarray:
        """
        Preprocess image for pi0.

        Args:
            rgb: (H, W, 3) uint8 image

        Returns:
            (H', W', 3) uint8 image resized to image_size
        """
        if rgb.shape[0] != self.image_size or rgb.shape[1] != self.image_size:
            rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return rgb

    def _robotwin_qpos_to_intern_state(self, qpos: np.ndarray) -> np.ndarray:
        """
        Convert RoboTwin qpos to InternData state format (padded to 32).

        For franka (single-arm):
            RoboTwin qpos: [arm:7, gripper:1] = 8 dim
        For dual_arm (aloha):
            RoboTwin qpos: [left_arm:7, left_gripper:1, right_arm:7, right_gripper:1] = 16 dim

        InternData state: [joint_position:7, gripper_position:1, gripper_pose:6]
        Note: InternData doesn't have explicit gripper_pose, so we pad with zeros.
        """
        if self.embodiment == 'franka':
            # Single-arm Franka: qpos is 8 dim
            joint_pos = qpos[:7]
            gripper_pos = qpos[7:8]
        else:
            # Dual-arm (aloha): qpos is 16 dim
            if self.arm_mode == 'left':
                joint_pos = qpos[:7]  # left arm joints
                gripper_pos = qpos[7:8]  # left gripper
            elif self.arm_mode == 'right':
                joint_pos = qpos[8:15]  # right arm joints
                gripper_pos = qpos[15:16]  # right gripper
            else:
                # For 'both', use left arm as primary
                joint_pos = qpos[:7]
                gripper_pos = qpos[7:8]

        # Create state in InternData format
        # [joint_position:7, gripper_position:1, gripper_pose:6 (zeros)]
        state = np.zeros(self.action_dim, dtype=np.float32)
        state[:7] = joint_pos
        state[7:8] = gripper_pos
        # state[8:14] = gripper_pose (zeros for now)

        return state

    def _intern_action_to_robotwin(self, action: np.ndarray) -> np.ndarray:
        """
        Convert InternData action (8 dim) to RoboTwin action format.

        InternData action: [joint_position:7, gripper_openness:1]

        For franka (single-arm): output 8 dim [arm:7, gripper:1]
        For dual_arm (aloha): output 14 dim [left_arm:7, left_gripper:1, right_arm:7, right_gripper:1]
        """
        # Extract from padded action
        joint_action = action[:7]
        gripper_action = action[7]

        # Always output 14-dim actions for RoboTwin (even for franka)
        # RoboTwin's take_action expects dual-arm format
        robotwin_action = np.zeros(14, dtype=np.float32)

        if self.arm_mode == 'left':
            robotwin_action[:7] = joint_action
            robotwin_action[7] = gripper_action
            # Right arm stays at zero (will use initial position)
        elif self.arm_mode == 'right':
            robotwin_action[8:15] = joint_action
            robotwin_action[15] = gripper_action
            # Left arm stays at zero
        else:
            # 'both' - mirror to both arms
            robotwin_action[:7] = joint_action
            robotwin_action[7] = gripper_action
            robotwin_action[8:15] = joint_action
            robotwin_action[15] = gripper_action

        return robotwin_action

    def predict(self, obs: dict, instruction: str = None) -> np.ndarray:
        """
        Run inference and get action chunk.

        Args:
            obs: Encoded observation dict
            instruction: Language instruction

        Returns:
            actions: (action_horizon, 14) action chunk for RoboTwin
        """
        if instruction is not None:
            self.current_instruction = instruction

        # Prepare images (already preprocessed)
        images = {
            "base_0_rgb": torch.from_numpy(obs["head_cam"]).unsqueeze(0).to(self.device),
            "left_wrist_0_rgb": torch.from_numpy(obs["left_cam"]).unsqueeze(0).to(self.device),
            "right_wrist_0_rgb": torch.from_numpy(obs["right_cam"]).unsqueeze(0).to(self.device),
        }

        image_masks = {
            "base_0_rgb": torch.tensor([True]).to(self.device),
            "left_wrist_0_rgb": torch.tensor([True]).to(self.device),
            "right_wrist_0_rgb": torch.tensor([True]).to(self.device),
        }

        # Prepare state
        state = torch.from_numpy(obs["state"]).unsqueeze(0).to(torch.float32).to(self.device)

        # Tokenize instruction
        prompt = self.current_instruction or "manipulation task"
        tokens, mask = self.tokenizer.tokenize(prompt)
        tokenized_prompt = torch.from_numpy(tokens).unsqueeze(0).to(self.device)
        tokenized_prompt_mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        # Create Observation
        obs_dict = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
            "tokenized_prompt": tokenized_prompt,
            "tokenized_prompt_mask": tokenized_prompt_mask,
        }
        observation = _model.Observation.from_dict(obs_dict)

        # Run inference
        with torch.no_grad():
            # sample_actions returns (1, action_horizon, action_dim)
            actions = self.model.sample_actions(
                device=self.device,
                observation=observation,
                num_steps=10,  # denoising steps
            )

        # Convert to numpy and map to RoboTwin format
        actions_np = actions.cpu().numpy()[0]  # (action_horizon, 32)

        # Map each action to RoboTwin format
        robotwin_actions = []
        for i in range(self.action_horizon):
            robotwin_action = self._intern_action_to_robotwin(actions_np[i])
            robotwin_actions.append(robotwin_action)

        return np.array(robotwin_actions)  # (action_horizon, 14)

    def reset(self):
        """Reset policy state for new episode."""
        self.obs_history = []
        self.action_buffer = []
        self.current_instruction = None


# Global model instance
_model_instance = None


def encode_obs(observation: dict) -> dict:
    """
    Convert RoboTwin observation to pi0 input format.

    Args:
        observation: RoboTwin observation dict

    Returns:
        Encoded observation for pi0
    """
    # Extract images
    head_rgb = observation["observation"]["head_camera"]["rgb"]  # (H, W, 3)
    left_rgb = observation["observation"]["left_camera"]["rgb"]
    right_rgb = observation["observation"]["right_camera"]["rgb"]

    # Resize to 224x224
    head_rgb = cv2.resize(head_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    left_rgb = cv2.resize(left_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    right_rgb = cv2.resize(right_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Extract qpos
    qpos = np.array(observation["joint_action"]["vector"], dtype=np.float32)  # (16,)

    # Convert qpos to InternData state format
    global _model_instance
    if _model_instance is not None:
        state = _model_instance._robotwin_qpos_to_intern_state(qpos)
    else:
        # Default: use left arm
        state = np.zeros(32, dtype=np.float32)
        state[:7] = qpos[:7]
        state[7:8] = qpos[7:8]

    return {
        "head_cam": head_rgb,      # (224, 224, 3) uint8
        "left_cam": left_rgb,      # (224, 224, 3) uint8
        "right_cam": right_rgb,    # (224, 224, 3) uint8
        "state": state,            # (32,) float32
        "qpos": qpos,              # (16,) original RoboTwin qpos
    }


def get_model(usr_args: dict):
    """
    Initialize pi0 model for RoboTwin evaluation.

    Args:
        usr_args: Configuration dict containing:
            - task_name: Task name
            - ckpt_setting: Checkpoint setting/experiment name
            - checkpoint_dir: Base directory for checkpoints
            - device: CUDA device
            - arm_mode: 'left', 'right', or 'both'
            - embodiment: 'franka' or 'dual_arm' (auto-detected from left_arm_dim)
    """
    global _model_instance

    # Construct checkpoint path
    checkpoint_dir = usr_args.get('checkpoint_dir', './policy/pi0_intern/checkpoints')
    task_name = usr_args.get('task_name', 'default')
    ckpt_setting = usr_args.get('ckpt_setting', 'latest')

    # Try different checkpoint path formats
    possible_paths = [
        os.path.join(checkpoint_dir, task_name, ckpt_setting),
        os.path.join(checkpoint_dir, ckpt_setting),
        os.path.join(checkpoint_dir, task_name, ckpt_setting, "model.safetensors"),
        ckpt_setting,  # Direct path
    ]

    checkpoint_path = None
    for path in possible_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
        if os.path.exists(path + "/model.safetensors"):
            checkpoint_path = path
            break

    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Checkpoint not found. Tried paths:\n" +
            "\n".join(f"  - {p}" for p in possible_paths)
        )

    # Auto-detect embodiment from arm dimensions
    # franka-panda has 7 joints, aloha has 7+7=14
    left_arm_dim = usr_args.get('left_arm_dim', 7)
    right_arm_dim = usr_args.get('right_arm_dim', 7)

    # Check if this is single-arm (franka) or dual-arm (aloha)
    # If right_arm_dim exists and > 0, it's dual-arm
    if right_arm_dim > 0 and left_arm_dim > 0:
        # Check the embodiment config name
        embodiment_name = usr_args.get('embodiment_name', '')
        if 'franka' in embodiment_name.lower():
            embodiment = 'franka'
        else:
            embodiment = 'dual_arm'
    else:
        embodiment = 'franka'

    # Allow explicit override
    embodiment = usr_args.get('embodiment', embodiment)

    print(f"Using embodiment: {embodiment} (left_arm_dim={left_arm_dim}, right_arm_dim={right_arm_dim})")

    _model_instance = Pi0InternPolicy(
        checkpoint_path=checkpoint_path,
        device=usr_args.get('device', 'cuda:0'),
        action_horizon=usr_args.get('action_horizon', 10),
        action_dim=usr_args.get('action_dim', 32),
        max_token_len=usr_args.get('max_token_len', 48),
        arm_mode=usr_args.get('arm_mode', 'left'),
        embodiment=embodiment,
    )

    return _model_instance


def eval(TASK_ENV, model: Pi0InternPolicy, observation: dict):
    """
    Execute one evaluation step.

    Args:
        TASK_ENV: RoboTwin task environment
        model: Pi0InternPolicy instance
        observation: Current observation from TASK_ENV.get_obs()
    """
    # Encode observation
    obs = encode_obs(observation)

    # Get language instruction
    instruction = TASK_ENV.get_instruction()

    # Get action chunk from model
    actions = model.predict(obs, instruction=instruction)  # (action_horizon, 14)

    # Execute action chunk
    for action in actions:
        TASK_ENV.take_action(action, action_type='qpos')

        # Check if task is done
        if TASK_ENV.eval_success:
            break

        # Get new observation (for potential history update)
        observation = TASK_ENV.get_obs()


def reset_model(model: Pi0InternPolicy):
    """Reset model state for new episode."""
    if model is not None:
        model.reset()


# For direct testing
if __name__ == "__main__":
    print("Pi0 Intern Policy for RoboTwin")
    print("=" * 50)
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"RoboTwin root: {ROBOTWIN_ROOT}")
    print(f"Repository root: {REPO_ROOT}")

    # Test imports
    try:
        from openpi.models.tokenizer import PaligemmaTokenizer
        print("Successfully imported PaligemmaTokenizer")
    except ImportError as e:
        print(f"Import error: {e}")
