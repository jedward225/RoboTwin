"""
Pi0 Policy Adapter for RoboTwin Evaluation (Delta Joints Format)

This adapter allows evaluating pi0 models (trained on InternData with delta joints)
in RoboTwin.

Data Format (matching InternVLA-M1):
- State: [joints(7), gripper_openness(1)] = 8 dims
- Action: [delta_joints(7), gripper_openness(1)] = 8 dims
- delta_joints = target_joints - current_joints

Key Pipeline:
1. Get current qpos from RoboTwin
2. Normalize state using training stats
3. Model outputs normalized delta actions
4. Denormalize to get real delta
5. Compute target = current + delta
6. Send target to RoboTwin

Usage:
    cd /home/jjliu/HistoryAwareVLA/third_party/RoboTwin
    bash policy/pi0_intern/eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id>
"""

import sys
import os
import json
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
from openpi.training.intern_data_loader import NormStats, load_norm_stats


class Pi0InternPolicy:
    """
    Pi0 policy wrapper for RoboTwin evaluation with delta joints.

    Handles:
    - Image preprocessing (resize)
    - State normalization (min-max to [-1, 1])
    - Delta action denormalization
    - Convert delta to absolute joint positions
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
        # [delta_joint_0, delta_joint_1, ..., delta_joint_6, gripper_openness]
        self.intern_action_dim = 8

        # RoboTwin action mapping: always 16 dim
        # [left_arm:7, left_gripper:1, right_arm:7, right_gripper:1]
        self.robotwin_action_dim = 16

        # Initialize tokenizer
        self.tokenizer = PaligemmaTokenizer(max_len=max_token_len)

        # Load normalization stats
        self.norm_stats = None
        self._load_norm_stats(checkpoint_path)

        # Load model
        self._load_model(checkpoint_path)

        # State buffers
        self.obs_history = []
        self.action_buffer = []
        self.current_instruction = None
        self.current_qpos = None  # Store current joint positions for delta computation

    def _load_norm_stats(self, checkpoint_path: str):
        """Load normalization statistics from checkpoint directory."""
        if os.path.isdir(checkpoint_path):
            norm_stats_path = os.path.join(checkpoint_path, "norm_stats.json")
        else:
            norm_stats_path = os.path.join(os.path.dirname(checkpoint_path), "norm_stats.json")

        if os.path.exists(norm_stats_path):
            self.norm_stats = load_norm_stats(norm_stats_path)
            print(f"Loaded norm_stats from: {norm_stats_path}")
            print(f"  State range: [{self.norm_stats['state'].min[:8]}] to [{self.norm_stats['state'].max[:8]}]")
            print(f"  Action range: [{self.norm_stats['actions'].min[:8]}] to [{self.norm_stats['actions'].max[:8]}]")
        else:
            print(f"WARNING: norm_stats.json not found at {norm_stats_path}")
            print("  Model outputs will NOT be denormalized!")

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
            model_path = os.path.join(checkpoint_path, "model.safetensors")
        else:
            model_path = checkpoint_path

        if os.path.exists(model_path):
            safetensors.torch.load_model(self.model, model_path, device=str(self.device))
            print(f"Loaded model from: {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model.eval()

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to [-1, 1] using training statistics."""
        if self.norm_stats is None or "state" not in self.norm_stats:
            return state

        # Only normalize first 8 dims (joints + gripper)
        state_8d = state[:8].copy()
        normalized = self.norm_stats["state"].normalize_minmax(state_8d)

        # Pad to action_dim
        result = np.zeros(self.action_dim, dtype=np.float32)
        result[:8] = normalized
        return result

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action from [-1, 1] to real values."""
        if self.norm_stats is None or "actions" not in self.norm_stats:
            return action[:8]

        # Only denormalize first 8 dims
        action_8d = action[:8].copy()
        return self.norm_stats["actions"].denormalize_minmax(action_8d)

    def _qpos_to_state(self, qpos: np.ndarray, gripper_val: float = None) -> np.ndarray:
        """
        Convert RoboTwin qpos to state format (8 dims, NOT normalized).

        State format: [joints(7), gripper_openness(1)]

        Args:
            qpos: Joint positions from RoboTwin (16 dim for dual-arm)
            gripper_val: Gripper value (0-1)
        """
        if self.embodiment == 'franka':
            joint_pos = qpos[:7]
            raw_gripper = qpos[7] if gripper_val is None else gripper_val
        else:
            # Dual-arm (aloha): qpos is 16 dim
            if self.arm_mode == 'left':
                joint_pos = qpos[:7]
                raw_gripper = qpos[7] if gripper_val is None else gripper_val
            elif self.arm_mode == 'right':
                joint_pos = qpos[8:15]
                raw_gripper = qpos[15] if gripper_val is None else gripper_val
            else:
                joint_pos = qpos[:7]
                raw_gripper = qpos[7] if gripper_val is None else gripper_val

        # State: [joints(7), gripper(1)]
        state = np.zeros(8, dtype=np.float32)
        state[:7] = joint_pos
        state[7] = raw_gripper  # Keep as 0-1, same as training

        return state

    def _delta_to_robotwin_action(self, delta_action: np.ndarray, current_joints: np.ndarray) -> np.ndarray:
        """
        Convert delta action to RoboTwin absolute action format.

        Args:
            delta_action: [delta_joints(7), gripper_openness(1)]
            current_joints: Current joint positions (7,)

        Returns:
            robotwin_action: (16,) absolute joint positions for RoboTwin
        """
        # Compute target joints
        delta_joints = delta_action[:7]
        gripper_action = delta_action[7]

        target_joints = current_joints + delta_joints

        # Safety clamp gripper to [0, 1]
        gripper_action = np.clip(gripper_action, 0.0, 1.0)

        # Output 16-dim actions for RoboTwin
        robotwin_action = np.zeros(16, dtype=np.float32)

        if self.embodiment == 'franka':
            # For Franka single-arm: left and right entity are the SAME robot!
            # Must set both sides to the same action, otherwise they conflict
            robotwin_action[:7] = target_joints
            robotwin_action[7] = gripper_action
            robotwin_action[8:15] = target_joints  # Same as left!
            robotwin_action[15] = gripper_action
        elif self.arm_mode == 'left':
            robotwin_action[:7] = target_joints
            robotwin_action[7] = gripper_action
            # Keep right arm at current position (don't set to 0!)
            # Actually for dual-arm, we need to handle this differently
        elif self.arm_mode == 'right':
            robotwin_action[8:15] = target_joints
            robotwin_action[15] = gripper_action
        else:
            robotwin_action[:7] = target_joints
            robotwin_action[7] = gripper_action
            robotwin_action[8:15] = target_joints
            robotwin_action[15] = gripper_action

        return robotwin_action

    def predict(self, obs: dict, instruction: str = None) -> np.ndarray:
        """
        Run inference and get action chunk.

        Args:
            obs: Encoded observation dict with 'state', 'head_cam', etc.
            instruction: Language instruction

        Returns:
            actions: (action_horizon, 16) action chunk for RoboTwin
        """
        if instruction is not None:
            self.current_instruction = instruction

        # Store current joint positions for delta computation
        current_joints = obs["current_joints"]  # (7,)

        # Prepare images
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

        # Prepare normalized state
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
            actions = self.model.sample_actions(
                device=self.device,
                observation=observation,
                num_steps=10,
            )

        # Convert to numpy
        actions_np = actions.cpu().numpy()[0]  # (action_horizon, 32)

        # DEBUG: Print raw model output
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        if self._debug_count < 3:
            print(f"\n=== Model Output Debug (step {self._debug_count}) ===")
            print(f"Raw model output (first 8): {actions_np[0, :8]}")
            print(f"Raw model output range: [{actions_np[0, :8].min():.4f}, {actions_np[0, :8].max():.4f}]")

        # Process each action: denormalize and convert delta to absolute
        robotwin_actions = []
        running_joints = current_joints.copy()

        for i in range(self.action_horizon):
            # Denormalize action
            delta_action = self._denormalize_action(actions_np[i])

            # DEBUG: Print denormalized delta
            if self._debug_count < 3 and i == 0:
                print(f"Denormalized delta (first 8): {delta_action}")
                print(f"Current joints: {running_joints}")
                print(f"Target = current + delta: {running_joints + delta_action[:7]}")
                self._debug_count += 1

            # Convert delta to absolute
            robotwin_action = self._delta_to_robotwin_action(delta_action, running_joints)
            robotwin_actions.append(robotwin_action)

            # Update running joints for next action in chunk
            if self.arm_mode == 'left':
                running_joints = robotwin_action[:7]
            elif self.arm_mode == 'right':
                running_joints = robotwin_action[8:15]
            else:
                running_joints = robotwin_action[:7]

        return np.array(robotwin_actions)

    def reset(self):
        """Reset policy state for new episode."""
        self.obs_history = []
        self.action_buffer = []
        self.current_instruction = None
        self.current_qpos = None


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
    global _model_instance

    # Extract images
    head_rgb = observation["observation"]["head_camera"]["rgb"]
    left_rgb = observation["observation"]["left_camera"]["rgb"]
    right_rgb = observation["observation"]["right_camera"]["rgb"]

    # Resize to 224x224
    head_rgb = cv2.resize(head_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    left_rgb = cv2.resize(left_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    right_rgb = cv2.resize(right_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Extract qpos
    qpos = np.array(observation["joint_action"]["vector"], dtype=np.float32)

    # Get gripper value
    gripper_val = None
    if "endpose" in observation and observation["endpose"]:
        if _model_instance is not None and _model_instance.arm_mode == 'right':
            gripper_val = observation["endpose"].get("right_gripper", None)
        else:
            gripper_val = observation["endpose"].get("left_gripper", None)

    # Convert qpos to state format (8 dims, raw)
    if _model_instance is not None:
        state_raw = _model_instance._qpos_to_state(qpos, gripper_val=gripper_val)
        # Get current joints for delta computation
        if _model_instance.arm_mode == 'left':
            current_joints = qpos[:7].copy()
        elif _model_instance.arm_mode == 'right':
            current_joints = qpos[8:15].copy()
        else:
            current_joints = qpos[:7].copy()

        # Normalize state
        state = _model_instance._normalize_state(state_raw)
    else:
        state_raw = np.zeros(8, dtype=np.float32)
        state_raw[:7] = qpos[:7]
        state_raw[7] = qpos[7]
        current_joints = qpos[:7].copy()
        state = np.zeros(32, dtype=np.float32)
        state[:8] = state_raw

    return {
        "head_cam": head_rgb,
        "left_cam": left_rgb,
        "right_cam": right_rgb,
        "state": state,              # (32,) normalized state
        "current_joints": current_joints,  # (7,) raw joints for delta computation
        "qpos": qpos,                # (16,) original RoboTwin qpos
    }


def get_model(usr_args: dict):
    """Initialize pi0 model for RoboTwin evaluation."""
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
        ckpt_setting,
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

    # Auto-detect embodiment from config
    left_arm_dim = usr_args.get('left_arm_dim', 7)
    right_arm_dim = usr_args.get('right_arm_dim', 7)

    # Check embodiment config - can be a list like ['franka-panda'] or a string
    embodiment_config = usr_args.get('embodiment', None)
    embodiment_name = usr_args.get('embodiment_name', '')

    # Extract embodiment name from config (handles list format from yml)
    if embodiment_config is not None:
        if isinstance(embodiment_config, list):
            # Config format: embodiment: [franka-panda] or [aloha-left, aloha-right, 0.5]
            embodiment_name = str(embodiment_config[0]) if embodiment_config else ''
        elif isinstance(embodiment_config, str):
            # Direct string: embodiment: franka
            if embodiment_config in ['franka', 'dual_arm']:
                # Already the target format
                embodiment = embodiment_config
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
            else:
                embodiment_name = embodiment_config

    # Determine embodiment based on name
    if 'franka' in embodiment_name.lower():
        embodiment = 'franka'
    elif right_arm_dim > 0 and left_arm_dim > 0:
        embodiment = 'dual_arm'
    else:
        embodiment = 'franka'

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


_debug_step_count = 0
_debug_save_dir = None
_debug_detailed = os.environ.get('VLA_DEBUG', '0') == '1'  # Set VLA_DEBUG=1 for detailed debugging

def eval(TASK_ENV, model: Pi0InternPolicy, observation: dict):
    """Execute one evaluation step."""
    global _debug_step_count, _debug_save_dir, _debug_detailed

    # Create debug save directory on first step
    if _debug_step_count == 0 and _debug_detailed:
        import os
        from datetime import datetime
        _debug_save_dir = f"./debug_output/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(_debug_save_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"DEBUG MODE: Saving detailed step info to {_debug_save_dir}")
        print(f"{'='*60}\n")

    # Encode observation
    obs = encode_obs(observation)

    # Get language instruction
    instruction = TASK_ENV.get_instruction()

    # Debug: print info for ALL steps when detailed mode is on
    if _debug_detailed or _debug_step_count < 3:
        print(f"\n{'='*60}")
        print(f"=== STEP {_debug_step_count} ===")
        print(f"{'='*60}")
        print(f"Instruction: {instruction}")
        print(f"\n[INPUT STATE]")
        print(f"  State (normalized, first 8): {obs['state'][:8]}")
        print(f"  Current joints (raw): {obs['current_joints']}")
        print(f"  Full qpos from RoboTwin: {obs['qpos'][:8]}")

        if model.norm_stats is not None:
            # Show denormalized state for clarity
            state_min = model.norm_stats['state'].min[:8]
            state_max = model.norm_stats['state'].max[:8]
            state_denorm = (obs['state'][:8] + 1) / 2 * (state_max - state_min) + state_min
            print(f"  State (denormalized): {state_denorm}")

    # Save images if detailed debug
    if _debug_detailed and _debug_save_dir:
        cv2.imwrite(f"{_debug_save_dir}/step{_debug_step_count:04d}_head.png",
                    cv2.cvtColor(obs['head_cam'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{_debug_save_dir}/step{_debug_step_count:04d}_left.png",
                    cv2.cvtColor(obs['left_cam'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{_debug_save_dir}/step{_debug_step_count:04d}_right.png",
                    cv2.cvtColor(obs['right_cam'], cv2.COLOR_RGB2BGR))

    # Get action chunk from model
    actions = model.predict(obs, instruction=instruction)

    # Debug: show predicted actions
    if _debug_detailed or _debug_step_count < 3:
        print(f"\n[MODEL OUTPUT]")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Action chunk (first 3 steps):")
        for i in range(min(3, len(actions))):
            print(f"    [{i}] joints: {actions[i, :7]}, gripper: {actions[i, 7]:.4f}")

    _debug_step_count += 1

    # Execute action chunk with detailed logging
    action_idx = 0
    for action in actions:
        if _debug_detailed:
            # Get state BEFORE action
            pre_qpos = np.array(TASK_ENV.get_obs()["joint_action"]["vector"], dtype=np.float32)
            print(f"\n  [EXEC {action_idx}] Pre-action qpos[:8]:  {pre_qpos[:8]}")
            print(f"  [EXEC {action_idx}] Sending action[:8]:    {action[:8]}")

        TASK_ENV.take_action(action, action_type='qpos')

        if _debug_detailed:
            # Get state AFTER action
            post_obs = TASK_ENV.get_obs()
            post_qpos = np.array(post_obs["joint_action"]["vector"], dtype=np.float32)
            delta = post_qpos[:8] - pre_qpos[:8]
            print(f"  [EXEC {action_idx}] Post-action qpos[:8]: {post_qpos[:8]}")
            print(f"  [EXEC {action_idx}] Actual delta[:8]:     {delta}")

        if TASK_ENV.eval_success:
            print(f"\n*** SUCCESS at step {_debug_step_count}, action {action_idx} ***")
            break

        observation = TASK_ENV.get_obs()
        action_idx += 1


def reset_model(model: Pi0InternPolicy):
    """Reset model state for new episode."""
    global _debug_step_count
    _debug_step_count = 0
    if model is not None:
        model.reset()


# For direct testing
if __name__ == "__main__":
    print("Pi0 Intern Policy for RoboTwin (Delta Joints Format)")
    print("=" * 50)
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"RoboTwin root: {ROBOTWIN_ROOT}")
    print(f"Repository root: {REPO_ROOT}")

    try:
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.training.intern_data_loader import NormStats, load_norm_stats
        print("Successfully imported required modules")
    except ImportError as e:
        print(f"Import error: {e}")
