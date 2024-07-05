from absl import logging
import gin
import os
from agent.common.common import TensorboardWriter
import torch
from agent.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)

import time
from typing import Dict, List, Any
from agent.gibson_extension.examples.configs import Config


class BaseTrainer:

    def __init__(self, FLAGS) -> None:

        logging.set_verbosity(logging.INFO)
        gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_c)

        for k, v in FLAGS.flag_values_dict().items():
            print(k, v)

        self.FLAGS = FLAGS

    def train(self) -> None:
        return NotImplementedError
    
    def eval(self, env_load_fn) -> None:
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            if os.path.isfile(self.config.EVAL_CKPT_PATH_DIR):
                # evaluate singe checkpoint
                proposed_index = get_checkpoint_id(
                    self.config.EVAL_CKPT_PATH_DIR
                )
                if proposed_index is not None:
                    ckpt_idx = proposed_index
                else:
                    ckpt_idx = 0
                self._eval_checkpoint(
                    self.config.EVAL_CKPT_PATH_DIR,
                    writer,
                    checkpoint_index=ckpt_idx,
                )
            else:
                # evaluate multiple checkpoints in order
                prev_ckpt_ind = -1
                while True:
                    current_ckpt = None
                    while current_ckpt is None:
                        current_ckpt = poll_checkpoint_folder(
                            self.config.EVAL_CKPT_PATH_DIR, prev_ckpt_ind
                        )
                        time.sleep(2)  # sleep for 2 secs before polling again
                    logging.info(f"=======current_ckpt: {current_ckpt}=======")
                    prev_ckpt_ind += 1
                    self._eval_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        checkpoint_index=prev_ckpt_ind,
                        env_load_fn = env_load_fn,

                    )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
        env_load_fn: Any = None
    ) -> None:
        raise NotImplementedError
    

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError
        



class BaseRLTrainer(BaseTrainer):
    r"""Base trainer class for RL trainers. Future RL-specific
    methods should be hosted here.
    """
    device: torch.device  # type: ignore
    config: Config
    video_option: List[str]
    num_updates_done: int
    num_steps_done: int
    _flush_secs: int
    _last_checkpoint_percent: float

    def __init__(self, config: Config, FLAGS) -> None:
        super().__init__(FLAGS)
        assert config is not None, "needs config file to initialize trainer"
        self.config = config
        self._flush_secs = 30
        self.num_updates_done = 0
        self.num_steps_done = 0
        self._last_checkpoint_percent = -1.0

        if config.NUM_UPDATES != -1 and config.TOTAL_NUM_STEPS != -1:
            raise RuntimeError(
                "NUM_UPDATES and TOTAL_NUM_STEPS are both specified.  One must be -1.\n"
                " NUM_UPDATES: {} TOTAL_NUM_STEPS: {}".format(
                    config.NUM_UPDATES, config.TOTAL_NUM_STEPS
                )
            )

        if config.NUM_UPDATES == -1 and config.TOTAL_NUM_STEPS == -1:
            raise RuntimeError(
                "One of NUM_UPDATES and TOTAL_NUM_STEPS must be specified.\n"
                " NUM_UPDATES: {} TOTAL_NUM_STEPS: {}".format(
                    config.NUM_UPDATES, config.TOTAL_NUM_STEPS
                )
            )

        if config.NUM_CHECKPOINTS != -1 and config.CHECKPOINT_INTERVAL != -1:
            raise RuntimeError(
                "NUM_CHECKPOINTS and CHECKPOINT_INTERVAL are both specified."
                "  One must be -1.\n"
                " NUM_CHECKPOINTS: {} CHECKPOINT_INTERVAL: {}".format(
                    config.NUM_CHECKPOINTS, config.CHECKPOINT_INTERVAL
                )
            )

        if config.NUM_CHECKPOINTS == -1 and config.CHECKPOINT_INTERVAL == -1:
            raise RuntimeError(
                "One of NUM_CHECKPOINTS and CHECKPOINT_INTERVAL must be specified"
                " NUM_CHECKPOINTS: {} CHECKPOINT_INTERVAL: {}".format(
                    config.NUM_CHECKPOINTS, config.CHECKPOINT_INTERVAL
                )
            )

    def percent_done(self) -> float:
        if self.config.NUM_UPDATES != -1:
            return self.num_updates_done / self.config.NUM_UPDATES
        else:
            return self.num_steps_done / self.config.TOTAL_NUM_STEPS

    def is_done(self) -> bool:
        return self.percent_done() >= 1.0

    def should_checkpoint(self) -> bool:
        needs_checkpoint = False
        if self.config.NUM_CHECKPOINTS != -1:
            checkpoint_every = 1 / self.config.NUM_CHECKPOINTS
            if (
                self._last_checkpoint_percent + checkpoint_every
                < self.percent_done()
            ):
                needs_checkpoint = True
                self._last_checkpoint_percent = self.percent_done()
        else:
            needs_checkpoint = (
                self.num_steps_done % self.config.CHECKPOINT_INTERVAL
            ) == 0

        return needs_checkpoint

    @property
    def flush_secs(self):
        return self._flush_secs

    @flush_secs.setter
    def flush_secs(self, value: int):
        self._flush_secs = value

    def train(self) -> None:
        raise NotImplementedError

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
        env_load_fn: Any = None
    ) -> None:
        r"""Evaluates a single checkpoint. Trainer algorithms should
        implement this.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError









