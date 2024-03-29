# -*- coding: utf-8 -*-
# @Time    : 2024/3/28
# @Author  : Yaojie Shen
# @Project : hydra-torchrun-launcher
# @File    : distributed_launcher.py
import logging
from typing import Optional, Sequence

from omegaconf import DictConfig

from hydra.types import HydraContext
from hydra.core.utils import JobReturn
from hydra.plugins.launcher import Launcher
from hydra.types import TaskFunction

logger = logging.getLogger(__name__)


class TorchDistributedLauncher(Launcher):
    def __init__(self) -> None:
        super().__init__()
        self.config: Optional[DictConfig] = None
        self.task_function: Optional[TaskFunction] = None
        self.hydra_context: Optional[HydraContext] = None

        self.launch_config = None

    def setup(
            self,
            *,
            hydra_context: HydraContext,
            task_function: TaskFunction,
            config: DictConfig,
    ) -> None:
        from . import _core

        return _core.setup(
            launcher=self,
            hydra_context=hydra_context,
            task_function=task_function,
            config=config,
        )

    def launch(
            self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        from . import _core

        return _core.launch(
            launcher=self, job_overrides=job_overrides, initial_job_idx=initial_job_idx
        )
