# -*- coding: utf-8 -*-
# @Time    : 2024/3/28
# @Author  : Yaojie Shen
# @Project : hydra-torchrun-launcher
# @File    : config.py
from dataclasses import dataclass, field
from typing import Optional, List

from packaging.version import Version
from importlib.metadata import version

from hydra.core.config_store import ConfigStore


@dataclass
class TorchDistributedLauncherConf:
    _target_: str = "hydra_plugins.hydra_torchrun_launcher.distributed_launcher.TorchDistributedLauncher"

    # Worker/node size related arguments.
    nnodes: str = "1:1"
    nproc_per_node: str = "1"

    # Rendezvous related arguments
    rdzv_backend: str = "static"
    rdzv_endpoint: str = ""
    rdzv_id: str = "none"
    rdzv_conf: str = ""
    standalone: bool = False

    # User-code launch related arguments.
    max_restarts: int = 0
    monitor_interval: int = 5
    start_method: str = "spawn"
    role: str = "default"
    module: bool = False
    no_python: bool = False
    run_path: bool = False
    log_dir: Optional[str] = None
    redirects: str = "0"
    tee: str = "0"

    # Backwards compatible parameters with caffe2.distributed.launch.
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    local_addr: Optional[str] = None

    # Positional arguments.
    # Should not set
    training_script: str = ""
    training_script_args: List[str] = field(default_factory=list)


if Version(version("torch")) >= Version("2.3"):
    @dataclass
    class TorchDistributedLauncherConf(TorchDistributedLauncherConf):
        logs_specs: Optional[str] = None
        local_ranks_filter: str = ""

ConfigStore.instance().store(
    group="hydra/launcher", name="torchrun", node=TorchDistributedLauncherConf
)
