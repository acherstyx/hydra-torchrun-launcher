# Hydra Torchrun Launcher

This plugin aims to make the launching of torch distributed training configurable in Hydra.

The configuration is as follows:

```yaml
hydra:
  launcher:
    _target_: hydra_plugins.hydra_torchrun_launcher.distributed_launcher.TorchDistributedLauncher
    nnodes: '1:1'
    nproc_per_node: '1'
    rdzv_backend: static
    rdzv_endpoint: ''
    rdzv_id: none
    rdzv_conf: ''
    standalone: false
    max_restarts: 0
    monitor_interval: 5
    start_method: spawn # Support start_method=spawn, required by CUDA
    role: default
    module: false
    no_python: false
    run_path: false
    log_dir: null
    redirects: '0'
    tee: '0'
    node_rank: 0
    master_addr: '127.0.0.1'
    master_port: 29500
    local_addr: null
    training_script: ''
    training_script_args: [ ]
```

The meaning of each parameter matches exactly with the arguments
of [torchrun](https://pytorch.org/docs/stable/elastic/run.html).
Please refer to its documentation for a more detailed introduction.

### Installation

```shell
pip3 install git+https://github.com/acherstyx/hydra-torchrun-launcher.git
```

### Usage

```shell
python3 run_net.py --multirun hydra/launcher=torchrun hydra.launcher.nproc_per_node=8
```

The behavior of this example should be the same as launching with `torchrun`:

```shell
torchrun --nproc_per_node=8 run_net.py
```

### Acknowledgement

This plugin is modified from the hydra-torchrun-launcher plugin at [hydra/contrib](https://github.com/facebookresearch/hydra/tree/main/contrib/hydra_torchrun_launcher).
Currently, the main difference includes:

- Following `loky`, the pickling error described in [facebookresearch/hydra#2038](https://github.com/facebookresearch/hydra/issues/2038) is fixed through the use of `cloudpickle`. This version of the launcher now supports `start_method=spawn`, which is required by CUDA (see [pytorch/pytorch#40403](https://github.com/pytorch/pytorch/issues/40403)).
- The config is adjusted to match with `torchrun`.
- Fix `hydra.runtime.output_dir` missing after spawn.
- Fix the return value of multi-node training.
