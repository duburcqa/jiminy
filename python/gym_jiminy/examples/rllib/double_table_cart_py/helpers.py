
import os
import socket
import pathlib
import logging
from datetime import datetime
from typing import Optional, Callable

from tensorboard.program import TensorBoard

import ray
from ray.tune.logger import UnifiedLogger


def initialize(num_cpus: int = 0,
               num_gpus: int = 0,
               log_root_path: Optional[str] = None,
               log_name: Optional[str] = None,
               debug: bool = False,
               verbose: bool = True) -> Callable[[], UnifiedLogger]:
    """Initialize Ray and Tensorboard daemons.

    It will be used later for almost everything from dashboard, remote/client
    management, to multithreaded environment.

    :param log_root_path: Fullpath of root log directory.
                          Optional: location of this file / log by default.
    :param log_name: Name of the subdirectory where to save data.
                     Optional: full date _ hostname by default.
    :param verbose: Whether or not to print information about what is going on.
                    Optional: True by default.

    :returns: lambda function to pass Ray Trainers to monitor the learning
              progress in tensorboard.
    """
    # Initialize Ray server, if not already running
    if not ray.is_initialized():
        ray.init(
            # Address of Ray cluster to connect to, if any
            address=None,
            # Number of CPUs assigned to each raylet (None to disable limit)
            num_cpus=num_cpus,
            # Number of GPUs assigned to each raylet (None to disable limit)
            num_gpus=num_gpus,
            # Enable object eviction in LRU order under memory pressure
            _lru_evict=False,
            # Whether or not to execute the code serially (for debugging)
            local_mode=False,
            # Logging level
            logging_level=logging.DEBUG if debug else logging.ERROR,
            # Whether to redirect the output from every worker to the driver
            log_to_driver=debug,
            # Whether to start Ray dashboard, which displays cluster's status
            include_dashboard=True,
            # The host to bind the dashboard server to
            dashboard_host="0.0.0.0"
        )

    # Configure Tensorboard
    if log_root_path is None:
        log_root_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "log")
    if 'tb' not in locals().keys():
        tb = TensorBoard()
        tb.configure(host="0.0.0.0", logdir=log_root_path)
        url = tb.launch()
        if verbose:
            print(f"Started Tensorboard {url}. "
                  f"Root directory: {log_root_path}")

    # Create log directory
    if log_name is None:
        log_name = "_".join((datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                            socket.gethostname().replace('-', '_')))
    log_path = os.path.join(log_root_path, log_name)
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Tensorboard logfiles directory: {log_path}")

    # Define Ray logger
    def logger_creator(config):
        return UnifiedLogger(config, log_path, loggers=None)

    return logger_creator
