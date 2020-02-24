## @file

import multiprocessing
import numpy as np
from collections import OrderedDict
from multiprocessing import Process, Lock

import gym

from stable_baselines.common.vec_env import SubprocVecEnv, VecEnv, CloudpickleWrapper
from stable_baselines.common.tile_images import tile_images


## Unique threading.Lock for every simulation.
# It is required for parallel rendering since corbaserver does not support multiple connection simultaneously.
lock = Lock()

def _worker(remote, parent_remote, env_fn_wrapper, lock=None):
    """
    @brief      Worker for each subprocess of SubprocVecEnvLock.

    @details    It enables the use of a threading.Lock. It is the only
                difference with the implementation provided with the
                class `SubprocVecEnv` of Gym OpenAI.

    @remark     `remote` and `parent_remote` are returned by the method `Pipe` of
                `multiprocessing`.  It is a pair of connection objects connected
                by a pipe which by default is duplex (two-way). See
                `multiprocessing` documentation for more information.

                This is a hidden function that is not automatically imported
                using 'from gym_jiminy import *'.

    @param[in]  remote              Child remote
    @param[in]  parent_remote       Parent remote
    @param[in]  env_fn_wrapper      Gym environment
    @param[in]  lock                threading.Lock object. Optional: None by default
    """
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    # save final observation where user can get it, then reset
                    info['terminal_observation'] = observation
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render(*data[0], lock=lock, **data[1]))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError
        except EOFError:
            break


class SubprocVecEnvLock(SubprocVecEnv):
    """
    @brief      Creates a multiprocess vectorized wrapper for multiple environments,
                distributing each environment to its own process, allowing
                significant speed up when the environment is computationally complex.

    @details    It features a unique threading.Lock and uses it to enable parallel
                rendering in Gepetto-viewer. It is the only difference with the
                based class `SubprocVecEnv` provided by Gym OpenAI.

    @warning    For performance reasons, if your environment is not IO bound, the
                number of environments should not exceed the number of logical cores
                on your CPU.
    """
    def __init__(self, env_fns, start_method=None):
        """
        @brief      Constructor

        @warning    Only 'forkserver' and 'spawn' start methods are thread-safe, which is
                    important when TensorFlow sessions or other non thread-safe libraries
                    are used in the parent.
                    However, compared to 'fork' they incur a small start-up cost and have
                    restrictions on global variables. With those methods, users must wrap
                    the code in an ``if __name__ == "__main__":``
                    For more information, see the multiprocessing documentation.

        @param[in]  env_fns             List of Gym Environments to run in subprocesses
        @param[in]  start_method        Method used to start the subprocesses. Must be one of the
                                        methods returned by multiprocessing.get_all_start_methods().
                                        Optional: Defaults to 'fork' on available platforms, and 'spawn' otherwise.

        @return     Instance of SubprocVecEnvLock.
        """
        global lock

        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            fork_available = 'fork' in multiprocessing.get_all_start_methods()
            start_method = 'fork' if fork_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn), lock)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)


    def render(self, mode='rgb_array', *args, **kwargs):
        """
        @brief      Parallel rendering of the current state of each environment.

        @param[in]  mode    Display mode. For now, only 'rgb_array' is supported.
        @param[in]  args    Extra arguments passed to `render` method of each Gym environment
        @param[in]  kwargs  Extra keyword arguments passed to `render` method of each Gym environment

        @return     None or iterator of RGB images if mode == 'rgb_array'.
        """

        # gather images from subprocesses. `mode` will be taken into account later
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(('render', (args, {'mode': 'rgb_array', **kwargs})))
        imgs = [pipe.recv() for pipe in self.remotes]

        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs)

        if mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError
