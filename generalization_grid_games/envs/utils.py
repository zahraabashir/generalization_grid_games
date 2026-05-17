import os

import numpy as np
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
from prpl_utils.structs import Image

def get_asset_path(asset_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    asset_dir_path = os.path.join(dir_path, 'assets')
    return os.path.join(asset_dir_path, asset_name)

def fig2data(fig: plt.Figure) -> Image:
    """Convert matplotlib figure into Image."""
    fig.canvas.draw()
    return np.array(fig.canvas.renderer.buffer_rgba())  # type: ignore


def run_random_agent_demo(env_cls, outdir=None, max_num_steps=10):
    if outdir is None:
        outdir = "/tmp/{}".format(env_cls.__name__)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    video_path = os.path.join(outdir, 'random_demo.mp4')
    env = env_cls(interactive=False, record_video=True, video_out_path=video_path)
    env.reset()
    
    for t in range(max_num_steps):
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        env.render()

        if done:
            break

    env.close()


def run_policy_demo(env_cls, policy, outdir=None, max_num_steps=100, video_name='policy_demo.mp4'):
    if outdir is None:
        outdir = "/tmp/{}".format(env_cls.__name__)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    video_path = os.path.join(outdir, video_name)
    env = env_cls(interactive=False, record_video=True, video_out_path=video_path)
    observation = env.reset()

    for _ in range(max_num_steps):
        action = policy(observation)
        observation, _, done, _ = env.step(action)
        if done:
            break

    env.close()
    return video_path


def run_plan_demo(env_cls, planner, outdir=None, video_name='expert_demo.mp4'):
    if outdir is None:
        outdir = "/tmp/{}".format(env_cls.__name__)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    video_path = os.path.join(outdir, video_name)
    env = env_cls(interactive=False, record_video=True, video_out_path=video_path)
    observation = env.reset()
    done = env.compute_done(observation)

    for action in planner(observation):
        observation, _, done, _ = env.step(action)
        if done:
            break

    env.close()
    if not done:
        raise RuntimeError("Plan did not solve {}.".format(env_cls.__name__))

    return video_path
