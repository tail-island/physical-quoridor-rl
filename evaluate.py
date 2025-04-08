import numpy as np
import torch

from pathlib import Path
from pyarrow.fs import LocalFileSystem
from ray.rllib.core.rl_module import RLModule
from time import sleep
from env import PhysicalQuoridorEnv_


checkpoint_paths = list(sorted(
    filter(
        lambda path: path.is_dir(),
        (Path(".") / "checkpoints").glob("*")
    ),
    reverse=True
))

rl_modules = [
    RLModule.from_checkpoint(checkpoint_paths[0] / "learner_group" / "learner" / "rl_module" / "policy_0", filesystem=LocalFileSystem()),
    RLModule.from_checkpoint(checkpoint_paths[len(checkpoint_paths) // 2] / "learner_group" / "learner" / "rl_module" / "policy_0", filesystem=LocalFileSystem())
]

env = PhysicalQuoridorEnv_(render_mode="human")


def get_action(observation, rl_module):
    return np.clip(
        rl_module.get_inference_action_dist_cls().from_logits(
            rl_module.forward_inference({"obs": torch.from_numpy(observation).unsqueeze(0)})["action_dist_inputs"]
        ).to_deterministic().sample()[0].numpy(),
        a_min=env.action_space(0).low[0],
        a_max=env.action_space(0).high[0],
    )


observations, infos = env.reset()

while True:
    actions = dict(zip(
        observations.keys(),
        map(
            lambda key: get_action(observations[key], rl_modules[key]),
            observations.keys()
        )
    ))

    observations, reward, terminations, truncations, infos = env.step(actions)

    if any(terminations.values()):
        sleep(5)
        break

    sleep(0.1)
