import numpy as np
import torch

from pathlib import Path
from pyarrow.fs import LocalFileSystem
from ray.rllib.core.rl_module import RLModule
from time import sleep
from env import PhysicalQuoridorEnv_, convert_to_box, convert_to_discrete


def get_action(env, observation, rl_module):
    return np.clip(
        rl_module.get_inference_action_dist_cls().from_logits(
            rl_module.forward_inference({"obs": torch.from_numpy(observation).unsqueeze(0)})["action_dist_inputs"]
        ).to_deterministic().sample()[0].detach().numpy(),
        a_min=env.action_space("player-0").low[0],
        a_max=env.action_space("player-0").high[0],
    )


checkpoint_paths = list(sorted(
    filter(
        lambda path: path.is_dir(),
        (Path(".") / "checkpoints").glob("*")
    ),
    reverse=True
))
rl_modules = {
    "player-0": RLModule.from_checkpoint(checkpoint_paths[0] / "learner_group" / "learner" / "rl_module" / "policy_0", filesystem=LocalFileSystem()),
    "player-1": RLModule.from_checkpoint(checkpoint_paths[len(checkpoint_paths) // 2] / "learner_group" / "learner" / "rl_module" / "policy_0", filesystem=LocalFileSystem())
}

env = PhysicalQuoridorEnv_(render_mode="human")
observations, infos = env.reset()

while True:
    actions = dict(zip(
        observations.keys(),
        map(
            lambda key: get_action(env, observations[key], rl_modules[key]),
            observations.keys()
        )
    ))

    if convert_to_discrete(actions["player-0"][0], 2) == 0:
        print(f"0: [{convert_to_box(actions["player-0"][1], -1, 1): .3f}, {convert_to_box(actions["player-0"][2], -1, 1): .3f}]")
    else:
        print(f"1: ({convert_to_discrete(actions["player-0"][3], 8)}, {convert_to_discrete(actions["player-0"][4], 8)}, {convert_to_discrete(actions["player-0"][5], 2)})")

    observations, rewards, terminations, truncations, infos = env.step(actions)

    if any(terminations.values()):
        print(rewards)
        sleep(5)
        break

    sleep(0.1)
