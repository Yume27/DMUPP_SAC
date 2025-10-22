import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import ray
from avs_rl_tools.utils import get_available_cores

# import dreamer algorithm and argument parser
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3, DreamerV3Config
from ray.rllib.utils.test_utils import add_rllib_example_script_args

# from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import UnifiedLogger

from bsk_rl.utils.rllib.callbacks import WrappedEpisodeDataCallbacks
from bsk_rl.utils.rllib.discounting import (  # EpisodeDataCallbacks,
    CondenseMultiStepActions,
    ContinuePreviousAction,
    MakeAddedStepActionValid,
    TimeDiscountedGAEPPOTorchLearner,
)

# os.environ["RAY_DEDUP_LOGS"] = "0"


def train_model(
    model_name,
    output_directory,
    env_args={},
    n_envs=1,
    checkpoint_frequency=1,
    checkpoints_to_keep=2,
    reload_frequency=500_000,
    total_timesteps=1_000_000,
    training_args={},
    temp_dir="/tmp",
):
    os.environ["RAY_TMPDIR"] = os.environ["TMPDIR"] = temp_dir
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    ray.init(
        ignore_reinit_error=True,
        num_cpus=get_available_cores(),
        object_store_memory=2_000_000_000,  # 2 GB
        _temp_dir=temp_dir,
    )
    # dreamer v3 hyperparameters
    config = (
        DreamerV3Config()
        .environment(  # tell dreamer what environment it is working in
            env="SatelliteTasking-RLlib",
            env_config=env_args,
        )
        .env_runners(  # can create multiple runners to train in parallel
            num_env_runners=1,
            # If we use >1 GPU and increase the batch size accordingly, we should also
            # increase the number of envs per worker.
            num_envs_per_env_runner=n_envs + 1,
            remote_worker_envs=0,
        )
        .learners(  # only used if you have multiple env_runners
            num_learners=0,
            num_gpus_per_learner=0,
        )
        .reporting(  # enable these if you want to see the "dream" (algorithms representation)
            metrics_num_episodes_for_smoothing=1,
            report_images_and_videos=True,
            report_dream_data=True,
            report_individual_batch_item_stats=False,
        )
        # See Appendix A.
        .training(  # training size, bigger numbers mean more compute power
            **training_args
        )
        .checkpointing(export_native_model_files=True)
    )
    # create algorithm
    dreamer = DreamerV3(config)

    iter = 0
    step = 0
    # training loop
    while True:
        prev_step = step
        results = dreamer.train()
        step = results["num_env_steps_sampled_lifetime"]
        # save checkpoint to load from later
        checkpoint_path = (
            output_directory / model_name / f"checkpoint_{str(iter).zfill(6)}"
        )
        if iter % checkpoint_frequency == 0:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            dreamer.save_checkpoint(checkpoint_path)

        if step > total_timesteps:
            break

        # if step % reload_frequency < prev_step % reload_frequency:
        #     checkpoint_path.mkdir(parents=True, exist_ok=True)
        #     ppo.save_checkpoint(checkpoint_path)
        #     ray.shutdown()
        #     ray.init(
        #         ignore_reinit_error=True,
        #         num_cpus=get_available_cores(),
        #         object_store_memory=2_000_000_000,  # 2 GB
        #         _temp_dir=temp_dir,
        #     )
        #     ppo = PPO.from_checkpoint(checkpoint_path)

        if iter > checkpoints_to_keep * checkpoint_frequency - 1:
            for i in range(checkpoint_frequency):
                remove_dir = (
                    output_directory
                    / model_name
                    / f"checkpoint_{str(iter - checkpoints_to_keep * checkpoint_frequency - i).zfill(6)}"
                )
                try:
                    shutil.rmtree(remove_dir)
                except FileNotFoundError:
                    pass

        iter += 1


if __name__ == "__main__":
    import sys

    import yaml
    import nominal_sat
    from avs_rl_tools.utils import build_job_array, sanitize_np

    N = 0  # int(sys.argv[1])  # Passed by sweep.sh script
    model_name = f"model_{N}"
    n_envs = get_available_cores() - 2  # leave some extra cores for other processes
    output_dir = (
        "/scratch/alpine/yuna1623/DMUPP"   #CHANGE PATH HERE
    )
    output_dir = Path(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    env, satellites, env_args = nominal_sat.setup_env(test=False)

    jobs = build_job_array(
        training_args=dict(
            model_size=["mini"],
            training_ratio=[32],
            batch_size_B=[16],
        ),
        env_args={k:[v] for k,v in env_args.items()},
    )

    print(f"Running job {N}: {N + 1} of {len(jobs)}")
    job_args = jobs[N]

    with open(output_dir / f"{model_name}_params.txt", "w") as file:
        yaml.dump(sanitize_np(job_args), file)

    train_model(
        model_name=model_name,
        output_directory=output_dir,
        checkpoint_frequency=2,
        checkpoints_to_keep=2,
        total_timesteps=20_000_000,
        reload_frequency=300_000,
        n_envs=n_envs,
        # temp_dir="/scratch/alpine/mast9128/tmp",
        **job_args,
    )
