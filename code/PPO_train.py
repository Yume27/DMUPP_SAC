import glob
import os
import shutil
from pathlib import Path

import torch
torch.set_num_threads(11)
os.environ["MKL_NUM_THREADS"] = "11"
import ray
from avs_rl_tools.utils import get_available_cores
from bsk_rl.utils.rllib.callbacks import WrappedEpisodeDataCallbacks
from bsk_rl.utils.rllib.discounting import \
    TimeDiscountedGAEPPOTorchLearner  # EpisodeDataCallbacks,
# from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.logger import UnifiedLogger
from configurations import T_ORBIT

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
    rl_module_args={},
    temp_dir="/tmp",
    n_steps_switch=None,
    continue_previous=False,
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
    config = (
        PPOConfig()
        .training(**training_args)
        .env_runners(
            num_env_runners=n_envs + 1,
            sample_timeout_s=1000.0,
            # module_to_env_connector=lambda env: (ContinuePreviousAction(),),
        )
        .environment(
            env="SatelliteTasking-RLlib",
            env_config=env_args,
        )
        .callbacks(WrappedEpisodeDataCallbacks)
        .reporting(
            metrics_num_episodes_for_smoothing=1,
            metrics_episode_collection_timeout_s=180,
        )
        .checkpointing(export_native_model_files=True)
        .framework(framework="torch")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .resources(num_gpus=0)
    )
    config = config.training(
        **training_args,
        # learner_connector=lambda obs_space, act_space: (
        #     MakeAddedStepActionValid(expected_train_batch_size=config.train_batch_size),
        #     CondenseMultiStepActions(),
        # ),
        learner_class=TimeDiscountedGAEPPOTorchLearner,
    )
    config.rl_module(**rl_module_args)
    config.logger_config = dict(
        type=UnifiedLogger, logdir=output_directory / model_name
    )

    start_from_zero = False
    if continue_previous:
        checkpoint_path_dir = (
            output_directory / model_name
        )
        checkpoints = glob.glob(str(checkpoint_path_dir) + "/checkpoint_*")
        if len(checkpoints) == 0:
            print("No model to re-load and continue training")
            start_from_zero = True
        
        if start_from_zero is False:
            checkpoint_number_str = ""
            checkpoint_number_int = 0
            #Figure out the most recent checkpoint
            for checkpoint_number in checkpoints:
                checkpoint_folder_i = checkpoint_number.split("/")[-1]
                checkpoint_number_i = int(checkpoint_folder_i.split("_")[-1])
                if checkpoint_number_i > checkpoint_number_int:
                    checkpoint_number_int = checkpoint_number_i
                    checkpoint_number_str = checkpoint_folder_i.split("_")[-1]

            latest_checkpoint_dir = checkpoint_path_dir / f"checkpoint_{checkpoint_number_str}"
            if not os.path.exists(output_dir):
                print(f"Selected path does not exist: {latest_checkpoint_dir}")
                exit()
            
            ppo = PPO(config)
            print(str(latest_checkpoint_dir))
            ppo.restore(str(latest_checkpoint_dir))
            iter = checkpoint_number_int + 1
            results = ppo.train()
            step = results["num_env_steps_trained_lifetime"]
            if n_steps_switch is None:
                cond_switch = True
                checkpoint_saved = True
            elif step >= n_steps_switch:
                cond_switch = True
                checkpoint_saved = True
            else:
                cond_switch = False
                checkpoint_saved = False
    
    if not continue_previous or start_from_zero:
        ppo = PPO(config)
        iter = 0
        step = 0
        if n_steps_switch is not None:
            cond_switch = False
            checkpoint_saved = False
        else:
            cond_switch = True
            checkpoint_saved = True

    while True:
        prev_step = step
        
        results = ppo.train()
        step = results["num_env_steps_trained_lifetime"]

        checkpoint_path = (
            output_directory / model_name / f"checkpoint_{str(iter).zfill(6)}"
        )
        if iter % checkpoint_frequency == 0:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            ppo.save_checkpoint(checkpoint_path)

        if step > total_timesteps:
            break

        if cond_switch is False and (step + reload_frequency) > n_steps_switch:
            cond_switch = True

        if step % reload_frequency < prev_step % reload_frequency:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            ppo.save_checkpoint(checkpoint_path)
            if cond_switch and checkpoint_saved is False:
                checkpoint_saved = True
                shutil.copytree(output_directory / model_name, output_directory / f"{model_name}_switch")
            ray.shutdown()
            ray.init(
                ignore_reinit_error=True,
                num_cpus=get_available_cores(),
                object_store_memory=2_000_000_000,  # 2 GB
                _temp_dir=temp_dir,
            )
            ppo = PPO.from_checkpoint(checkpoint_path)

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

def episode_data_callback(env):
    reward = env.rewarder.cum_reward
    reward = sum(reward.values()) / len(reward)
    orbits = env.simulator.sim_time / (T_ORBIT)

    data = dict(
        reward=reward,
        alive=float(env.satellite.is_alive()),
        rw_status_valid=float(env.satellite.dynamics.rw_speeds_valid()),
        battery_status_valid=float(env.satellite.dynamics.battery_valid()),
        orbits_complete=orbits,
        imaged = env.satellite.imaged,
        missed = env.satellite.missed,
        image_action = env.satellite.image_count,
    )
    if orbits > 0:
        data["reward_per_orbit"] = reward / orbits
    if not env.satellite.is_alive():
        data["orbits_complete_partial_only"] = orbits
        
    return data

if __name__ == "__main__":
    import sys

    import yaml
    import moving_targets
    from avs_rl_tools.utils import build_job_array, sanitize_np


    N = int(sys.argv[1])  # Passed by sweep.sh script
    moving = bool(int(sys.argv[2]))
    model_name = f"model_moving_{moving}"
    n_envs = get_available_cores() - 2  # leave some extra cores for other processes
    output_dir = Path(
        "/scratch/alpine/yuna1623/moving_target"   #CHANGE PATH HERE
        #"./models_test"
        # Should be in /scratch/alpine/[user] when running on the cluster
    )

    os.makedirs(output_dir, exist_ok=True)

    env, satellites, env_args = moving_targets.setup_env(test=False, use_moving_targets=moving)
    env_args["episode_data_callback"] = episode_data_callback

    #We need to check which parameters we are going to use for training (gamma etc)
    jobs = build_job_array(
        training_args=dict(
            lr=[0.00003],
            gamma=[0.997],
            train_batch_size=[int(100 * n_envs)],
            num_sgd_iter=[10],
            lambda_=[0.95],
            use_kl_loss=[False],
            clip_param=[0.2],
            grad_clip=[0.5],
            entropy_coeff=[0.0],
        ),
        env_args={k:[v] for k,v in env_args.items()},
    )

    print(f"Running job {N}: {N+1} of {len(jobs)}")
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
        rl_module_args=dict(
            model_config_dict={
                "use_lstm": False,
                "fcnet_hiddens": [2048] * 2,
                "vf_share_layers": False,
            },
        ),
        temp_dir="/scratch/alpine/yuna1623/tmp",
        continue_previous=True,
        **job_args,
    )
