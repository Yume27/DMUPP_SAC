import gymnasium as gym
import numpy as np

from Basilisk.architecture import bskLogging
from Basilisk.utilities import orbitalMotion
from bsk_rl import act, data, obs, scene, sats
from bsk_rl.scene.targets import Target
from bsk_rl.sim import dyn, fsw, world
from bsk_rl.utils.functional import aliveness_checker
from typing import Union

from bsk_rl.utils.orbital import random_orbit
import configurations as config
from configurations import T_ORBIT
# import moving_wrapper as mw
# import animation_wrapper as aw

bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

def s_hat_H(sat):
    r_SN_N = (
        sat.simulator.world.gravFactory.spiceObject.planetStateOutMsgs[
            sat.simulator.world.sun_index
        ]
        .read()
        .PositionVector
    )
    r_BN_N = sat.dynamics.r_BN_N
    r_SB_N = np.array(r_SN_N) - np.array(r_BN_N)
    r_SB_H = sat.dynamics.HN @ r_SB_N
    return r_SB_H / np.linalg.norm(r_SB_H)

class Density(obs.Observation):
    def __init__(
        self,
        interval_duration=60 * 3,
        intervals=10,
        norm=3,
    ):
        self.satellite: "sats.ImagingSatellite"
        super().__init__()
        self.interval_duration = interval_duration
        self.intervals = intervals
        self.norm = norm

    def get_obs(self):
        if self.intervals == 0:
            return []

        self.satellite.calculate_additional_windows(
            self.simulator.sim_time
            + (self.intervals + 1) * self.interval_duration
            - self.satellite.window_calculation_time
        )
        soonest = self.satellite.upcoming_opportunities_dict(types="target")
        rewards = np.array([opportunity.priority for opportunity in soonest])
        times = np.array([opportunities[0][1] for opportunities in soonest.values()])
        time_bins = np.floor((times - self.simulator.sim_time) / self.interval_duration)
        densities = [sum(rewards[time_bins == i]) for i in range(self.intervals)]
        return np.array(densities) / self.norm

# Turn off aliveness checks for battery and wheel speeds
class CustomDynModel(dyn.FullFeaturedDynModel):
    @aliveness_checker
    def rw_speeds_valid(self) -> bool:
        """Check if any wheel speed exceeds the ``maxWheelSpeed``."""
    
        return True
    
    @aliveness_checker
    def battery_valid(self) -> bool:
        """Check if the battery has charge remaining.

        Note that this check is instantaneous. If a satellite runs out of power during a
        environment step but then recharges to have positive power at the end of the step,
        the satellite will still be considered alive.
        """
        return True

class CustomSatComposed(sats.ImagingSatellite):
    observation_spec = [
        obs.SatProperties(
            dict(prop="omega_BN_B", norm=0.03),  # 0,1,2 - CHECKED
            dict(prop="c_hat_H"),  # 3,4,5 - CHECKED
            dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),  # 6,7,8 - CHECKED
            dict(prop="v_BN_P", norm=7616.5),  # 9,10,11 - CHECKED
            # dict(prop="battery_charge_fraction"),  # 12 - CHECKED
            # dict(prop="storage_level_fraction"),  # 13
            # dict(prop="wheel_speeds_fraction"),  # 14,15,16
            # dict(prop="s_hat_H", fn=s_hat_H),  # 17, 18, 19 - CHECKED
        ),
        # obs.OpportunityProperties(
        #     dict(prop="opportunity_open", norm=T_ORBIT),
        #     dict(prop="opportunity_close", norm=T_ORBIT),
        #     type="ground_station",
        #     n_ahead_observe=1,
        # ),  # 20, 21
        # obs.Eclipse(norm=T_ORBIT),  # 22,23 - CHECKED
        Density(intervals=20, norm=5),  # 24 - CHECKED
        obs.OpportunityProperties(
            dict(prop="priority"),  # - CHECKED
            dict(prop="r_LB_H", norm=orbitalMotion.REQ_EARTH * 1e3),  # - CHECKED
            dict(
                prop="target_angle", norm=np.pi / 2
            ),  # - CHECKED (LOOK IF THIS IS 1 OR 3)
            dict(
                prop="target_angle_rate", norm=0.03
            ),  # - CHECKED (LOOK IF THIS IS 1 OR 3)
            dict(prop="opportunity_open", norm=300.0),  # - CHECKED
            dict(prop="opportunity_close", norm=300.0),  # - CHECKED
            type="target",
            n_ahead_observe=32,
        ),
    ]

    action_spec = [
        # act.Charge(duration=60.0),  # 0
        # act.Downlink(),  # 1
        # act.Desat(duration=60.0),  # 2
        act.Image(n_ahead_image=32),  
    ]

    dyn_type = CustomDynModel
    fsw_type = fsw.SteeringImagerFSWModel

    def __init__(self, *args, image_count=0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.image_count = image_count

    def set_action(self, action: Union[int, Target, str]):
        super().set_action(action)
        self.image_count += 1

    def reset_post_sim_init(self):
        super().reset_post_sim_init()
        self.image_count = 0

def setup_env(
        test=True,
        horizon=None,
        n_targets=None, 
        target_distribution=None, 
    ):
    # This function sets up the environment and satellites for the simulation
    # It pulls in the configuration parameters from the configurations.py file
    # Return the gym environment and the satellites list

    if n_targets is None:
        n_targets = config.SIM_PARAMS["n_targets"]
    if target_distribution is None:
        target_distribution = config.SIM_PARAMS["target_distribution"]
    if horizon is None:
        horizon = config.SIM_PARAMS["horizon"] * T_ORBIT
    
    # communication = config.SIM_PARAMS["communication"] %We are always assuming full communication
    data_storage_capacity = config.STEERING_2Hz_POWER["sat_args"]["dataStorageCapacity"]

    if target_distribution == "uniform":
        env_features = scene.UniformTargets(n_targets=n_targets)
    elif target_distribution == "cities":
        env_features = scene.CityTargets(n_targets=n_targets)
    else:
        raise (ValueError("Invalid distribution type"))

    satellites = []
    sat_args = CustomSatComposed.default_sat_args(
        **config.STEERING_2Hz_POWER["sat_args"],
        storageInit=lambda: np.random.randint(
            config.RANDOM_INIT["storage_low_percent"] * data_storage_capacity,
            config.RANDOM_INIT["storage_high_percent"] * data_storage_capacity,
        ),
        wheelSpeeds=lambda: np.random.uniform(
            config.RANDOM_INIT["rw_speed_low"],
            config.RANDOM_INIT["rw_speed_high"],
            3,
        ),
    )
    satellites.append(
        CustomSatComposed(
            "EO",
            sat_args,
        )
    )

    #print("Opp dictionary", satellites[0].observation_space)

    if test:
        # Make the environment with Gymnasium
        env = gym.make(
            "GeneralSatelliteTasking-v1",
            satellites=satellites,
            terminate_on_time_limit=True,
            # Select an EnvironmentModel compatible with the models in the satellite
            world_type=world.GroundStationWorldModel,
            world_args=world.GroundStationWorldModel.default_world_args(),
            scenario=env_features,
            #scenario=scene.UniformTargets(1000),
            rewarder=data.UniqueImageReward(),
            sim_rate=config.STEERING_2Hz_POWER["sim_rate"],
            max_step_duration=config.SIM_PARAMS["max_step_duration"],
            time_limit=horizon,
            log_level="WARNING",
            failure_penalty=config.TRAINING_PARAMS["failure_penalty"],
            #vizard_dir="/tmp/vizard",
            #disable_env_checker=True,
        )
        env_args_dict = None
    else:
        env = None
        env_args_dict = dict(
            satellite = satellites[0],
            world_type = world.GroundStationWorldModel,
            world_args = world.GroundStationWorldModel.default_world_args(),
            scenario = env_features,
            rewarder = data.UniqueImageReward(),
            sim_rate = config.STEERING_2Hz_POWER["sim_rate"],
            max_step_duration = config.SIM_PARAMS["max_step_duration"],
            time_limit = horizon,
            log_level="WARNING",
            failure_penalty = config.TRAINING_PARAMS["failure_penalty"]
        )

    return env, satellites, env_args_dict


if __name__ == "__main__":

    #env, satellites, env_args_dict = setup_env(n_targets=100)
    env, satellites, env_args_dict = setup_env(n_targets=1000)
    # env_origin, satellites, env_args_dict = setup_env(n_targets=3000)
    # env_origin, satellites, env_args_dict = setup_env(n_targets=100)
    # env = aw.AnimationWrapper(env_origin, "../animations/")

    # Run the simulation until timeout or agent failure
    total_reward = 0.0
    observation, info = env.reset(seed=0)
    # The composed satellite action space returns a human-readable action map
    # Each satellite has the same action space and similar observation space
    print("Actions:", satellites[0].action_description)
    print("States:", env.satellites[0].observation_description)
    # print("TEST BATTERY: ", observation[0][12])
    # print("TEST ECLIPSE START: ", observation[0][16])
    # print("TEST ECLIPSE END: ", observation[0][17])
    
    obs_desc = env.satellites[0].observation_description
    obs_values = observation[0]  # assuming a single satellite

    # Extract indices for target-related features
    target_indices = [i for i, name in enumerate(obs_desc) if name.startswith("target.")]
    target_obs = [obs_values[i] for i in target_indices]

    # Optionally, pair with names for readability
    target_info = [(obs_desc[i], obs_values[i]) for i in target_indices]
    for name, val in target_info:
        print(f"{name}: {val}")

    # # Using the composed satellite features also provides a human-readable state:
    print("Observation:", env.satellites[0].observation_builder.obs_dict())
    for satellite in env.satellites:
        for k, v in satellite.observation_builder.obs_dict().items():
            print(f"\t\t{k}:  {v}")

    count = 0
    while True:
        print(f"<time: {env.simulator.sim_time:.1f}s>")

        if count == 0:
            # Vector with an action for each satellite (we can pass different actions for each satellite)
            # Tasking all satellites to charge (tasking None as the first action will raise a warning)
            action_vector = [0]
        elif count == 1:
            # None will continue the last action, but will also raise a warning
            action_vector = [None]
        elif count == 2:
            # Tasking different actions for each satellite
            action_vector = [1]
        else:
            # Tasking random actions
            action_vector = env.action_space.sample()
        count += 1

        observation, reward, terminated, truncated, info = env.step([5])

        # Show the custom normalized observation vector
        # print("\tObservation:", observation)
        obs_values = observation[0]  # assuming one satellite
        # print("\tTarget info:")
        # for i in target_indices:
        #     print(f"\t\t{obs_desc[i]}: {obs_values[i]}")

        total_reward += reward
        print(f"\tReward: {reward:.3f} ({total_reward:.3f} cumulative)")
        if terminated or truncated:
            print("Episode complete.")
            break

    print("Total reward:", total_reward)
    print("Total image actions taken:", env.satellites[0].image_count)
    # print("Total charge actions taken:", env.satellites[0].charge_count)
