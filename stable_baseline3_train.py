import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
import os
from stable_baselines3.common.env_checker import check_env
import time
from env.mec_env import MECEnv
import json

def ensure_list(value):
    if isinstance(value, list):
        return value  # If it's already a list, return it unchanged
    if isinstance(value, np.ndarray):
        if value.shape == (1,):  # Convert 1D array with one element to scalar
            value = value.item()  # Extract the scalar value
        return [np.array(value)]  # Wrap in a list and ensure it's a NumPy scalar
    return [np.array(value)]

def train():
    conf_file = 'env/config.json'
    config = json.load(open(conf_file, 'r'))
    
    config = config['config1']
    

    env = MECEnv(params=config)
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=int(15000), progress_bar=True)
    model.save("A2C_MEC")
    del model


def calc_delay_hist(obs):
    total_delay = np.zeros()
    for i in range(obs.shape[1]):
        pass

def to_json(info, json_filename="info.json"):
    
    if os.path.exists(json_filename):
        with open(json_filename, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append the new info
    item = {
        "action": str(info[0]["action"]),
        "reward": info[0]["reward"],
        "energy_(mWs)": info[0]["energy_mWs"],
        "time delay_(ms)": info[0]["time_delay_ms"],
        "cloud_exe_task": info[0]["cloud_exe_task"],
        "task_to_server": info[0]["task_to_server"]
    }

    for i in range(info[0]["edge_num"]):
        item[f"edge_{i}_exe_task"] = info[0][f"edge_{i}_exe_task"]

    data.append(item)

    # Write back to file
    with open(json_filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

def eval():
    conf_file = 'env/config.json'
    config = json.load(open(conf_file, 'r'))
    
    config = config['config1']
    env = MECEnv(params=config)
    model = A2C.load("PPO_MEC", env=env)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(mean_reward, std_reward)

def test():
    conf_file = 'env/config.json'
    config = json.load(open(conf_file, 'r'))
    
    config = config['config1']
    env = MECEnv(params=config)
    
    
    #check_env(env)
    model = A2C.load("A2C_MEC", env=env)

    env = model.get_env()
    obs = env.reset()[0]
    # print(obs)
    
    while True:
        
        # for action in range(6):
        #     obs, reward, done, info = env.step(actions=[np.array(action)])
            
        #     time.sleep(0.2)
        
        action, _states = model.predict(obs, deterministic=True)
        # print(action)
        # print(np.array(action.tolist()))
        # print(action)
        action = ensure_list(action)
        # print(action)
        # action = np.array([action])
        # print(action.shape)
        obs, rewards, dones, info = env.step(action)

        time.sleep(1)

        # to_json(info)
        
if __name__=="__main__":
    test()