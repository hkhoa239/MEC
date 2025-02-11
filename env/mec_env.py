from copy import deepcopy
import numpy as np
import json
import gymnasium as gym
import csv
import os

ACTION_TO_CLOUD = 0

RAYLEIGH_VAR = 1
RAYLEIGH_PATH_LOSS_A = 35
RAYLEIGH_PATH_LOSS_B = 133.6
RAYLEIGH_ANTENNA_GAIN = 0
RAYLEIGH_SHADOW_FADING = 8
RAYLEIGH_NOISE_dBm = -174

ZERO_RES = 1e-6


class MECEnv(gym.Env):
    def __init__(self, params={}):
        super().__init__()
        self.dt = params['dt']
        self.Tmax = params['Tmax']
        self.done = False
        self.possion_lamda = params['possion_lamda']
        
        self.wave_peak = params['wave_peak']
        self.wave_cycle = params['wave_cycle']
        self.user_num = params['user_num']
        self.task_size_L = params['task_size_L']
        self.task_size_H = params['task_size_H']
        self.unassigned_tasks = []
        self.task_to_server = []
        self.edge_num = params['edge_num']
        self.user_num = params['user_num']
        self.cloud_off_power = params['cloud_off_power']
        self.edge_off_power = params['edge_off_power']

        self.cloud_C = params['cloud_C']
        self.edge_C = params['edge_C']
        self.cloud_k = params['cloud_k']
        self.edge_k = params['edge_k']

        self.cloud_user_dist_H = params['cloud_user_dist_H']
        self.cloud_user_dist_L = params['cloud_user_dist_L']
        self.edge_user_dist_H = params['edge_user_dist_H']
        self.edge_user_dist_L = params['edge_user_dist_L']
        
        self.cloud_off_lists = []
        self.cloud_exe_lists = []
        self.edge_off_lists = []
        self.edge_exe_lists = []

        self.cloud_freq = params['cloud_cpu_freq']
        self.edge_freq = params['edge_cpu_freq']
        self.cloud_cpu_freq_peak = params['cloud_cpu_freq_peak']
        self.edge_cpu_freq_peak = params['edge_cpu_freq_peak']

        self.cloud_off_band_width = params['cloud_off_band_width']
        self.edge_off_band_width = params['edge_off_band_width']
        self.noise_dBm = params['noise_dBm']

        self.w = params['reward_alpha']
        
        self.fc = None
        self.fe = None

        self.action_space = gym.spaces.Discrete(self.edge_num+1)
        self.observation_space = gym.spaces.Box(
            low=-10.0, 
            high=10.0, 
            shape=(10, self.edge_num+1),  # Convert to 1D shape
            dtype=np.float32
        )

        self.task_size_exp_theta = params["task_size_exp_theta"]

        self.init_flag = False
        self.csv_filename = "reward_log.csv"
        # Check if the file exists, if not, create and add header
        # if not os.path.exists(self.csv_filename):
        #     with open(self.csv_filename, mode='w', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow(["rew_t", "rew_e", "reward"])

    def reset(self, seed=42):
        # print("THIS IS RESET FUNCTION\n-----------------------\n")
        if not self.init_flag:
            self.step_cnt = 0
            self.task_to_server.append({
                'task_size': 0,
                'task_user_id': 0
            })
            self.rew_t = 0
            self.rew_e = 0
            self.arrive_flag = False
            self.invalid_act_flag = False
            self.action = ACTION_TO_CLOUD
            
            
            self.finish_time = np.array([0]*(self.edge_num+1))
            self.cloud_off_lists = []
            self.cloud_exe_lists = []
            self.edge_off_lists = []
            self.edge_exe_lists = []


            self.cloud_cpu_freq = np.random.uniform(self.cloud_freq-self.cloud_cpu_freq_peak, self.cloud_freq+self.cloud_cpu_freq_peak)
            self.edge_cpu_freq = [0]*self.edge_num
            #  self.task_size_exp_theta = self.cloud_cpu_freq/self.cloud_C



            for i in range(self.edge_num):
                self.edge_cpu_freq[i] = np.random.uniform(self.edge_freq-self.edge_cpu_freq_peak, self.edge_freq+self.edge_cpu_freq_peak)
                self.edge_cpu_freq[i] = self.fe if self.fe else self.edge_cpu_freq[i]
                self.edge_off_lists.append([])
                self.edge_exe_lists.append([])
                self.task_size_exp_theta += self.edge_cpu_freq[i]/self.edge_C


            self.done = False
            self.reward_buff = []
            self.cloud_dist = np.random.uniform(self.cloud_user_dist_L, self.cloud_user_dist_H, size=(1, self.user_num))
            self.user_dist = self.cloud_dist
            for i in range (self.edge_num):
                edge_dist = np.random.uniform(self.edge_user_dist_L, self.edge_user_dist_H, size=(1, self.user_num))
                self.user_dist = np.concatenate((self.user_dist, edge_dist),axis=0)

            self.cloud_off_datarate, self.edge_off_datarate = self.updata_off_datarate()
            self.step_energy = 0
            self.step_time = 0
            # print(self.edge_off_datarate)
            self.generate_task()

            self.init_flag = True
            # print("RESET---------")
            
        info = {}
        obs = self.get_obs()
        return obs, info
        
    
    def step(self, actions):
        assert self.done==False, 'done'

        self.step_cnt += 1
        self.step_clound_dtime = 0
        

        finished_task = []

        if self.arrive_flag:
            assert actions <= self.edge_num and actions >= ACTION_TO_CLOUD, "action not in the interval %d, %d'%(actions,self.edge_num)"
            self.action = actions
            task = {
                'start_step': self.step_cnt,
                'user_id': self.task_to_server[0]['task_user_id'],
                'size': self.task_to_server[0]['task_size'],
                'remain': self.task_to_server[0]['task_size'],

                'off_time': 0,
                'exe_time': 0
            }

            if actions == ACTION_TO_CLOUD:
                task['to'] = 0
                task['off_energy'] = (task['size']/self.cloud_off_datarate[task['user_id']])*self.cloud_off_power
                task['exe_energy'] = task['size']*self.cloud_k*self.cloud_C*(self.cloud_cpu_freq**2)
                self.step_energy = task['off_energy'] + task['exe_energy']
                self.cloud_off_lists.append(task)

            else:
                e = actions
                task['to'] = e
                task['off_energy'] = (task['size']/self.edge_off_datarate[e-1, task['user_id']])*self.edge_off_power
                task['exe_energy'] = task['size']*self.edge_k*self.edge_C*(self.edge_cpu_freq[e-1]**2)
                self.step_energy = task['off_energy'] + task['exe_energy']
                self.edge_off_lists[e-1].append(task)

            self.rew_t = self.estimate_time_rew()            

            self.generate_task()

            used_time = 0
            while(used_time<self.dt):
                off_estimate_time = []
                exe_estimate_time = []
                task_off_num = len(self.cloud_off_lists)
                task_exe_num = len(self.cloud_exe_lists)

                # Calculate time for min size task
                for i in range(task_off_num):
                    user = self.cloud_off_lists[i]['user_id']
                    estimate_time = self.cloud_off_lists[i]['remain']/self.cloud_off_datarate[user]
                    off_estimate_time.append(estimate_time)

                if task_exe_num > 0:
                    cloud_exe_rate = self.cloud_cpu_freq/(self.cloud_C*task_exe_num)
                    # print("cloud_cpu_freq", self.cloud_cpu_freq)
                    # print("cloud_exe_rate: ", cloud_exe_rate)


                for i in range(task_exe_num):
                    estimate_time = self.cloud_exe_lists[i]['remain']/cloud_exe_rate
                    exe_estimate_time.append(estimate_time)
                    # print(self.cloud_exe_lists[i]['remain'], estimate_time )


                if len(off_estimate_time) + len(exe_estimate_time) > 0:
                    min_time = min(off_estimate_time + exe_estimate_time)
                else:
                    min_time = self.dt

                run_time = min(self.dt - used_time, min_time)
                


                # Calculate remain
                cloud_pre_exe_list = []
                retain_flag_off = np.ones(task_off_num, dtype=bool)
                for i in range(task_off_num):
                    user = self.cloud_off_lists[i]['user_id']
                    self.cloud_off_lists[i]['remain'] -= self.cloud_off_datarate[user]*run_time

                    # Need review
                    self.cloud_off_lists[i]['off_energy'] += self.cloud_off_power*run_time
                    self.cloud_off_lists[i]['off_time'] += run_time
                    self.step_energy += self.cloud_off_lists[i]['off_energy']
                    if self.cloud_off_lists[i]['remain'] <= ZERO_RES:
                        retain_flag_off[i] = False
                        task = deepcopy(self.cloud_off_lists[i])
                        task['remain'] = self.cloud_off_lists[i]['size']
                        cloud_pre_exe_list.append(task)

                pt = 0
                for i in range(task_off_num):
                    if retain_flag_off[i] == False:
                        self.cloud_off_lists.pop(pt)
                    else:
                        pt += 1

                if task_exe_num > 0:
                    cloud_exe_size = self.cloud_cpu_freq*run_time/(self.cloud_C*task_exe_num)
                    cloud_exe_energy=self.cloud_k*run_time*(self.cloud_cpu_freq**3)/task_exe_num

                retain_flag_exe = list(np.ones(task_exe_num, dtype=np.bool_))
                # print("TASK_EXE_NUM")
                # print(task_exe_num)
                # if (task_exe_num > 0):
                #     print(retain_flag_exe[0])
                for i in range(task_exe_num):
                    self.cloud_exe_lists[i]['remain'] -= cloud_exe_size
                    self.cloud_exe_lists[i]['exe_energy'] += cloud_exe_energy
                    self.cloud_exe_lists[i]['exe_time'] += run_time
                    self.step_energy += self.cloud_exe_lists[i]['exe_energy']
                    if self.cloud_exe_lists[i]['remain'] <= ZERO_RES:
                        retain_flag_exe[i] = False
                pt =  0
                for i in range(task_exe_num):
                    if retain_flag_exe[i] == False:
                        self.cloud_exe_lists.pop(pt)
                    else:
                        pt += 1

                # print("task list: ", self.task_to_server)
                # print("cloud_pre_exe_lists: ", len(cloud_pre_exe_list))
                # print("cloud_exe_lists", len(self.cloud_exe_lists))
                self.cloud_exe_lists = self.cloud_exe_lists + cloud_pre_exe_list
                # print("cloud_exe_lists_2: ", len(self.cloud_exe_lists))
                used_time += run_time
                
                # if (len(self.cloud_exe_lists) > 0):

                #     print("task_size_remain: ", self.cloud_exe_lists[0]['remain'])
                # print("task cloud: ", len(self.cloud_exe_lists))
                # print("------------------")

            for n in range(self.edge_num):
                used_time = 0
                while(used_time<self.dt):
                    off_estimate_time = []
                    exe_estimate_time = []
                    task_off_num = len(self.edge_off_lists[n])
                    task_exe_num = len(self.edge_exe_lists[n])

                    for i in range(task_off_num):
                        user = self.edge_off_lists[n][i]['user_id']
                        estimate_time = self.edge_off_lists[n][i]['remain']/self.edge_off_datarate[n, user]
                        off_estimate_time.append(estimate_time)

                    if task_exe_num > 0:
                        edge_exe_rate = self.edge_cpu_freq[n]/(self.edge_C*task_exe_num)
                    for i in range(task_exe_num):
                        estimate_time = self.edge_exe_lists[n][i]['remain']/edge_exe_rate
                        exe_estimate_time.append(estimate_time)
                    if len(off_estimate_time)+len(exe_estimate_time) > 0:
                        min_time = min(off_estimate_time+exe_estimate_time)
                    else:
                        min_time = self.dt
                    run_time = min(self.dt-used_time, min_time)

                    edge_pre_exe_list = []
                    retain_flag_off = np.ones(task_off_num, dtype=bool)
                    for i in range(task_off_num):
                        user = self.edge_off_lists[n][i]['user_id']
                        self.edge_off_lists[n][i]['remain'] -= self.edge_off_datarate[n,user]*run_time
                        self.edge_off_lists[n][i]['off_energy'] += run_time*self.edge_off_power
                        self.edge_off_lists[n][i]['off_time'] += run_time
                        self.step_energy += self.edge_off_lists[n][i]['off_energy']
                        if self.edge_off_lists[n][i]['remain'] <= ZERO_RES:
                            retain_flag_off[i] = False
                            task = deepcopy(self.edge_off_lists[n][i])
                            task['remain'] = self.edge_off_lists[n][i]['size']
                            edge_pre_exe_list.append(task)
                    pt = 0
                    for i in range(task_off_num):
                        if retain_flag_off[i] == False:
                            self.edge_off_lists[n].pop(pt)
                        else:
                            pt += 1

                    if task_exe_num > 0:
                        edge_exe_size = self.edge_cpu_freq[n]*run_time/(self.edge_C*task_exe_num)
                        edge_exe_energy = self.edge_k*run_time*(self.edge_cpu_freq[n]**3)/task_exe_num
                    retain_flag_exe = np.ones(task_exe_num, dtype=bool)
                    for i in range(task_exe_num):
                        self.edge_exe_lists[n][i]['remain'] -= edge_exe_size
                        self.edge_exe_lists[n][i]['exe_energy'] += edge_exe_energy
                        self.edge_exe_lists[n][i]['exe_time'] += run_time
                        self.step_energy += self.edge_exe_lists[n][i]['exe_energy']
                        if self.edge_exe_lists[n][i]['remain'] <= ZERO_RES:
                            retain_flag_exe[i] = False
                    pt = 0
                    for i in range(task_exe_num):
                        if retain_flag_exe[i]==False:
                            self.edge_exe_lists[n].pop(pt)
                        else:
                            pt += 1
                    self.edge_exe_lists[n] = self.edge_exe_lists[n] + edge_pre_exe_list
                    used_time += run_time



            if self.step_cnt >= self.Tmax:
                self.done = True
            done = self.done
            
            self.rew_e = -self.step_energy*10
            # print("step: ", self.step_cnt)
            obs = self.get_obs()

            reward = self.get_reward(finished_task)

            # info = {}
            info = self.get_info(reward)

            truncated = done

            self.task_to_server.pop(0)

            return obs, reward, done, truncated, info

    def get_info(self, reward):
        info = {}
        # info = {
        #         "step": self.step_cnt,
                
        #         "edge_num": self.edge_num,
        #         "action": self.action,
        #         "energy_mWs": self.step_energy,
        #         "time_delay_ms": self.rew_t*-100,
        #         # "cloud_exe_task": len(self.cloud_exe_lists),
        #         "task_to_server": self.task_to_server[0],
        #         "reward": reward,
        #         "exe_in_cloud": self.cloud_exe_lists
                
        #     }

        # for i in range(self.edge_num):
        #     info[f"edge_{i}_exe_task"] = len(self.edge_exe_lists[i])
        
        # print(self.action, "-------", self.rew_e, "------", self.rew_t, "--------", reward)
        # print(info)
        # print("-----------------------------------------")
        return info
    

    
    def generate_task(self):
        task_num = np.random.poisson(self.possion_lamda)
        for i in range(task_num):
            task = {}
            lamda = self.task_size_exp_theta + self.wave_peak*np.sin(self.step_cnt*np.pi/self.wave_cycle)
            task_size = np.random.exponential(lamda)
            task['task_size'] = np.clip(task_size, self.task_size_L, self.task_size_H)
            task['task_user_id'] = np.random.randint(0, self.user_num)
            self.unassigned_tasks.append(task)

        # print("TASK_NUM: ", task_num)

        # for i in range(task_num):
        #     print(self.unassigned_tasks[i])

        if self.step_cnt < self.Tmax:
            if len(self.unassigned_tasks) > 0:
                self.arrive_flag = True
                # self.task_to_server = {}
                self.task_to_server.append(self.unassigned_tasks.pop(0))

            else:
                self.arrive_flag = True
                self.task_to_server.append({
                    "task_size" : 0,
                    "task_user_id" : np.random.randint(0, self.user_num)
                })

    def estimate_time_rew(self):
        remain_list = []
        if self.action == ACTION_TO_CLOUD:
            for task in self.cloud_exe_lists:
                remain_list.append(task['remain'])
            computing_speed = self.cloud_cpu_freq/self.cloud_C
            offload_time = self.task_to_server[0]['task_size']/self.cloud_off_datarate[self.task_to_server[0]['task_user_id']] if self.task_to_server[0]['task_size'] > 0 else 0
        else:
            for task in self.edge_exe_lists[self.action-1]:
                remain_list.append(task['remain'])
            computing_speed = self.edge_cpu_freq[self.action-1]/self.edge_C
            offload_time =  self.task_to_server[0]['task_size']/self.edge_off_datarate[self.action-1,self.task_to_server[0]['task_user_id']] if self.task_to_server[0]['task_size'] > 0 else 0

        remain_list = np.sort(remain_list)

        last_size = 0
        t2 = 0
        task_num = len(remain_list)
        for i in range(task_num):
            size = remain_list[i]
            current_speed=  computing_speed/(task_num-i)
            t2 += (task_num-i)*(size-last_size)/current_speed
            last_size = size

        last_size = 0
        t_norm = 0
        t1 = 0
        task_num = len(remain_list)
        for i in range(task_num):
            size = remain_list[i]
            current_speed = computing_speed/(task_num-i)
            use_t = (size-last_size)/current_speed
            if t_norm + use_t >= offload_time:
                t_cut = offload_time - t_norm
                t1 += (task_num-i)*t_cut
                t_norm = offload_time
                remain_list[i] -= t_cut*current_speed
                remain_list[i] = 0 if remain_list[i]<ZERO_RES else remain_list[i]
                remain_list = remain_list[i:]
                break
            else:
                t1 += (task_num-i) * (size-last_size)/current_speed
                t_norm += use_t
            last_size=size

        remain_list = remain_list.tolist()
        remain_list.append(self.task_to_server[0]['task_size'])
        remain_list = np.sort(remain_list)
        last_size = 0
        task_num = len(remain_list)
        for i in range(task_num):
            size = remain_list[i]
            current_speed = computing_speed/(task_num-i)
            t1 += (task_num-i)*(size-last_size)/current_speed
            last_size = size
        
        #reward_dt = t2 - t1 - offload_time 
        reward_dt = -t1 - offload_time
        self.step_time = t1 + offload_time
        if self.task_to_server[0]['task_size'] > 0:
            reward_dt = reward_dt*0.01
            # reward_de = -self.step_energy*50
        else:
           # reward_de = 0
            reward_dt = 0
        return reward_dt #, reward_de


    def updata_off_datarate(self):
        rayleigh = RAYLEIGH_VAR/2*(np.random.randn(self.edge_num+1, self.user_num)**2 + np.random.randn(self.edge_num+1, self.user_num)**2)  
        path_loss_dB = RAYLEIGH_PATH_LOSS_A*np.log10(self.user_dist/1000) + RAYLEIGH_PATH_LOSS_B
        total_path_loss_IndB = RAYLEIGH_ANTENNA_GAIN - RAYLEIGH_SHADOW_FADING - path_loss_dB
        path_loss = 10**(total_path_loss_IndB/10)
        rayleigh_noise_cloud = 10**((RAYLEIGH_NOISE_dBm-30)/10)*self.cloud_off_band_width;
        rayleigh_noise_edge = 10**((RAYLEIGH_NOISE_dBm-30)/10)*self.edge_off_band_width;
        gain_ = (path_loss*rayleigh)
        cloud_gain = gain_[0,:]/rayleigh_noise_cloud
        edge_gain = gain_[1:,:]/rayleigh_noise_edge
        cloud_noise = 10**((self.noise_dBm-30)/10)*self.cloud_off_band_width;
        edge_noise = 10**((self.noise_dBm-30)/10)*self.edge_off_band_width;
        cloud_off_datarate = self.cloud_off_band_width*np.log2(1 + (self.cloud_off_power*(cloud_gain**2))/cloud_noise)  
        edge_off_datarate = self.edge_off_band_width*np.log2(1 + (self.edge_off_power*(edge_gain**2))/edge_noise)
        # print("EDGE OF DATARATE")
        # print(edge_off_datarate)  
        return cloud_off_datarate, edge_off_datarate
                    
    
    def get_reward(self, finished_task:list=[]):
        
        reward = self.w*self.rew_t + (1.0-self.w)*self.rew_e
        
        # with open(self.csv_filename, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([self.rew_t, self.rew_e, reward])

        return reward

    def get_obs(self):
        obs = {}

        servers = []
        cloud = []
        cloud.append(1)
        cloud.append(self.step_energy*10)
        #cloud.append(self.rew_t)
        cloud.append(self.step_time*0.01)
        cloud.append(self.cloud_cpu_freq/1e9)
        cloud.append(self.edge_num)
        cloud.append(self.task_to_server[0]['task_size']/1e6)
        cloud.append(1-self.done)
        cloud.append(self.cloud_off_datarate[self.task_to_server[0]['task_user_id']]/1e6/100)
        cloud.append(len(self.cloud_exe_lists))
        cloud.append(sum(x["remain"] for x in self.cloud_exe_lists) if len(self.cloud_exe_lists) > 0 else 0)

        # task_exe_hist = np.zeros([60])
        # for task in self.cloud_exe_lists:
        #     task_feature = int(task['remain']/1e6)
        #     task_feature = min(task_feature, 59)  # Ensure index is in range
        #     task_exe_hist[task_feature] += 1

        # cloud = np.concatenate([np.array(cloud), task_exe_hist], axis=0)
        servers.append(cloud)
        
        for ii in range(self.edge_num):
            edge = []
            edge.append(1)
            edge.append(self.step_energy*10)
            edge.append(self.step_time*0.01)
            edge.append(self.edge_cpu_freq[ii]/1e9)
            edge.append(self.edge_num)
            edge.append(self.task_to_server[0]['task_size']/1e6)
            edge.append(1-self.done)
            edge.append(self.edge_off_datarate[ii, self.task_to_server[0]['task_user_id']]/1e6/100)
            edge.append(len(self.edge_exe_lists[ii]))
            edge.append(sum(x["remain"] for x in self.edge_exe_lists[ii]) if len(self.edge_exe_lists[ii]) > 0 else 0)
            

            # task_exe_hist = np.zeros([60])
            # for task in self.edge_exe_lists[ii]:
            #     task_feature = int(task['remain']/1e6)
            #     task_feature = min(task_feature, 59)
            #     task_exe_hist[task_feature] += 1

            # edge = np.concatenate([np.array(edge), task_exe_hist], axis=0)
            
            servers.append(edge)

        # Convert to 1D shape (1, 67 * (edge_num+1))
        obs['servers'] = np.array(servers).swapaxes(0, 1)#.flatten().reshape(1, -1)
       
        re = obs['servers']
        return re  # Now matches the (1, 67 * (edge_num + 1)) shape


def main():
    conf_file = 'config.json'
    config = json.load(open(conf_file, 'r'))
    
    config = config['config1']
    env = MECEnv(params=config)
    obs = env.reset()
    
    # print("RESET STEP -----")
    # print(env.get_obs())
    # obs, reward, done, infor = env.step(actions=ACTION_TO_CLOUD)
    # print(obs.shape)
    # print("AFTER ACTION TO CLOUD ----")
    # print(obs)

if __name__=="__main__":
    main()
