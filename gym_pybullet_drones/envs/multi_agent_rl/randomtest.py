import  pybullet as p
import time
from gym_pybullet_drones.envs.multi_agent_rl.xyzMultiAviary import xyzMultiAviary
import gym
import numpy as np
env = gym.make('xyzmultidrone-aviary-v0')
obs = env.reset()
a = env.observation_space
b = env.action_space
agent_num = env.NUM_DRONES
print("*"*50)
print(f"obs = {obs}")
print(f"obs[0]={obs[0]}")
print("*"*50)
neighbor_radius = env.NEIGHBOURHOOD_RADIUS
print(f"neighbor_radius = {neighbor_radius}")
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
for i in  range(10000):
    print(i)
    if i<1000:
        action = {i :np.array([.3,.3,.3]) for i in range(agent_num)}
    else:
        action = {i: np.array([-.3, -.3, .3]) for i in range(agent_num)}
    obs,reward,done,info = env.step(action)
    step = env.step_counter
    clipp_position = obs[0][0:3]
    pos = env.pos[0]
    fly_R = env.fly_reward
    target_R = env.target_reward
    height_R = env.height_reward
    neighbor_R = env.neighbor_reward
    obstacle_R  = env.neighbor_reward
    print(env.target_reward[0])
    # print(f"fly_R :{fly_R}")
    # print(f"target_R : {target_R}")
    # print(f"height_R : {height_R}")
    # print(f"neighbor_R : {neighbor_R}")
    # print(f"obstacle_R : {obstacle_R}")
    # if i%100==0:
    #     print(obs)
    #obstacle_done = env.obstacle_done
    # print(f"real_position={pos}")
    # # print(f"position={clipp_position}")
    # print(f"reward ={reward}")
    # print(f"fly_error = {fly_error}")
    # print(f"neighbor_error={neighbor_error}")
    #print(f"obstacle_error={obstacle_error}")
    #print(f"obstacle_done = {obstacle_done}")
    # print(f"target_error ={target_error}")
    # print(f"height_error = {height_error}")
    # print(f"done = {done}")

    #print(done)
    #print(env.DRONE_IDS) #1,2,3,4,5
    #print(env.PLANE_ID) 0
    # print(env.id_tuple)
