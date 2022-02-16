from gym.envs.registration import register
import sys 
sys.path.append('/content/drive/MyDrive/xyz_master/master_thesis')









register(
    id='xyzmultidrone-aviary-v0',
    entry_point='gym_pybullet_drones.envs.multi_agent_rl:xyzMultiAviary',
)
