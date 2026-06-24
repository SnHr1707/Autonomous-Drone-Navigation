import os
import re
import glob
import torch
import numpy as np
import time
from collections import deque
from colosseum_env import ColosseumNavEnv
from rddpg import RDDPGAgent

def main():
    env = ColosseumNavEnv(target_pos=(-8.00, -6.82, -0.98))
    agent = RDDPGAgent()
    
    models_dir = "models/RDDPG"
    actor_files = glob.glob(f"{models_dir}/actor_*.pth")
    
    # Extract only the numbered files (ignores SUCCESS files) and sort them
    valid_files = []
    for f in actor_files:
        match = re.search(r'actor_(\d+)\.pth', f)
        if match:
            valid_files.append((f, int(match.group(1))))
    valid_files.sort(key=lambda x: x[1]) 
    
    trajectories = {}
    seq_size = 5
    
    print(f"Found {len(valid_files)} models to evaluate.")
    
    try:
        for filepath, ep_num in valid_files:
            print(f"Evaluating Episode {ep_num}...")
            agent.actor.load_state_dict(torch.load(filepath))
            agent.actor.eval()
            
            obs, _ = env.reset()
            time.sleep(1.0) # Let physics settle
            
            history = deque(maxlen=seq_size)
            for _ in range(seq_size):
                history.append(obs['image'])
            state = obs['state']
            
            done = False
            traj = []
            
            while not done:
                pos = env._get_drone_pos()
                traj.append(pos.copy())
                
                img_seq = np.stack(history)
                action = agent.get_action(img_seq, state, noise_scale=0.0) 
                
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                history.append(obs['image'])
                state = obs['state']
            
            traj.append(env._get_drone_pos().copy())
            trajectories[f"actor_{ep_num}"] = np.array(traj)
            
    finally:
        env.close()
        
    np.save("flight_paths.npy", trajectories)
    print("\n✅ All trajectories saved to 'flight_paths.npy'!")

if __name__ == "__main__":
    main()