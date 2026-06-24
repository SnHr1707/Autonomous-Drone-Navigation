import torch
import numpy as np
from collections import deque
from colosseum_env import ColosseumNavEnv
from rddpg import RDDPGAgent

def main():
    # TARGET UPDATED TO NEW AIRSIM COORDINATES
    env = ColosseumNavEnv(target_pos=(-6.22, -4.16, 0.55))
    agent = RDDPGAgent()
    
    model_path = "models/RDDPG"
    print("Loading models...")
    try:
        agent.actor.load_state_dict(torch.load(f"{model_path}/actor_2000.pth"))
        agent.actor.eval()
    except Exception as e:
        print(f"Could not load model. Error: {e}")
        return

    seq_size = 5
    episodes = 5
    
    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            history = deque(maxlen=seq_size)
            for _ in range(seq_size):
                history.append(obs['image'])
            state = obs['state']
            
            done = False
            print(f"Starting AI Flight - Episode {ep+1}!")
            
            while not done:
                img_seq = np.stack(history)
                action = agent.get_action(img_seq, state, noise_scale=0.0)
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                history.append(obs['image'])
                state = obs['state']
                
            print("Episode Finished.")
    finally:
        env.close()

if __name__ == "__main__":
    main()