import torch
import numpy as np
import time
from collections import deque
from colosseum_env import ColosseumNavEnv
from rddpg import RDDPGAgent

def main():
    # TARGET UPDATED TO NEW AIRSIM COORDINATES
    env = ColosseumNavEnv(target_pos=(-8.00, -6.82, -0.98))
    agent = RDDPGAgent()
    
    model_path = "models/RDDPG"
    print("Loading models...")
    try:
        # Make sure to put the EXACT filename of your best model here
        agent.actor.load_state_dict(torch.load(f"{model_path}/actor_SUCCESS_ep2528.pth"))
        agent.actor.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Could not load model. Error: {e}")
        return

    seq_size = 5
    episodes = 5
    
    try:
        for ep in range(episodes):
            print(f"\n====================================")
            print(f"Starting AI Flight - Episode {ep+1}!")
            print(f"====================================")
            
            obs, _ = env.reset()
            # Give Unreal Engine physics a second to settle after teleporting
            time.sleep(1.0) 
            
            history = deque(maxlen=seq_size)
            for _ in range(seq_size):
                history.append(obs['image'])
            state = obs['state']
            
            done = False
            step_count = 0
            
            while not done:
                step_count += 1
                img_seq = np.stack(history)
                
                # Ask AI for an action (No random noise)
                action = agent.get_action(img_seq, state, noise_scale=0.0)
                
                # Take the step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # --- TELEMETRY PRINTING ---
                # state[3:6] holds the relative distance to the target in our env
                dist_to_target = np.linalg.norm(state[3:6]) 
                # state[0:3] holds the drone's current velocity
                current_speed = np.linalg.norm(state[0:3])
                
                print(f"Step {step_count:03d} | Dist: {dist_to_target:.2f}m | Speed: {current_speed:.2f}m/s | Action (vx,vy,vz): [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]")
                
                # Update history and state for the next loop
                history.append(obs['image'])
                state = obs['state']
                
            print(f"Episode {ep+1} Finished.")
            
    finally:
        env.close()

if __name__ == "__main__":
    main()