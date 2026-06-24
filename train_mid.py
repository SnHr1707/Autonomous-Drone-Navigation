import os
import glob
import re
import numpy as np
import torch
import traceback
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from colosseum_env import ColosseumNavEnv
from rddpg import RDDPGAgent

def main():
    models_dir = "models/RDDPG"
    os.makedirs(models_dir, exist_ok=True)
    writer = SummaryWriter(log_dir="logs/RDDPG")

    env = ColosseumNavEnv(target_pos=(-8.00, -6.82, -0.98))
    agent = RDDPGAgent()
    
    # ==========================================
    # AUTO-RESUME TRAINING LOGIC
    # ==========================================
    # ==========================================
    # AUTO-RESUME TRAINING LOGIC
    # ==========================================
    actor_files = glob.glob(f"{models_dir}/actor_*.pth")
    start_episode = 0
    
    # Safely extract episode numbers from filenames, ignoring non-matching files
    valid_checkpoints = []
    for f in actor_files:
        match = re.search(r'(\d+)\.pth', f) # Looks for the number right before .pth
        if match:
            valid_checkpoints.append((f, int(match.group(1))))
            
    if valid_checkpoints:
        # Find the file with the highest episode number
        latest_actor = max(valid_checkpoints, key=lambda x: x[1])[0]
        start_episode = max(valid_checkpoints, key=lambda x: x[1])[1]
        
        # Safely convert the actor filename to the critic filename
        latest_critic = latest_actor.replace('actor', 'critic')
        
        print(f"Loading pre-trained model from episode {start_episode}...")
        try:
            agent.actor.load_state_dict(torch.load(latest_actor))
            agent.critic.load_state_dict(torch.load(latest_critic))
            agent.actor_target.load_state_dict(agent.actor.state_dict())
            agent.critic_target.load_state_dict(agent.critic.state_dict())
            print("Successfully loaded! Continuing training.")
        except Exception as e:
            print(f"Failed to load models: {e}. Starting fresh.")
            start_episode = 0

    seq_size = 5
    noise = 1.0 if start_episode == 0 else max(0.1, 1.0 * (0.9995 ** start_episode)) 
    noise_decay = 0.9995
    beta = 0.4
    total_steps = 0
    episodes = start_episode + 2000 
    
    # NEW: Track consecutive successes to ensure it wasn't just a lucky random flight
    consecutive_successes = 0 

    print("=====================================================")
    print(f"Starting Training at Episode {start_episode + 1}!")
    print("Press Ctrl+C at any time to safely stop the script.")
    print("=====================================================")
    
    try:
        for ep in range(start_episode, episodes):
            obs, _ = env.reset()
            
            history = deque(maxlen=seq_size)
            for _ in range(seq_size):
                history.append(obs['image'])
            
            state = obs['state']
            done = False
            ep_reward = 0
            step = 0
            
            while not done:
                img_seq = np.stack(history)
                action = agent.get_action(img_seq, state, noise_scale=noise)
                
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                next_history = history.copy()
                next_history.append(next_obs['image'])
                next_img_seq = np.stack(next_history)
                next_state = next_obs['state']
                
                img_seq_t = torch.FloatTensor(img_seq).unsqueeze(0).to(agent.device)
                state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                action_t = torch.FloatTensor(action).unsqueeze(0).to(agent.device)
                
                with torch.no_grad():
                    q_val = agent.critic(img_seq_t, state_t, action_t).item()
                    next_img_t = torch.FloatTensor(next_img_seq).unsqueeze(0).to(agent.device)
                    next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
                    next_act = agent.actor_target(next_img_t, next_state_t)
                    q_next = agent.critic_target(next_img_t, next_state_t, next_act).item()
                    target_val = reward + (0 if done else agent.gamma * q_next)
                    td_error = abs(target_val - q_val)

                agent.memory.add(td_error, (img_seq, state, action, reward, next_img_seq, next_state, float(done)))
                
                history = next_history
                state = next_state
                ep_reward += reward
                total_steps += 1
                step += 1
                
                if total_steps > 500 or start_episode > 0:
                    a_loss, c_loss = agent.train(beta)
                    if total_steps % 10 == 0:
                        writer.add_scalar("Loss/Actor", a_loss, total_steps)
                        writer.add_scalar("Loss/Critic", c_loss, total_steps)
                        
                noise = max(0.1, noise * noise_decay)
                beta = min(1.0, beta + 0.0001)

            print(f"Episode {ep+1}/{episodes} | Reward: {ep_reward:.2f} | Steps: {step} | Noise: {noise:.3f}")
            writer.add_scalar("Reward/Episode", ep_reward, ep)
            
            # ==========================================
            # EARLY STOPPING & SUCCESS HANDLING LOGIC
            # ==========================================
            # With the new continuous sliding scale, a properly braked drone (Speed <= 1.0)
            # will generate a total episode reward of 480 to 600+.
            if ep_reward >= 480.0:
                consecutive_successes += 1
                print(f"\n🎉 AMAZING! The drone reached the target perfectly! (Streak: {consecutive_successes}/3)")
                
                torch.save(agent.actor.state_dict(), f"{models_dir}/actor_SUCCESS_ep{ep+1}.pth")
                torch.save(agent.critic.state_dict(), f"{models_dir}/critic_SUCCESS_ep{ep+1}.pth")
                
                if consecutive_successes >= 3:
                    print("\n✅ GOAL MET: The model has perfectly stopped at the target 3 times in a row!")
                    print("Stopping training early. You are ready to test!")
                    break 
            else:
                consecutive_successes = 0
            
            # Normal saving routine every 50 episodes
            if (ep + 1) % 50 == 0:
                torch.save(agent.actor.state_dict(), f"{models_dir}/actor_{ep+1}.pth")
                torch.save(agent.critic.state_dict(), f"{models_dir}/critic_{ep+1}.pth")

    except KeyboardInterrupt:
        print("\nTraining interrupted manually by user (Ctrl+C).")
    except Exception as e:
        print(f"\nCRITICAL PYTHON ERROR OCCURRED:")
        traceback.print_exc()
    finally:
        if hasattr(env, 'close'):
            env.close()
        writer.close()
        print("Safe exit complete. You can now safely hit 'Stop' in Unreal Engine.")

if __name__ == "__main__":
    main()