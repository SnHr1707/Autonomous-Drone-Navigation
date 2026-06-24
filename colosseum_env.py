import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import time
import cv2

class ColosseumNavEnv(gym.Env):
    # Target: (-8.00, -6.82, -0.98)
    def __init__(self, target_pos=(-8.00, -6.82, -0.98), max_steps=400, img_height=72, img_width=128):
        super(ColosseumNavEnv, self).__init__()
        
        self.target_pos = np.array(target_pos, dtype=np.float32)
        self.img_height = img_height
        self.img_width = img_width
        
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0.0, high=1.0, shape=(1, img_height, img_width), dtype=np.float32),
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        })
        
        self.step_duration = 0.5
        self.max_steps = max_steps
        self.current_step = 0
        self.prev_dist = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        try:
            self.client.moveByVelocityAsync(0, 0, 0, duration=0.1)
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except:
            pass
            
        try:
            self.client.reset()
            time.sleep(0.5)
            
            # Restored Original Spawn: Spawn exactly at 0,0, popped up by 0.5 meters
            pose = airsim.Pose(airsim.Vector3r(0.0, 0.0, -0.5), airsim.to_quaternion(0, 0, 0))
            self.client.simSetVehiclePose(pose, ignore_collision=True)
            
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.client.moveByVelocityAsync(0, 0, 0, duration=2.0)
            time.sleep(1.0)
            
            obs = self._get_obs()
            self.prev_dist = np.linalg.norm(self.target_pos - self._get_drone_pos())
            
        except Exception as e:
            print(f"Error during reset (API Timeout/Glitch): {e}. Reconnecting safely...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            obs = {'image': np.zeros((1, self.img_height, self.img_width), dtype=np.float32), 
                   'state': np.zeros(6, dtype=np.float32)}
            self.prev_dist = 10.0
            
        return obs, {}
        
    def step(self, action):
        self.current_step += 1
        
        vx = np.clip(float(action[0]), -5.0, 5.0)
        vy = np.clip(float(action[1]), -5.0, 5.0)
        vz = np.clip(float(action[2]), -2.0, 2.0)
        
        padded_duration = self.step_duration + 1.5 
        
        has_collided = False
        obs = {'image': np.zeros((1, self.img_height, self.img_width), dtype=np.float32), 'state': np.zeros(6, dtype=np.float32)}
        drone_pos = np.zeros(3)
        current_dist = self.prev_dist
        reward = 0.0
        
        # Wrapped in Try-Except to prevent "No API Call Received" from crashing your whole training session!
        try:
            self.client.moveByVelocityAsync(vx, vy, vz, duration=padded_duration)
            
            start_time = time.time()
            
            while time.time() - start_time < self.step_duration:
                collision_info = self.client.simGetCollisionInfo()
                if collision_info.has_collided:
                    has_collided = True
                    print(f"CRASHED into: {collision_info.object_name} at step {self.current_step}")
                    self.client.moveByVelocityAsync(0, 0, 0, duration=0.1)
                    break
                time.sleep(0.05)
                
            obs = self._get_obs()
            drone_pos = self._get_drone_pos()
            current_dist = np.linalg.norm(self.target_pos - drone_pos)
            
            # STUCK CHECKER: Prevents the drone from spending 200 steps scraping against a wall
            # that lacks a proper collision hitbox in Unreal Engine.
            kinematics = self.client.simGetGroundTruthKinematics()
            actual_speed = np.linalg.norm([kinematics.linear_velocity.x_val, kinematics.linear_velocity.y_val, kinematics.linear_velocity.z_val])
            commanded_speed = np.linalg.norm([vx, vy, vz])
            
            if commanded_speed > 2.0 and actual_speed < 0.2 and self.current_step > 5:
                print("Drone wedged/stuck against wall. Triggering collision penalty.")
                has_collided = True
                
        except Exception as e:
            print(f"AirSim API Error (Timeout/Physics Glitch): {e}")
            print("Treating as a fatal crash to safely reset environment without crashing Python.")
            has_collided = True
        
        # ==========================================
        # REWARD SYSTEM 
        # ==========================================
        # 1. Forward Progress
        progress = self.prev_dist - current_dist
        reward += progress * 20.0  
        self.prev_dist = current_dist
        
        # 2. Velocity Alignment 
        if current_dist > 0.5 and not has_collided:
            try:
                dir_to_target = (self.target_pos - drone_pos) / current_dist
                kinematics = self.client.simGetGroundTruthKinematics()
                vel = np.array([kinematics.linear_velocity.x_val, kinematics.linear_velocity.y_val, kinematics.linear_velocity.z_val])
                alignment = np.dot(dir_to_target, vel)
                reward += 1.0 * alignment 
            except:
                pass
            
        # 3. Action Smoothness 
        reward -= 0.05 * np.linalg.norm(action)
        
        # 4. End State Checks
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        if has_collided:
            reward -= 200.0   
            terminated = True
            
        elif current_dist < 1.0: 
            reward += 300.0 
            terminated = True
            print("SUCCESS! Reached target!")
            
        # 5. Extreme Map Boundaries (Only kills the drone if it physically breaks out of the map and falls into the void)
        elif abs(drone_pos[0]) > 100 or abs(drone_pos[1]) > 100 or drone_pos[2] > 20.0 or drone_pos[2] < -30.0:
            reward -= 200.0
            terminated = True
            print("Map boundary breached (Clipping into void). Resetting.")
            
        return obs, reward, terminated, truncated, {}
        
    def _get_drone_pos(self):
        pos = self.client.simGetGroundTruthKinematics().position
        return np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)

    def _get_obs(self):
        kinematics = self.client.simGetGroundTruthKinematics()
        drone_pos = np.array([kinematics.position.x_val, kinematics.position.y_val, kinematics.position.z_val], dtype=np.float32)
        drone_vel = np.array([kinematics.linear_velocity.x_val, kinematics.linear_velocity.y_val, kinematics.linear_velocity.z_val], dtype=np.float32)
        rel_target = self.target_pos - drone_pos
        state_vec = np.concatenate((drone_vel, rel_target))

        try:
            responses = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthVis, True)])
            if responses and len(responses) > 0:
                img1d = np.array(responses[0].image_data_float, dtype=np.float32)
                img1d = np.clip(255 * 3 * img1d, 0, 255).astype(np.uint8)
                img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
                img_resized = cv2.resize(img2d, (self.img_width, self.img_height))
                img_normalized = np.float32(img_resized) / 255.0
                image = np.expand_dims(img_normalized, axis=0) 
            else:
                image = np.zeros((1, self.img_height, self.img_width), dtype=np.float32)
        except Exception as e:
            image = np.zeros((1, self.img_height, self.img_width), dtype=np.float32)
            
        return {'image': image, 'state': state_vec}

    def close(self):
        print("Safely shutting down API control...")
        try:
            self.client.moveByVelocityAsync(0, 0, 0, duration=0.1)
            time.sleep(0.1)
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except:
            pass