import airsim
import numpy as np
import time

def main():
    print("Connecting to Unreal Engine...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    try:
        trajectories = np.load("flight_paths.npy", allow_pickle=True).item()
    except FileNotFoundError:
        print("Error: Could not find 'flight_paths.npy'. Run record_trajectories.py first!")
        return

    print("Clearing old drawings...")
    client.simFlushPersistentMarkers()

    max_frames = max([len(t) for t in trajectories.values()])
    
    print("=========================================================")
    print("🎬 PLAYBACK STARTING IN 5 SECONDS!")
    print("Switch to the Unreal Engine window and position your camera!")
    print("Hit Windows Key + G or open OBS to start screen recording.")
    print("=========================================================")
    time.sleep(5)
    
    # Plot a giant Green Sphere at the target
    target = airsim.Vector3r(-8.00, -6.82, -0.98)
    client.simPlotSphere([target], radius=0.8, color_rgba=[0.0, 1.0, 0.0, 1.0], is_persistent=True)
    
    # Animate the frames
    for frame in range(1, max_frames):
        grey_lines = []
        red_lines = []
        grey_heads = []
        red_heads = []
        
        for name, traj in trajectories.items():
            if frame < len(traj):
                # Get the points for this specific frame
                p_prev = traj[frame-1]
                p_curr = traj[frame]
                
                v_prev = airsim.Vector3r(float(p_prev[0]), float(p_prev[1]), float(p_prev[2]))
                v_curr = airsim.Vector3r(float(p_curr[0]), float(p_curr[1]), float(p_curr[2]))
                
                if "2000" in name:
                    red_lines.extend([v_prev, v_curr])
                    red_heads.append(v_curr)
                else:
                    grey_lines.extend([v_prev, v_curr])
                    grey_heads.append(v_curr)
        
        # 1. Draw Persistent Trails (These stay permanently)
        if grey_lines:
            client.simPlotLineList(grey_lines, color_rgba=[0.5, 0.5, 0.5, 0.2], thickness=3.0, is_persistent=True)
        if red_lines:
            client.simPlotLineList(red_lines, color_rgba=[1.0, 0.0, 0.0, 1.0], thickness=15.0, is_persistent=True)
            
        # 2. Draw Moving "Drone" Orbs (These vanish after 0.1 seconds, creating an animation effect)
        if grey_heads:
            client.simPlotPoints(grey_heads, color_rgba=[0.7, 0.7, 0.7, 0.8], size=15.0, duration=0.1, is_persistent=False)
        if red_heads:
            client.simPlotPoints(red_heads, color_rgba=[1.0, 0.0, 0.0, 1.0], size=30.0, duration=0.1, is_persistent=False)
            
        time.sleep(0.05) # Adjust this to change the playback speed (0.05 = ~20 FPS)
        
    print("Playback complete! You can stop recording.")
    
if __name__ == "__main__":
    main()