# render_video.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def main():
    print("Loading flight data...")
    trajectories = np.load("flight_paths.npy", allow_pickle=True).item()
    
    # Setup the 3D Plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Find min/max limits so the camera doesn't jump around
    all_points = np.vstack(list(trajectories.values()))
    x_min, x_max = all_points[:,0].min(), all_points[:,0].max()
    y_min, y_max = all_points[:,1].min(), all_points[:,1].max()
    z_min, z_max = all_points[:,2].min(), all_points[:,2].max()
    
    ax.set_xlim([x_min - 1, x_max + 1])
    ax.set_ylim([y_min - 1, y_max + 1])
    
    # AirSim uses NED (Negative Z is UP). We invert it here so the plot looks natural.
    ax.set_zlim([-z_max - 1, -z_min + 1]) 
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Altitude (Z)')
    ax.set_title("AI Learning Progression\nGrey = Past Iterations | Red = Final Model")
    
    # Plot Target and Start Locations
    target_pos = (-8.00, -6.82, -0.98 * -1) # Invert Z for target
    ax.scatter(*target_pos, color='green', marker='*', s=300, label='Target', zorder=5)
    ax.scatter(0, 0, 0.5, color='blue', marker='o', s=150, label='Start', zorder=5) # Invert start Z (-0.5 -> 0.5)
    
    # Prepare lines for animation
    lines = {}
    max_frames = 0
    
    for name, traj in trajectories.items():
        max_frames = max(max_frames, len(traj))
        
        # Apply the styling you requested
        if "2000" in name:
            line, = ax.plot([], [], [], color='red', linewidth=3.5, zorder=10, label='Model 2000')
        else:
            line, = ax.plot([], [], [], color='grey', linewidth=1, alpha=0.3, zorder=1)
            
        lines[name] = line
        
    ax.legend()
    
    # The function that draws each frame of the video
    def update(frame):
        for name, traj in trajectories.items():
            idx = min(frame, len(traj) - 1)
            current_traj = traj[:idx+1]
            
            lines[name].set_data(current_traj[:, 0], current_traj[:, 1])
            lines[name].set_3d_properties(current_traj[:, 2] * -1) # Invert Z for plotting
            
        return list(lines.values())
        
    print(f"Generating animation with {max_frames} frames...")
    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=50, blit=False)
    
    # Save as GIF (works natively without installing FFMPEG)
    print("Saving as GIF. This might take a minute...")
    ani.save("learning_progression.gif", writer='pillow', fps=15)
    print("✅ Video saved as learning_progression.gif!")
    
    # NOTE: If you specifically want an MP4, you must install ffmpeg on your PC, 
    # then comment out the GIF line above and uncomment the line below:
    # ani.save("learning_progression.mp4", writer='ffmpeg', fps=15)

if __name__ == "__main__":
    main()