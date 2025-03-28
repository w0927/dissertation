from model import ThreeCarFollowingModel
import visualization as vis
import analysis as ana
import os

def main():
    """
    Main function to run the simulation.
    """
    # set parameters
    parameters = {
        'a11': 2.0,  # F1 acceleration when both distances are large
        'a10': 1.5,  # F1 acceleration when front gap large, rear gap small
        'a01': 1.0,  # F1 deceleration when front gap small, rear gap large
        'a00': 0.5,  # F1 deceleration when both distances are small
        'b1': 2.0,   # F2 acceleration when distance to F1 is large
        'b0': 1.0,   # F2 deceleration when distance to F1 is small
        'c1': 1.5,   # L deceleration when distance to F1 is large
        'c0': 1.0    # L acceleration when distance to F1 is small
    }
    
    # customize the initial speed
    initial_velocities = [25.0, 20.0, 15.0]  # [L, F1, F2]
    
    # customize the initial location
    initial_positions = [200.0, 120.0, 60.0]  # [L, F1, F2]
    
    # distance threshold
    distance_threshold = 30.0  # m
    
    # Create output directory
    output_dir = "simulation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Running the three-car following model with custom initial conditions...")
    print(f"Lead car: position = {initial_positions[0]} m, velocity = {initial_velocities[0]} m/s")
    print(f"Following car 1: position = {initial_positions[1]} m, velocity = {initial_velocities[1]} m/s")
    print(f"Following car 2: position = {initial_positions[2]} m, velocity = {initial_velocities[2]} m/s")
    print(f"Initial distances: L-F1 = {initial_positions[0] - initial_positions[1]} m, F1-F2 = {initial_positions[1] - initial_positions[2]} m")
    
    # create and run the model
    model = ThreeCarFollowingModel(
        initial_velocities=initial_velocities,
        initial_positions=initial_positions,
        d=distance_threshold,
        parameters=parameters
    )
    
    model.run_simulation()
    
    # Create a theme with the specified three colors
    custom_colors = {
        'plot_colors': {
            'lead_car': '#FFD700',        # 金色 - 领头车
            'following_car1': '#00B4D8',  # 湖蓝色 - 跟随车1
            'following_car2': '#7FFFD4',  # 碧绿色 - 跟随车2
            'threshold_line': '#FF6B6B',  # 红色 - 阈值线
            'background': 'white',        # 白色背景
            'grid': '#E0E0E0'             # 浅灰色网格
        },
        'animation_colors': {
            'lead_car': '#FFD700',        # 金色 - 领头车
            'following_car1': '#00B4D8',  # 湖蓝色 - 跟随车1
            'following_car2': '#7FFFD4',  # 碧绿色 - 跟随车2
            'car_edge': '#333333',        # 深灰色车辆边缘
            'background': 'white',        # 白色背景
            'text': '#333333'             # 深灰色文本
        }
    }
    
    # Create folders to store different styles of output
    custom_dir = os.path.join(output_dir, "custom_colors")
    os.makedirs(custom_dir, exist_ok=True)
    
    # Analyze the stability
    ana.analyze_stability(model)
    
    # Draw the result with a custom color and save
    plots_path = os.path.join(custom_dir, "vehicle_plots.png")
    print(f"Creating plots with custom colors and saving to {plots_path}...")
    vis.plot_results(model, save_path=plots_path, colors=custom_colors['plot_colors'])
    
    # Create animations with custom colors and save them
    gif_path = os.path.join(custom_dir, "vehicle_animation.gif")
    print(f"Creating animation with custom colors and saving to {gif_path}...")
    ani = vis.animate_vehicles(model, save_path=gif_path, fps=10, colors=custom_colors['animation_colors'])
    
    print("Simulation with custom colors completed successfully.")

if __name__ == "__main__":
    main()