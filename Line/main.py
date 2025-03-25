from model import ThreeCarFollowingModel
import visualization as vis
import analysis as ana

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
    
    # plot result
    vis.plot_results(model)
    
    # analyse stability
    ana.analyze_stability(model)
    
    # create animation
    vis.animate_vehicles(model)


if __name__ == "__main__":
    main()