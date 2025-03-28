from model import CircularCarFollowingModel
from visualization import CircularTrackVisualizer
from analysis import CircularTrackAnalyzer

def main():
    """
    Main program: Demonstrates the complete workflow of the three-car follow model
    """
    # Ceratet a model instance 
    # Different parameters can be used to test the model's performance in various scenarios
    model = CircularCarFollowingModel(
        initial_velocities=[60.0, 70.0, 90.0],  # Initial speed
        initial_positions=[800.0, 600.0, 220.0],  # Initial position
        d=(30.0, 50.0),  # Expected distance range
        parameters=None  # Use default parameters
    )
    
    # Operation similation
    print("Running vehicle following simulation...")
    model.run_simulation()
    
    # Create visual tools
    visualizer = CircularTrackVisualizer(model)
    
    # Create analysis tools
    analyzer = CircularTrackAnalyzer(model)
    
    # Execution analysis
    print("\nConduct system stability analysis...")
    stability_analysis = analyzer.analyze_stability()
    
    # Store the visual results
    print("\nPlot the simulation results...")
    visualizer.plot_results(save=True)  # Add parameters: save=True
    
    # Create and save the animations
    print("\nGenerate vehicle motion animation...")
    visualizer.animate_vehicles(save=True)  # Add parameters: save=True

if __name__ == "__main__":
    main()