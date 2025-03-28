from model import CircularCarFollowingModel
from visualization import CircularTrackVisualizer
from analysis import CircularTrackAnalyzer

def main():
    """
    主程序：演示三车跟随模型的完整工作流程
    """
    # 创建模型实例
    # 不同参数可以测试模型在各种场景下的表现
    model = CircularCarFollowingModel(
        initial_velocities=[60.0, 70.0, 90.0],  # 初始速度
        initial_positions=[800.0, 600.0, 220.0],  # 初始位置
        d=(30.0, 50.0),  # 期望车距范围
        parameters=None  # 使用默认参数
    )
    
    # 运行模拟
    print("Running vehicle following simulation...")
    model.run_simulation()
    
    # 创建可视化工具
    visualizer = CircularTrackVisualizer(model)
    
    # 创建分析工具
    analyzer = CircularTrackAnalyzer(model)
    
    # 执行分析
    print("\nConduct system stability analysis...")
    stability_analysis = analyzer.analyze_stability()
    
    # 可视化结果并保存
    print("\nPlot the simulation results...")
    visualizer.plot_results(save=True)  # 添加save=True参数
    
    # 创建车辆动画并保存
    print("\nGenerate vehicle motion animation...")
    visualizer.animate_vehicles(save=True)  # 添加save=True参数

if __name__ == "__main__":
    main()