#!/usr/bin/env python3
"""
測試配置文件
"""

import yaml

def test_config():
    """測試配置文件"""
    
    # 讀取配置
    with open('inference_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("=== 配置文件測試 ===")
    
    # 檢查motorcycle_smooth配置是否存在
    if 'motorcycle_smooth' in config['configs']:
        print("✅ motorcycle_smooth配置存在")
        
        motorcycle_config = config['configs']['motorcycle_smooth']
        kinematics_config = motorcycle_config['kinematics']
        
        # 檢查lateral_correction配置
        if 'lateral_correction' in kinematics_config:
            print("✅ lateral_correction配置存在")
            
            lateral_config = kinematics_config['lateral_correction']
            print(f"橫向修正啟用: {lateral_config.get('enabled', False)}")
            print(f"車輛類別: {lateral_config.get('vehicle_classes', [])}")
            print(f"檢測閾值: {lateral_config.get('threshold', 'N/A')}")
            print(f"修正強度: {lateral_config.get('strength', 'N/A')}")
            print(f"窗口大小: {lateral_config.get('window_size', 'N/A')}")
            print(f"最大距離: {lateral_config.get('max_distance', 'N/A')}")
            
            # 檢查自適應強度配置
            if 'adaptive_strength' in lateral_config:
                adaptive = lateral_config['adaptive_strength']
                print(f"自適應啟用: {adaptive.get('enabled', False)}")
                print(f"最小強度: {adaptive.get('min_strength', 'N/A')}")
                print(f"最大強度: {adaptive.get('max_strength', 'N/A')}")
                print(f"連續閾值: {adaptive.get('consecutive_threshold', 'N/A')}")
            else:
                print("❌ adaptive_strength配置缺失")
        else:
            print("❌ lateral_correction配置缺失")
    else:
        print("❌ motorcycle_smooth配置缺失")
    
    # 檢查車輛類別匹配
    print("\n=== 車輛類別匹配測試 ===")
    if 'motorcycle_smooth' in config['configs']:
        lateral_config = config['configs']['motorcycle_smooth']['kinematics'].get('lateral_correction', {})
        vehicle_classes = lateral_config.get('vehicle_classes', [])
        
        test_classes = ['motor', 'two_wheeler', 'car', 'truck', 'bus']
        for vehicle_class in test_classes:
            match = vehicle_class in vehicle_classes
            print(f"{vehicle_class}: {'✅ 匹配' if match else '❌ 不匹配'}")

if __name__ == "__main__":
    test_config()