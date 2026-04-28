#!/usr/bin/env python3
"""
測試機車橫向修正功能
"""

import yaml
import sys
import os

# 添加項目路徑
sys.path.append('.')

from trafficlab.motion.kinematics import MotorcycleLateralCorrector, TrackSmoother

def test_motorcycle_correction():
    """測試機車橫向修正功能"""
    
    # 讀取配置
    with open('inference_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 獲取motorcycle_smooth配置
    motorcycle_config = config['configs']['motorcycle_smooth']
    kinematics_config = motorcycle_config['kinematics']
    
    print("=== 配置測試 ===")
    print(f"lateral_correction配置存在: {'lateral_correction' in kinematics_config}")
    
    if 'lateral_correction' in kinematics_config:
        lateral_config = kinematics_config['lateral_correction']
        print(f"橫向修正啟用: {lateral_config.get('enabled', False)}")
        print(f"車輛類別: {lateral_config.get('vehicle_classes', [])}")
        print(f"檢測閾值: {lateral_config.get('threshold', 'N/A')}")
        print(f"修正強度: {lateral_config.get('strength', 'N/A')}")
    
    print("\n=== 創建測試 ===")
    
    # 測試創建MotorcycleLateralCorrector
    try:
        corrector = MotorcycleLateralCorrector(kinematics_config, 'motor')
        print("✅ MotorcycleLateralCorrector創建成功")
        print(f"橫向修正啟用: {corrector.lateral_enabled}")
        print(f"車輛類別: {corrector.vehicle_classes}")
        print(f"是否為機車: {corrector.is_motorcycle()}")
    except Exception as e:
        print(f"❌ MotorcycleLateralCorrector創建失敗: {e}")
    
    # 測試創建標準TrackSmoother
    try:
        smoother = TrackSmoother(kinematics_config)
        print("✅ TrackSmoother創建成功")
    except Exception as e:
        print(f"❌ TrackSmoother創建失敗: {e}")
    
    print("\n=== 車輛類別測試 ===")
    test_classes = ['motor', 'two_wheeler', 'car', 'truck', 'bus']
    
    for vehicle_class in test_classes:
        try:
            corrector = MotorcycleLateralCorrector(kinematics_config, vehicle_class)
            is_motorcycle = corrector.is_motorcycle()
            print(f"{vehicle_class}: {'✅ 機車' if is_motorcycle else '❌ 非機車'}")
        except Exception as e:
            print(f"{vehicle_class}: ❌ 錯誤 - {e}")

if __name__ == "__main__":
    test_motorcycle_correction()