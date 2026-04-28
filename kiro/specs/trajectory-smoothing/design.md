# 機車軌跡平滑化錯誤修復設計

## 概述

本設計文檔針對機車垂直行駛時出現的軌跡跳動問題，提出一個基於kinematics參數優化的解決方案。問題的核心在於機車體積較小，容易受到偵測噪音影響，特別是在垂直於監視器方向行駛時，微小的偵測變化會導致顯著的橫向位置跳動。解決策略是在inference_config.yaml中新增專門針對機車的配置模式，通過調整displacement_threshold和heading相關參數來實現更強的位置平滑化。

## 詞彙表

- **Bug_Condition (C)**: 觸發錯誤的條件 - 機車垂直於監視器方向行駛時出現位置跳動
- **Property (P)**: 期望行為 - 機車垂直行駛時應產生平滑的直線軌跡
- **Preservation**: 現有的汽車和其他車輛類型的軌跡處理效果必須保持不變
- **TrackSmoother**: 位於`trafficlab/motion/kinematics.py`中的類別，負責軌跡平滑化處理
- **displacement_threshold**: kinematics配置中的參數，控制位置更新的敏感度
- **heading_sat_coords_jitter_radius**: 控制位置抖動檢測的半徑閾值
- **two_wheeler/motor**: 在prior_dimensions.json中定義的機車類別標識符

## 錯誤詳情

### 錯誤條件

錯誤在機車與監視器方向垂直行駛時出現。TrackSmoother類別無法有效過濾機車偵測中的噪音，導致計算出的sat_coords在橫向方向上出現不穩定的跳動。

**正式規格說明:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type VehicleTrackingData
  OUTPUT: boolean
  
  RETURN input.vehicle_class IN ['motor', 'two_wheeler']
         AND input.movement_direction PERPENDICULAR_TO camera_direction
         AND input.lateral_position_variance > acceptable_threshold
         AND trajectory_appears_jagged(input.position_history)
END FUNCTION
```

### 範例

- **範例1**: 機車從畫面左側垂直向右行駛，期望軌跡為水平直線，實際軌跡呈現鋸齒狀
- **範例2**: 機車從畫面上方垂直向下行駛，期望軌跡為垂直直線，實際軌跡左右搖擺
- **範例3**: 機車在十字路口垂直穿越，期望軌跡為平滑直線，實際軌跡出現多個橫向跳動點
- **邊界案例**: 機車以45度角行駛時，期望軌跡平滑但可能仍有輕微抖動

## 預期行為

### 保持不變的行為

**不變行為:**
- 汽車使用car_heading_smooth模式時的軌跡平滑效果必須完全保持不變
- 其他車輛類型(卡車、巴士、三輪車等)的軌跡處理效果必須保持不變
- 機車平行於監視器方向行駛時的現有軌跡品質必須保持不變

**範圍:**
所有不涉及機車垂直行駛的輸入都應該完全不受此修復影響。這包括:
- 汽車的所有行駛方向和軌跡處理
- 機車的平行行駛和其他角度行駛
- 其他車輛類型的所有軌跡處理

## 假設根本原因

基於錯誤描述，最可能的問題是:

1. **位置敏感度過高**: 當前的displacement_threshold值對機車來說過於敏感
   - 機車體積小，偵測框變化相對較大
   - 垂直行駛時橫向噪音被放大

2. **抖動檢測不足**: heading_sat_coords_jitter_radius參數未針對機車優化
   - 機車的抖動模式與汽車不同
   - 需要更嚴格的抖動檢測

3. **平滑化強度不足**: heading相關的EMA參數對機車噪音處理不夠強
   - alpha_min和alpha_max值需要調整
   - speed_ref參考值可能不適合機車速度範圍

4. **缺乏車輛類型特定配置**: 系統未根據車輛類型選擇不同的kinematics參數
   - 需要實現基於vehicle_class的配置選擇機制

## 正確性屬性

Property 1: Bug Condition - 機車垂直行駛軌跡平滑化

_對於任何_ 機車垂直於監視器方向行駛的輸入(isBugCondition返回true)，修復後的軌跡處理系統應該產生平滑的直線軌跡，橫向位置變化應該最小化，消除鋸齒狀跳動現象。

**驗證: Requirements 2.1, 2.2, 2.3**

Property 2: Preservation - 非機車垂直行駛行為保持

_對於任何_ 不是機車垂直行駛的輸入(isBugCondition返回false)，修復後的系統應該產生與原始系統完全相同的結果，保持所有現有車輛類型和行駛方向的軌跡處理效果。

**驗證: Requirements 3.1, 3.2, 3.3, 3.4**

## 修復實作

### 所需變更

假設我們的根本原因分析正確:

**檔案**: `inference_config.yaml`

**新增配置**: `motorcycle_smooth`

**具體變更**:
1. **新增機車專用配置**: 在configs區段新增motorcycle_smooth配置
   - 基於car_heading_smooth配置進行優化
   - 針對機車特性調整關鍵參數

2. **調整displacement_threshold**: 從0.25降低到0.15
   - 減少位置更新的敏感度
   - 過濾更多的偵測噪音

3. **強化抖動檢測**: 調整heading_sat_coords_jitter_radius從0.5到0.3
   - 更嚴格的抖動檢測標準
   - 增加jitter_frames到20幀

4. **優化平滑化參數**: 調整heading_ema參數
   - alpha_min從0.02降低到0.01 (更強的平滑化)
   - alpha_max從0.15降低到0.1 (限制最大變化率)
   - speed_ref調整到2.0 (適合機車速度範圍)

5. **增強穩定性參數**: 調整其他相關參數
   - heading_max_jump從3降低到2度
   - heading_recovery_frames增加到40幀
   - speed_ema_alpha從0.15降低到0.1

## 測試策略

### 驗證方法

測試策略採用兩階段方法：首先在未修復的程式碼上展示錯誤的反例，然後驗證修復後的程式碼能正確工作並保持現有行為。

### 探索性錯誤條件檢查

**目標**: 在實作修復之前，展示錯誤的反例。確認或反駁根本原因分析。如果反駁，我們需要重新假設。

**測試計劃**: 編寫測試來模擬機車垂直行駛的場景，並斷言軌跡應該是平滑的。在未修復的程式碼上執行這些測試以觀察失敗並理解根本原因。

**測試案例**:
1. **垂直向右行駛測試**: 模擬機車從左到右垂直行駛 (在未修復程式碼上會失敗)
2. **垂直向下行駛測試**: 模擬機車從上到下垂直行駛 (在未修復程式碼上會失敗)
3. **十字路口穿越測試**: 模擬機車在十字路口垂直穿越 (在未修復程式碼上會失敗)
4. **高噪音環境測試**: 模擬偵測噪音較高的情況下機車垂直行駛 (在未修復程式碼上可能失敗)

**預期反例**:
- 軌跡出現鋸齒狀跳動而非平滑直線
- 可能原因: displacement_threshold過高、抖動檢測不足、平滑化參數不當

### 修復檢查

**目標**: 驗證對於所有錯誤條件成立的輸入，修復後的函數產生預期行為。

**虛擬碼:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := motorcycleSmoothKinematics(input)
  ASSERT smoothTrajectoryBehavior(result)
END FOR
```

### 保持檢查

**目標**: 驗證對於所有錯誤條件不成立的輸入，修復後的函數產生與原始函數相同的結果。

**虛擬碼:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT originalKinematics(input) = fixedKinematics(input)
END FOR
```

**測試方法**: 建議使用基於屬性的測試進行保持檢查，因為:
- 它自動在輸入域中生成許多測試案例
- 它捕捉手動單元測試可能遺漏的邊界案例
- 它為所有非錯誤輸入提供強有力的行為不變保證

**測試計劃**: 首先在未修復程式碼上觀察汽車和其他車輛的行為，然後編寫基於屬性的測試來捕捉該行為。

**測試案例**:
1. **汽車軌跡保持**: 驗證汽車使用car_heading_smooth模式的軌跡效果在修復後保持不變
2. **機車平行行駛保持**: 驗證機車平行於監視器方向行駛的軌跡在修復後保持不變
3. **其他車輛類型保持**: 驗證卡車、巴士等其他車輛類型的軌跡處理在修復後保持不變
4. **配置選擇保持**: 驗證現有配置的選擇機制在修復後繼續正常工作

### 單元測試

- 測試新的motorcycle_smooth配置參數載入
- 測試機車類別識別邏輯
- 測試垂直行駛場景下的軌跡平滑化效果
- 測試邊界案例(極低速、高噪音環境)

### 基於屬性的測試

- 生成隨機的機車軌跡數據並驗證垂直行駛時的平滑化效果
- 生成隨機的車輛配置並驗證非機車車輛的軌跡處理保持不變
- 測試跨多種場景的配置選擇邏輯正確性

### 整合測試

- 測試完整的inference pipeline使用新配置處理機車軌跡
- 測試在不同監視器角度下機車垂直行駛的軌跡品質
- 測試混合車輛場景中各車輛類型的軌跡處理效果