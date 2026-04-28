# 實施計劃

- [ ] 1. 編寫錯誤條件探索測試
  - **Property 1: Bug Condition** - 機車垂直行駛軌跡跳動問題
  - **重要**: 此測試必須在未修復的程式碼上失敗 - 失敗確認錯誤存在
  - **不要嘗試修復測試或程式碼當它失敗時**
  - **注意**: 此測試編碼了預期行為 - 它將在實施後通過時驗證修復
  - **目標**: 展示證明錯誤存在的反例
  - **範圍化PBT方法**: 對於確定性錯誤，將屬性範圍限定為具體的失敗案例以確保可重現性
  - 測試機車垂直於監視器方向行駛時出現位置跳動(來自設計中的錯誤條件)
  - 測試斷言應該匹配設計中的預期行為屬性
  - 在未修復程式碼上執行測試
  - **預期結果**: 測試失敗(這是正確的 - 證明錯誤存在)
  - 記錄發現的反例以了解根本原因
  - 當測試編寫、執行並記錄失敗時標記任務完成
  - _Requirements: 1.1, 1.2, 1.3_

- [ ] 2. 編寫保持屬性測試(在實施修復之前)
  - **Property 2: Preservation** - 非機車垂直行駛行為保持
  - **重要**: 遵循觀察優先方法
  - 觀察未修復程式碼上非錯誤輸入的行為
  - 編寫基於屬性的測試捕捉來自保持需求的觀察行為模式
  - 基於屬性的測試生成許多測試案例以提供更強保證
  - 在未修復程式碼上執行測試
  - **預期結果**: 測試通過(這確認要保持的基線行為)
  - 當測試在未修復程式碼上編寫、執行並通過時標記任務完成
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 3. 機車軌跡平滑化修復

  - [x] 3.1 在inference_config.yaml中實施修復
    - 新增motorcycle_smooth配置區段
    - 基於car_heading_smooth配置進行優化
    - 設定displacement_threshold為0.15(從0.25降低)
    - 設定heading_sat_coords_jitter_radius為0.3(從0.5降低)
    - 設定heading_sat_coords_jitter_frames為20(從15增加)
    - 調整heading_ema參數:
      - alpha_min: 0.01(從0.02降低)
      - alpha_max: 0.1(從0.15降低)  
      - speed_ref: 2.0(從3.5調整)
    - 設定heading_max_jump為2(從3降低)
    - 設定heading_recovery_frames為40(從30增加)
    - 設定speed_ema_alpha為0.1(從0.15降低)
    - _Bug_Condition: isBugCondition(input) 其中 input.vehicle_class IN ['motor', 'two_wheeler'] AND input.movement_direction PERPENDICULAR_TO camera_direction_
    - _Expected_Behavior: expectedBehavior(result) 來自設計 - 機車垂直行駛時產生平滑直線軌跡_
    - _Preservation: 保持需求來自設計 - 汽車和其他車輛類型的軌跡處理效果保持不變_
    - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4_

  - [ ] 3.2 驗證錯誤條件探索測試現在通過
    - **Property 1: Expected Behavior** - 機車垂直行駛軌跡平滑化
    - **重要**: 重新執行步驟1中的相同測試 - 不要編寫新測試
    - 步驟1中的測試編碼了預期行為
    - 當此測試通過時，它確認預期行為得到滿足
    - 執行步驟1中的錯誤條件探索測試
    - **預期結果**: 測試通過(確認錯誤已修復)
    - _Requirements: 設計中的預期行為屬性_

  - [ ] 3.3 驗證保持測試仍然通過
    - **Property 2: Preservation** - 非機車垂直行駛行為保持
    - **重要**: 重新執行步驟2中的相同測試 - 不要編寫新測試
    - 執行步驟2中的保持屬性測試
    - **預期結果**: 測試通過(確認無回歸)
    - 確認修復後所有測試仍然通過(無回歸)

- [ ] 4. 檢查點 - 確保所有測試通過
  - 確保所有測試通過，如有問題請詢問用戶。