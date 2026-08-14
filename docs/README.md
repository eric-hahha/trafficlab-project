# 文件索引

以下文件維護「多方案/多方法比較」的狀態總覽（哪個方案已採用、哪個受阻、哪個待評估），討論相關主題前應先讀該文件確認現況，狀態以文件內容為準：

- 高度校正（parallax correction）演算法狀態 → [height-correction-algorithm-survey.md](height-correction-algorithm-survey.md)
- 車輛朝向（heading）估算方法狀態 → [method-survey.md](method-survey.md)
- 車輛定位（localization）方案狀態 → [localization-methods.md](localization-methods.md)

## 分析記錄

針對特定 replay/frame/物件的一次性數據分析，供之後查閱比對：

- 關鍵點距離 vs. 車輛模板偏差（test21-4 frame162 tracked_id5） → [keypoint-distance-deviation-test21-4-frame162-id5.md](keypoint-distance-deviation-test21-4-frame162-id5.md)
- Keypoint 長寬比與垂直度分析（output/haware 全 location，套用車輛樣板標準化） → [wheel-aspect-ratio-perpendicularity-analysis.md](wheel-aspect-ratio-perpendicularity-analysis.md)
- 跨幀同車 wheelbase 自洽性檢查（output/haware 全 location，驗證 base homography 位置相關失真） → [wheelbase-track-consistency-homography-check.md](wheelbase-track-consistency-homography-check.md)
