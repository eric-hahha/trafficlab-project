# Visualization Tab 側欄滾動修復

## 問題描述
在 TrafficLab 的 visualization tab 中，當視窗高度不足時，側欄底部的控制選項會被推到視窗外面而無法點選。

## 解決方案
將側欄內容包裝在 `QScrollArea` 中，使其在內容超出視窗高度時可以滾動。

## 修改內容

### 1. 添加導入
在 `trafficlab/gui/tabs/tab_visualization.py` 中添加了 `QScrollArea` 導入：
```python
from PyQt5.QtWidgets import (..., QScrollArea)
```

### 2. 修改 `_build_sidebar` 方法
- 創建 `QScrollArea` 作為側欄容器
- 設置滾動區域屬性：
  - 固定寬度 340px
  - 啟用垂直滾動條（需要時顯示）
  - 禁用水平滾動條
  - 啟用 `widgetResizable` 以適應內容

### 3. 調整佈局
- 移除原本的 `sidebar_layout.addStretch()` 以避免滾動區域中的佈局問題
- 將側欄 widget 設置到滾動區域中
- 更新摺疊/展開功能以操作滾動區域而非原始側欄

## 主要變更

```python
def _build_sidebar(self, main_layout):
    # 創建滾動區域
    sidebar_scroll = QScrollArea()
    sidebar_scroll.setFixedWidth(340)
    sidebar_scroll.setWidgetResizable(True)
    sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    
    # 創建可滾動的側欄內容
    sidebar = QWidget()
    sidebar_layout = QVBoxLayout(sidebar)
    
    # ... 原有的側欄內容 ...
    
    # 將側欄設置到滾動區域
    sidebar_scroll.setWidget(sidebar)
    
    # 更新摺疊功能以操作滾動區域
    def _toggle_sidebar(checked):
        if checked:
            sidebar_scroll.setVisible(False)  # 改為操作 sidebar_scroll
            self.btn_toggle_sidebar.setText('▶')
        else:
            sidebar_scroll.setVisible(True)   # 改為操作 sidebar_scroll
            self.btn_toggle_sidebar.setText('◀')
    
    # 添加滾動區域到主佈局
    main_layout.addWidget(sidebar_scroll)  # 改為添加 sidebar_scroll
```

## 效果
- 當視窗高度不足時，側欄會顯示垂直滾動條
- 用戶可以滾動查看所有控制選項
- 保持原有的摺疊/展開功能
- 不影響其他功能的正常運作

## 測試
可以運行 `test_sidebar_scroll.py` 來測試修復效果：
```bash
python test_sidebar_scroll.py
```

將視窗高度調整得很小，檢查側欄是否出現滾動條並可以正常滾動。