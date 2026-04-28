#!/usr/bin/env python3
"""
測試腳本：驗證 visualization tab 側欄滾動功能
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout

# 添加 trafficlab 模組路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from trafficlab.gui.tabs.tab_visualization import VisualizationTab
    
    class TestWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("測試 Visualization Tab 側欄滾動")
            self.setGeometry(100, 100, 1200, 400)  # 設置較低的高度來測試滾動
            
            # 創建 visualization tab
            self.viz_tab = VisualizationTab()
            self.setCentralWidget(self.viz_tab)
    
    def main():
        app = QApplication(sys.argv)
        window = TestWindow()
        window.show()
        
        print("測試視窗已開啟")
        print("請檢查側欄是否可以滾動，特別是在視窗高度較小時")
        print("按 Ctrl+C 或關閉視窗來結束測試")
        
        sys.exit(app.exec_())
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"導入錯誤: {e}")
    print("請確保在正確的目錄中運行此腳本")
except Exception as e:
    print(f"其他錯誤: {e}")