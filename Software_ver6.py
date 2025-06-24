import sys
import random
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # 使用Qt5后端
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QStatusBar, QMessageBox, QFileDialog,
    QFormLayout, QLineEdit, QGridLayout, QFrame,
    QProgressBar, QSpacerItem, QSizePolicy, QDoubleSpinBox,
    QSpinBox, QTableWidget, QTableWidgetItem, QCheckBox,
    QComboBox, QDialog, QDialogButtonBox, QStackedWidget
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import Function_Format_for_Capacity_Sizing_of_a_Solar_Storage_Charging_Energy_Park_2025_6_19 as fcn1

class EnergyOptimizationPlatform(QMainWindow):
    def __init__(self):

         # 设置matplotlib全局字体
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

        super().__init__()
        self.init_styles()
        self.progress_max = 11  # 总进度项数（背景信息2 + 方案预期6 + 数据导入3）
        self.progress_value = 0
        self.energy_management_settings = None
        self.light_data = None
        self.traffic_data = None
        self.load_data = None
        self.background_imported = False  # 标记是否已点击背景带入按钮
        self.initUI()
        self.setup_signal_slots()

    def init_styles(self):
        style = """
        QMainWindow { background-color: #f0f2f5; }
        QGroupBox { 
            border: 1px solid #dcdfe6; 
            border-radius: 4px; 
            margin-top: 8px; 
            font-weight: 500;
            background-color: white;
        }
        QGroupBox::title { 
            subcontrol-origin: padding; 
            left: 10px; 
            padding: 0 5px; 
            background-color: white;
            font-size: 20px;
        }
        QPushButton {
            background-color: #409eff;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            min-width: 80px;
            font-size: 16px;
        }
        QPushButton:hover {
            background-color: #66b1ff;
        }
        QPushButton:pressed {
            background-color: #3a8ee6;
        }
        QPushButton:disabled {
            background-color: #c0c4cc;
        }
        QStatusBar { padding: 4px; font-size: 14px; }
        QLineEdit { padding: 6px; border: 1px solid #dcdfe6; border-radius: 4px; }
        .CurvePlaceholder {
            border: 1px dashed #dcdfe6;
            border-radius: 4px;
            background-color: #fafafa;
            min-height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .CurvePlaceholder QLabel {
            color: #909399;
            font-size: 14px;
        }
        QTableWidget {
            gridline-color: #dcdfe6;
            selection-background-color: #e6f7ff;
        }
        QTableWidget::item {
            padding: 6px;
        }
        QTableWidget::item:selected {
            color: #1890ff;
        }
        QProgressBar {
            border: 2px solid #dcdfe6;
            border-radius: 10px;
            background-color: #f5f7fa;
            height: 20px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #409eff;
            border-radius: 8px;
        }
        QProgressBar::indicator {
            color: transparent; /* 隐藏进度条上的数字 */
        }
        """
        QApplication.setStyle('Fusion')
        self.setStyleSheet(style)

    def initUI(self):
        self.setWindowTitle('光储充一体化系统配置与能量优化平台')
        self.setGeometry(100, 100, 1200, 800)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)
        self.start_widget = self.create_start_widget()
        main_layout.addWidget(self.start_widget)
        self.main_widget = self.create_main_widget()
        main_layout.addWidget(self.main_widget)
        self.main_widget.hide()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.show_status("就绪")

    def create_start_widget(self):
        start_widget = QWidget()
        start_layout = QVBoxLayout(start_widget)
        title_label = QLabel("光储充一体化系统配置与能量优化平台")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 36px; font-weight: bold; color: #409eff; margin-bottom: 30px;")
        start_layout.addWidget(title_label)
        btn_layout = QHBoxLayout()
        config_btn = QPushButton("进入配置")
        config_btn.setMinimumHeight(40)
        config_btn.clicked.connect(self.switch_to_main)
        btn_layout.addWidget(config_btn)
        history_btn = QPushButton("查看历史数据")
        history_btn.setMinimumHeight(40)
        history_btn.clicked.connect(self.show_history_data)
        btn_layout.addWidget(history_btn)
        start_layout.addLayout(btn_layout)
        return start_widget

    def create_main_widget(self):
        main_widget = QWidget()
        main_tab_layout = QVBoxLayout(main_widget)
        tabs = QTabWidget()
        main_tab_layout.addWidget(tabs)
        self.tabs = tabs  # 保存引用以便后续使用
        self.create_info_import_tab(tabs)
        self.create_optimization_config_tab(tabs)
        self.create_energy_management_tab(tabs)
        self.create_result_view_tab(tabs)
        back_layout = QHBoxLayout()
        back_btn = QPushButton("返回首页")
        back_btn.setMinimumHeight(40)
        back_btn.clicked.connect(self.switch_to_start)
        back_layout.addWidget(back_btn, alignment=Qt.AlignRight)
        main_tab_layout.addLayout(back_layout)
        return main_widget

    def setup_signal_slots(self):
        """优化信号槽连接，避免重复连接导致多次弹窗"""
        # 先断开已有的信号连接（避免重复连接）
        try:
            self.import_light_btn.clicked.disconnect()
            self.import_traffic_btn.clicked.disconnect()
            self.import_load_btn.clicked.disconnect()
        except TypeError:
            pass  # 首次连接时可能没有已连接的信号，捕获异常
        
        # 使用命名函数替代lambda，提高可维护性
        self.import_light_btn.clicked.connect(self.import_light_data)
        self.import_traffic_btn.clicked.connect(self.import_traffic_data)
        self.import_load_btn.clicked.connect(self.import_load_data)
        
        # 其他信号连接（保持原有逻辑）
        signals = [
            self.area_edit.textChanged,
            self.vehicle_count_edit.textChanged,
            self.charger_dc_min_edit.valueChanged,
            self.charger_dc_max_edit.valueChanged,
            self.charger_ac_min_edit.valueChanged,
            self.charger_ac_max_edit.valueChanged,
            self.solar_dc_min_edit.valueChanged,
            self.solar_ac_min_edit.valueChanged,
            self.solar_dc_max_edit.valueChanged,
            self.solar_ac_max_edit.valueChanged,
            self.storage_dc_min_edit.valueChanged,
            self.storage_dc_max_edit.valueChanged,
            self.storage_ac_min_edit.valueChanged,
            self.storage_ac_max_edit.valueChanged
        ]
        for signal in signals:
            signal.connect(self.update_progress)
        
        self.clear_light_btn.clicked.connect(lambda: self.clear_data("light"))
        self.clear_traffic_btn.clicked.connect(lambda: self.clear_data("traffic"))
        self.clear_load_btn.clicked.connect(lambda: self.clear_data("load"))
        self.import_from_background_btn.clicked.connect(self.import_from_background)

    def show_status(self, msg):
        self.statusBar().showMessage(msg, 3000)

    def switch_to_main(self):
        self.start_widget.hide()
        self.main_widget.show()

    def switch_to_start(self):
        self.main_widget.hide()
        self.start_widget.show()

    def show_history_data(self):
        QMessageBox.information(self, "查看历史数据", "这里可以实现查看历史数据的功能")

    def create_info_import_tab(self, tabs):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        left_widget = self.create_left_info_widget()
        right_widget = self.create_right_info_widget()
        layout.addWidget(left_widget, 1)
        layout.addWidget(right_widget, 2)
        tabs.addTab(tab, "信息导入")

    def create_left_info_widget(self):
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(24)
        self.create_background_info_group(left_layout)
        left_layout.addSpacing(20)
        self.create_scheme_expectation_group(left_layout)
        left_layout.addStretch()
        self.setup_progress_layout(left_layout)
        return left_widget

    def create_right_info_widget(self):
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 0, 0, 0)
        data_import_frame = QFrame()
        data_import_frame.setFrameShape(QFrame.StyledPanel)
        data_import_layout = QVBoxLayout(data_import_frame)  # 使用QVBoxLayout替代QGridLayout
        data_import_layout.setContentsMargins(0, 0, 0, 0)
        data_import_layout.setSpacing(16)  # 设置垂直间距
        self.create_data_import_group(data_import_layout)
        right_layout.addWidget(data_import_frame)
        return right_widget

    def setup_progress_layout(self, layout):
        progress_layout = QVBoxLayout()
        progress_label = QLabel("数据填写进度:")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.progress_max)
        self.progress_bar.setValue(0)
        self.next_step_btn = QPushButton("下一步")
        self.next_step_btn.setMinimumHeight(40)
        self.next_step_btn.clicked.connect(self.next_step)
        self.next_step_btn.setEnabled(False)
        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.next_step_btn)
        layout.addLayout(progress_layout)

    def create_background_info_group(self, parent_layout):
        group = QGroupBox("背景信息")
        group.setSizePolicy(group.sizePolicy().horizontalPolicy(), group.sizePolicy().Expanding)
        group_layout = QVBoxLayout()
        group_layout.addSpacing(20)
        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(12)
        self.area_edit = QLineEdit()
        self.vehicle_count_edit = QLineEdit()
        form_layout.addRow("园区面积 (平方米):", self.area_edit)
        form_layout.addRow("园区电动汽车保有量:", self.vehicle_count_edit)
        group_layout.addLayout(form_layout)
        group.setLayout(group_layout)
        parent_layout.addWidget(group)

    def create_scheme_expectation_group(self, parent_layout):
        group = QGroupBox("方案预期")
        group.setSizePolicy(group.sizePolicy().horizontalPolicy(), group.sizePolicy().Expanding)
        group_layout = QVBoxLayout()
        
        # 按钮布局 - 靠右对齐
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()  # 添加伸缩项将按钮推到右侧
        self.import_from_background_btn = QPushButton("背景代入")
        self.import_from_background_btn.setMinimumHeight(36)
        self.import_from_background_btn.clicked.connect(self.import_from_background)
        btn_layout.addWidget(self.import_from_background_btn)
        
        group_layout.addLayout(btn_layout)
        group_layout.addSpacing(10)
        
        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(12)
        
        # 充电桩数（直流）
        charger_dc_label = QLabel("充电桩数 (直流个):")
        charger_dc_layout = QHBoxLayout()
        self.charger_dc_min_edit = QSpinBox()
        self.charger_dc_min_edit.setRange(0, 1000)
        self.charger_dc_min_edit.setValue(20)
        charger_dc_layout.addWidget(self.charger_dc_min_edit)
        charger_dc_layout.addWidget(QLabel("至"))
        self.charger_dc_max_edit = QSpinBox()
        self.charger_dc_max_edit.setRange(0, 1000)
        self.charger_dc_max_edit.setValue(50)
        charger_dc_layout.addWidget(self.charger_dc_max_edit)
        form_layout.addRow(charger_dc_label, charger_dc_layout)

        # 充电桩数（交流）
        charger_ac_label = QLabel("充电桩数 (交流个):")
        charger_ac_layout = QHBoxLayout()
        self.charger_ac_min_edit = QSpinBox()
        self.charger_ac_min_edit.setRange(0, 1000)
        self.charger_ac_min_edit.setValue(20)
        charger_ac_layout.addWidget(self.charger_ac_min_edit)
        charger_ac_layout.addWidget(QLabel("至"))
        self.charger_ac_max_edit = QSpinBox()
        self.charger_ac_max_edit.setRange(0, 1000)
        self.charger_ac_max_edit.setValue(50)
        charger_ac_layout.addWidget(self.charger_ac_max_edit)
        form_layout.addRow(charger_ac_label, charger_ac_layout)

        # 光伏容量（直流）
        solar_dc_label = QLabel("光伏容量 (直流MVA):")
        solar_dc_layout = QHBoxLayout()
        self.solar_dc_min_edit = QDoubleSpinBox()
        self.solar_dc_min_edit.setRange(0, 10)
        self.solar_dc_min_edit.setDecimals(1)
        self.solar_dc_min_edit.setValue(0.5)
        solar_dc_layout.addWidget(self.solar_dc_min_edit)
        solar_dc_layout.addWidget(QLabel("至"))
        self.solar_dc_max_edit = QDoubleSpinBox()
        self.solar_dc_max_edit.setRange(0, 10)
        self.solar_dc_max_edit.setDecimals(1)
        self.solar_dc_max_edit.setValue(1.0)
        solar_dc_layout.addWidget(self.solar_dc_max_edit)
        form_layout.addRow(solar_dc_label, solar_dc_layout)

        # 光伏容量（交流）
        solar_ac_label = QLabel("光伏容量 (交流MVA):")
        solar_ac_layout = QHBoxLayout()
        self.solar_ac_min_edit = QDoubleSpinBox()
        self.solar_ac_min_edit.setRange(0, 10)
        self.solar_ac_min_edit.setDecimals(1)
        self.solar_ac_min_edit.setValue(0.5)
        solar_ac_layout.addWidget(self.solar_ac_min_edit)
        solar_ac_layout.addWidget(QLabel("至"))
        self.solar_ac_max_edit = QDoubleSpinBox()
        self.solar_ac_max_edit.setRange(0, 10)
        self.solar_ac_max_edit.setDecimals(1)
        self.solar_ac_max_edit.setValue(1.0)
        solar_ac_layout.addWidget(self.solar_ac_max_edit)
        form_layout.addRow(solar_ac_label, solar_ac_layout)

        # 储能容量（直流）
        storage_dc_label = QLabel("储能容量 (直流MVA):")
        storage_dc_layout = QHBoxLayout()
        self.storage_dc_min_edit = QDoubleSpinBox()
        self.storage_dc_min_edit.setRange(0, 5)
        self.storage_dc_min_edit.setDecimals(1)
        self.storage_dc_min_edit.setValue(0.3)
        storage_dc_layout.addWidget(self.storage_dc_min_edit)
        storage_dc_layout.addWidget(QLabel("至"))
        self.storage_dc_max_edit = QDoubleSpinBox()
        self.storage_dc_max_edit.setRange(0, 5)
        self.storage_dc_max_edit.setDecimals(1)
        self.storage_dc_max_edit.setValue(0.8)
        storage_dc_layout.addWidget(self.storage_dc_max_edit)
        form_layout.addRow(storage_dc_label, storage_dc_layout)

        # 储能容量（交流）
        storage_ac_label = QLabel("储能容量 (交流MVA):")
        storage_ac_layout = QHBoxLayout()
        self.storage_ac_min_edit = QDoubleSpinBox()
        self.storage_ac_min_edit.setRange(0, 5)
        self.storage_ac_min_edit.setDecimals(1)
        self.storage_ac_min_edit.setValue(0.3)
        storage_ac_layout.addWidget(self.storage_ac_min_edit)
        storage_ac_layout.addWidget(QLabel("至"))
        self.storage_ac_max_edit = QDoubleSpinBox()
        self.storage_ac_max_edit.setRange(0, 5)
        self.storage_ac_max_edit.setDecimals(1)
        self.storage_ac_max_edit.setValue(0.8)
        storage_ac_layout.addWidget(self.storage_ac_max_edit)
        form_layout.addRow(storage_ac_label, storage_ac_layout)

        group_layout.addLayout(form_layout)
        group.setLayout(group_layout)
        parent_layout.addWidget(group)

    def create_data_import_group(self, parent_layout):
        group = QGroupBox("数据导入")
        group_layout = QVBoxLayout()

        # 光照强度导入
        light_frame = QFrame()
        light_frame.setObjectName("lightFrame")
        light_layout = QVBoxLayout(light_frame)
        light_header_layout = QHBoxLayout()
        light_label = QLabel("光照强度:")
        self.import_light_btn = QPushButton("导入")
        self.clear_light_btn = QPushButton("清除")
        light_header_layout.addWidget(light_label)
        light_header_layout.addWidget(self.import_light_btn)
        light_header_layout.addWidget(self.clear_light_btn)
        light_header_layout.addStretch()
        
        # 光照强度曲线容器
        self.light_curve_container = QWidget()
        self.light_curve_container.setObjectName("CurvePlaceholder")
        self.light_curve_layout = QVBoxLayout(self.light_curve_container)
        self.light_curve_layout.setContentsMargins(0, 0, 0, 0)
        
        # 初始占位符
        self.light_curve_placeholder = QLabel("光照强度曲线将显示在这里")
        self.light_curve_placeholder.setAlignment(Qt.AlignCenter)
        self.light_curve_layout.addWidget(self.light_curve_placeholder)
        
        light_layout.addLayout(light_header_layout)
        light_layout.addWidget(self.light_curve_container)

        # 车流量导入
        traffic_frame = QFrame()
        traffic_frame.setObjectName("trafficFrame")
        traffic_layout = QVBoxLayout(traffic_frame)
        traffic_header_layout = QHBoxLayout()
        traffic_label = QLabel("车流量:")
        self.import_traffic_btn = QPushButton("导入")
        self.clear_traffic_btn = QPushButton("清除")
        traffic_header_layout.addWidget(traffic_label)
        traffic_header_layout.addWidget(self.import_traffic_btn)
        traffic_header_layout.addWidget(self.clear_traffic_btn)
        traffic_header_layout.addStretch()
        
        # 车流量曲线容器
        self.traffic_curve_container = QWidget()
        self.traffic_curve_container.setObjectName("CurvePlaceholder")
        self.traffic_curve_layout = QVBoxLayout(self.traffic_curve_container)
        self.traffic_curve_layout.setContentsMargins(0, 0, 0, 0)
        
        # 初始占位符
        self.traffic_curve_placeholder = QLabel("车流量曲线将显示在这里")
        self.traffic_curve_placeholder.setAlignment(Qt.AlignCenter)
        self.traffic_curve_layout.addWidget(self.traffic_curve_placeholder)
        
        traffic_layout.addLayout(traffic_header_layout)
        traffic_layout.addWidget(self.traffic_curve_container)

        # 主要负荷导入
        load_frame = QFrame()
        load_frame.setObjectName("loadFrame")
        load_layout = QVBoxLayout(load_frame)
        load_header_layout = QHBoxLayout()
        load_label = QLabel("主要负荷:")
        self.import_load_btn = QPushButton("导入")
        self.clear_load_btn = QPushButton("清除")
        load_header_layout.addWidget(load_label)
        load_header_layout.addWidget(self.import_load_btn)
        load_header_layout.addWidget(self.clear_load_btn)
        load_header_layout.addStretch()
        
        # 主要负荷曲线容器
        self.load_curve_container = QWidget()
        self.load_curve_container.setObjectName("CurvePlaceholder")
        self.load_curve_layout = QVBoxLayout(self.load_curve_container)
        self.load_curve_layout.setContentsMargins(0, 0, 0, 0)
        
        # 初始占位符
        self.load_curve_placeholder = QLabel("主要负荷曲线将显示在这里")
        self.load_curve_placeholder.setAlignment(Qt.AlignCenter)
        self.load_curve_layout.addWidget(self.load_curve_placeholder)
        
        load_layout.addLayout(load_header_layout)
        load_layout.addWidget(self.load_curve_container)

        # 添加到父布局中，使用垂直布局并设置相同的伸缩因子
        parent_layout.addWidget(light_frame, 1)  # 伸缩因子为1
        parent_layout.addWidget(traffic_frame, 1)  # 伸缩因子为1
        parent_layout.addWidget(load_frame, 1)  # 伸缩因子为1

        group.setLayout(group_layout)

    def create_optimization_config_tab(self, tabs):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(24)
        left_widget = self.create_optimization_left_widget()
        right_widget = self.create_optimization_right_widget()
        layout.addWidget(left_widget, 1)
        layout.addWidget(right_widget, 2)
        tabs.addTab(tab, "优化配置")

    def create_optimization_left_widget(self):
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(16)
        optimization_metrics_group = self.create_optimization_metrics_group()
        left_layout.addWidget(optimization_metrics_group)
        result_group = self.create_optimization_result_group()
        left_layout.addWidget(result_group)
        return left_widget

    def create_optimization_metrics_group(self):
        group = QGroupBox("优化指标")
        metrics_layout = QGridLayout()
        metrics_layout.setSpacing(12)
        metrics_layout.setColumnStretch(0, 1)
        metrics_layout.setColumnStretch(1, 1)
        self.metrics_checkboxes = {
            "经济型": QCheckBox("经济型"),
            "可靠性": QCheckBox("可靠性"),
            "碳排放效益": QCheckBox("碳排放效益"),
            "消纳率": QCheckBox("消纳率"),
            "能效": QCheckBox("能效")
        }
        for i, (name, checkbox) in enumerate(self.metrics_checkboxes.items()):
            row = i // 2
            col = i % 2
            metrics_layout.addWidget(checkbox, row, col)
        metrics_layout.setContentsMargins(12, 20, 12, 12)
        group.setLayout(metrics_layout)
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        group.setMinimumWidth(300)
        group.setMinimumHeight(120)
        return group

    def create_optimization_result_group(self):
        group = QGroupBox("优化结果")
        result_layout = QVBoxLayout(group)
        result_layout.setContentsMargins(12, 20, 12, 12)
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["指标名称", "分数（满分10.0）"])
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        result_layout.addWidget(self.result_table)

        #增加方案选择下拉框
        self.scheme_selector = QComboBox()
        self.scheme_selector.addItem("请选择方案")
        self.scheme_selector.currentIndexChanged.connect(self.on_scheme_selected)
        result_layout.addWidget(self.scheme_selector)
        return group

    def create_optimization_right_widget(self):
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        curve_group = self.create_optimization_curve_group()
        right_layout.addWidget(curve_group)
        btn_layout = self.create_optimization_button_layout()
        right_layout.addLayout(btn_layout)
        return right_widget

    def create_optimization_curve_group(self):
        group = QGroupBox("优化曲线")
        curve_layout = QVBoxLayout(group)
        self.curve_placeholder = QLabel("优化曲线示意图\n（点击任意位置查看分数）")
        self.curve_placeholder.setStyleSheet("""
            border: 1px dashed #dcdfe6;
            border-radius: 4px;
            background-color: #fafafa;
            min-height: 250px;
            text-align: center;
            padding: 20px;
        """)
        self.curve_placeholder.setMouseTracking(True)
        #self.curve_placeholder.mousePressEvent = self.handle_curve_click
        curve_layout.addWidget(self.curve_placeholder)
        return group

    def create_optimization_button_layout(self):
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(16)
        self.optimize_btn = QPushButton("开始优化")
        self.optimize_btn.setMinimumHeight(40)
        self.optimize_btn.clicked.connect(self.start_optimization)
        self.detail_btn = QPushButton("查看详细配置")
        self.detail_btn.setMinimumHeight(40)
        self.detail_btn.clicked.connect(self.show_detail_config)
        self.confirm_btn = QPushButton("确定方案")
        self.confirm_btn.setMinimumHeight(40)
        self.confirm_btn.clicked.connect(self.confirm_scheme)
        btn_layout.addWidget(self.optimize_btn)
        btn_layout.addWidget(self.detail_btn)
        btn_layout.addWidget(self.confirm_btn)
        return btn_layout

    def on_scheme_selected(self, index):
        if index <= 0:
            return  # “请选择方案”或者非法下标，不更新

        scheme_index = index - 1  # 下拉第1项是方案1 → index=1 → 实际数组下标是0
        if not hasattr(self, "optimization_results") or scheme_index >= len(self.optimization_results):
            print(f"无效的方案索引: {scheme_index}")
            return

        # 拿到第 scheme_index 个方案的优化值（比如 [total_cost, lpsp, 1 - pv_util]）
        result = self.optimization_results[scheme_index]

        # 清空旧表格
        self.result_table.setRowCount(0)

        # 展示新的指标值（这里你可以自定义打分逻辑）
        metrics = ["年化成本", "负荷缺电率", "光伏未利用率"]
        for i, (name, value) in enumerate(zip(metrics, result)):
            self.result_table.insertRow(i)
            self.result_table.setItem(i, 0, QTableWidgetItem(name))
            score = self.evaluate_score(name, value)
            self.result_table.setItem(i, 1, QTableWidgetItem(f"{score:.2f}"))

    def evaluate_score(self, name, value):
        """根据具体优化指标给出评分（10分满分）"""
        if name == "年化成本":
            return max(0, 10 - value / 5000000)  # 预算越低越好
        elif name == "负荷缺电率":
            return max(0, 10 - value * 200)
        elif name == "光伏未利用率":
            return max(0, 10 - value * 10)
        else:
            return 0


    def create_energy_management_tab(self, tabs):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(24)
        settings_group = self.create_energy_management_settings_group()
        layout.addWidget(settings_group)
        result_group = self.create_energy_management_result_group()
        layout.addWidget(result_group)
        tabs.addTab(tab, "能量管理")

    def create_energy_management_settings_group(self):
        group = QGroupBox("能量管理目标占比")  # 修改标题
        settings_layout = QVBoxLayout()
        
        # 模式选择 - 修改为水平布局，与标题并排
        mode_layout = QHBoxLayout()
        mode_label = QLabel("选择能量管理模式:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["离网模式", "并网模式"])
        self.mode_combo.currentIndexChanged.connect(self.update_settings_form)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        settings_layout.addLayout(mode_layout)
        
        # 设置表单堆叠
        self.settings_stacked = QStackedWidget()
        
        # 离网模式设置
        self.offgrid_settings = QWidget()
        offgrid_layout = QVBoxLayout(self.offgrid_settings)
        
        # 使用水平布局排列三个指标
        offgrid_h_layout = QHBoxLayout()
        offgrid_h_layout.setSpacing(20)
        
        # 光伏消纳率
        pv_layout = QVBoxLayout()
        pv_label = QLabel("光伏消纳率")  
        pv_h_layout = QHBoxLayout()
        self.photovoltaic_absorption_spin = QDoubleSpinBox()
        self.photovoltaic_absorption_spin.setRange(0, 100)
        self.photovoltaic_absorption_spin.setValue(34)
        pv_h_layout.addWidget(self.photovoltaic_absorption_spin)
        pv_h_layout.addWidget(QLabel("%")) 
        pv_layout.addWidget(pv_label)
        pv_layout.addLayout(pv_h_layout)
        offgrid_h_layout.addLayout(pv_layout)
        
        # 电负荷满足率
        load_layout = QVBoxLayout()
        load_label = QLabel("电负荷满足率")  
        load_h_layout = QHBoxLayout()
        self.load_satisfaction_spin = QDoubleSpinBox()
        self.load_satisfaction_spin.setRange(0, 100)
        self.load_satisfaction_spin.setValue(33)
        load_h_layout.addWidget(self.load_satisfaction_spin)
        load_h_layout.addWidget(QLabel("%")) 
        load_layout.addWidget(load_label)
        load_layout.addLayout(load_h_layout)
        offgrid_h_layout.addLayout(load_layout)
        
        # 储能损耗
        storage_layout = QVBoxLayout()
        storage_label = QLabel("储能损耗") 
        storage_h_layout = QHBoxLayout()
        self.storage_loss_offgrid_spin = QDoubleSpinBox()
        self.storage_loss_offgrid_spin.setRange(0, 100)
        self.storage_loss_offgrid_spin.setValue(33)
        storage_h_layout.addWidget(self.storage_loss_offgrid_spin)
        storage_h_layout.addWidget(QLabel("%")) 
        storage_layout.addWidget(storage_label)
        storage_layout.addLayout(storage_h_layout)
        offgrid_h_layout.addLayout(storage_layout)
        
        offgrid_layout.addLayout(offgrid_h_layout)
        self.settings_stacked.addWidget(self.offgrid_settings)
        
        # 并网模式设置
        self.grid_settings = QWidget()
        grid_layout = QVBoxLayout(self.grid_settings)
        
        # 使用水平布局排列三个指标
        grid_h_layout = QHBoxLayout()
        grid_h_layout.setSpacing(30)
        
        # 碳排放量
        carbon_layout = QVBoxLayout()
        carbon_label = QLabel("碳排放量")  
        carbon_h_layout = QHBoxLayout()
        self.carbon_emission_spin = QDoubleSpinBox()
        self.carbon_emission_spin.setRange(0, 100)
        self.carbon_emission_spin.setValue(34)
        carbon_h_layout.addWidget(self.carbon_emission_spin)
        carbon_h_layout.addWidget(QLabel("%")) 
        carbon_layout.addWidget(carbon_label)
        carbon_layout.addLayout(carbon_h_layout)
        grid_h_layout.addLayout(carbon_layout)
        
        # 经济指标
        economic_layout = QVBoxLayout()
        economic_label = QLabel("经济指标")  
        economic_h_layout = QHBoxLayout()
        self.economic_indicator_spin = QDoubleSpinBox()
        self.economic_indicator_spin.setRange(0, 100)
        self.economic_indicator_spin.setValue(33)
        economic_h_layout.addWidget(self.economic_indicator_spin)
        economic_h_layout.addWidget(QLabel("%"))  
        economic_layout.addWidget(economic_label)
        economic_layout.addLayout(economic_h_layout)
        grid_h_layout.addLayout(economic_layout)
        
        # 储能损耗
        storage_layout = QVBoxLayout()
        storage_label = QLabel("储能损耗")
        storage_h_layout = QHBoxLayout()
        self.storage_loss_grid_spin = QDoubleSpinBox()
        self.storage_loss_grid_spin.setRange(0, 100)
        self.storage_loss_grid_spin.setValue(33)
        storage_h_layout.addWidget(self.storage_loss_grid_spin)
        storage_h_layout.addWidget(QLabel("%"))  
        storage_layout.addWidget(storage_label)
        storage_layout.addLayout(storage_h_layout)
        grid_h_layout.addLayout(storage_layout)
        
        grid_layout.addLayout(grid_h_layout)
        self.settings_stacked.addWidget(self.grid_settings)
        
        settings_layout.addWidget(self.settings_stacked)
        
        # 按钮布局
        btn_layout = QHBoxLayout()
        self.set_params_btn = QPushButton("设置参数")
        self.set_params_btn.setMinimumHeight(40)
        self.set_params_btn.clicked.connect(self.set_energy_management_params)
        self.clear_params_btn = QPushButton("清空参数")
        self.clear_params_btn.setMinimumHeight(40)
        self.clear_params_btn.clicked.connect(self.clear_energy_management_params)
        self.clear_params_btn.setEnabled(False)
        self.start_management_btn = QPushButton("开始能量管理")
        self.start_management_btn.setMinimumHeight(40)
        self.start_management_btn.clicked.connect(self.start_energy_management)
        self.start_management_btn.setEnabled(False)
        btn_layout.addWidget(self.set_params_btn)
        btn_layout.addWidget(self.clear_params_btn)
        btn_layout.addWidget(self.start_management_btn)
        
        settings_layout.addLayout(btn_layout)
        group.setLayout(settings_layout)
        return group

    def create_energy_management_result_group(self):
        group = QGroupBox("能量管理结果")
        result_layout = QVBoxLayout()
        
        # 结果选项卡
        self.result_tabs = QTabWidget()
        
        # 离网模式结果选项卡
        self.offgrid_result_tabs = QTabWidget()
        # 用图片替换原有占位符
        self.offgrid_result_tabs.addTab(self.create_result_image("fig_radar_scores.png", "雷达图"), "雷达图")
        self.offgrid_result_tabs.addTab(self.create_result_image("fig_pv_vs_curtailment.png", "光伏发电与弃光功率图"), "光伏发电与弃光功率图")
        self.offgrid_result_tabs.addTab(self.create_result_image("fig_load_vs_loss.png", "负载满足与缺电功率"), "负载满足与缺电功率")
        self.offgrid_result_tabs.addTab(self.create_result_image("fig_dc_ac_exchange.png", "交直流能量交换曲线"), "交直流能量交换曲线")
        self.offgrid_result_tabs.addTab(self.create_result_image("fig_battery_power.png", "储能功率曲线"), "储能功率曲线")
        self.offgrid_result_tabs.addTab(self.create_result_image("fig_soc_status.png", "储能SOC状态"), "储能SOC状态")
        
        # 并网模式结果选项卡（保持原样）
        self.grid_result_tabs = QTabWidget()
        self.grid_result_tabs.addTab(self.create_result_placeholder("雷达图"), "雷达图")
        self.grid_result_tabs.addTab(self.create_result_placeholder("直流侧能量去向流动图"), "直流侧能量去向流动图")
        self.grid_result_tabs.addTab(self.create_result_placeholder("直流侧能量来源流动图"), "直流侧能量来源流动图")
        self.grid_result_tabs.addTab(self.create_result_placeholder("交流侧电能去向组成图"), "交流侧电能去向组成图")
        self.grid_result_tabs.addTab(self.create_result_placeholder("交流侧能量来源流动图"), "交流侧能量来源流动图")
        self.grid_result_tabs.addTab(self.create_result_placeholder("每小时购电成本与售电收益图"), "每小时购电成本与售电收益图")
        
        # 结果堆叠
        self.result_stacked = QStackedWidget()
        self.result_stacked.addWidget(self.offgrid_result_tabs)
        self.result_stacked.addWidget(self.grid_result_tabs)
        
        result_layout.addWidget(self.result_stacked)
        group.setLayout(result_layout)
        return group

    def create_result_image(self, image_path, alt_text):
        """创建图片显示控件，如果图片不存在则显示占位文本"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            label.setPixmap(pixmap.scaled(700, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            label.setText(f"{alt_text}图片未找到\n请先运行liwang.py生成图片")
            label.setStyleSheet("color: #888; font-size: 16px;")
        layout.addWidget(label)
        return widget
    
    def create_result_placeholder(self, title):
        """创建结果占位符"""
        placeholder = QLabel(f"{title}将显示在这里")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("""
            background-color: #f5f7fa; 
            border-radius: 4px; 
            padding: 20px;
            min-height: 300px;
        """)
        return placeholder

    def create_result_view_tab(self, tabs):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group = QGroupBox("结果查看")
        group_layout = QVBoxLayout(group)
        view_btn = QPushButton("查看结果")
        view_btn.clicked.connect(self.view_results)
        group_layout.addWidget(view_btn)
        self.results_display = QLabel("结果将显示在这里")
        self.results_display.setAlignment(Qt.AlignCenter)
        self.results_display.setStyleSheet("padding: 20px; background-color: #f5f7fa; border-radius: 4px; min-height: 200px;")
        group_layout.addWidget(self.results_display)
        layout.addWidget(group)
        tabs.addTab(tab, "结果查看")

    def update_settings_form(self):
        """更新设置表单显示"""
        if self.mode_combo.currentIndex() == 0:  # 离网模式
            self.settings_stacked.setCurrentIndex(0)
            self.result_stacked.setCurrentIndex(0)
        else:  # 并网模式
            self.settings_stacked.setCurrentIndex(1)
            self.result_stacked.setCurrentIndex(1)

    def import_light_data(self):
        """单独处理光照数据导入，避免lambda导致的信号重复"""
        self.import_data("light")

    def import_traffic_data(self):
        """单独处理车流量数据导入"""
        self.import_data("traffic")

    def import_load_data(self):
        """单独处理负荷数据导入"""
        self.import_data("load")
# ===========import_data 逻辑修改了一下===============
    def import_data(self, data_type):
        """数据导入核心逻辑"""
        print(f"Importing {data_type} data")  # 调试输出，确认函数调用次数
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"选择{data_type}数据文件", "", "数据文件 (*.csv *.xlsx)"
        )
        if file_path:
            try:
                # 读取Excel文件
                df = pd.read_excel(file_path)
                # 创建图表
                fig = Figure(facecolor='none')  # 透明背景                
                # 根据数据类型设置图表标题和坐标轴
                if data_type == "light":
                    # light的Excel采集两列：时间和对应的数据
                    time = df.iloc[:24, 0].values.astype(int)
                    data = df.iloc[:24, 7].values.astype(float)
                    ax = fig.add_subplot(111)
                    ax.plot(time, data, 'b-')
                    fcn1.solar_irradiance = data
                    ax.set_title('光照强度曲线')
                    ax.set_xlabel('时间')
                    ax.set_ylabel('光照强度')
                    container = self.light_curve_container
                    placeholder = self.light_curve_placeholder
                elif data_type == "traffic":
                    # traffic的Excel采集两列：时间和对应的数据
                    time = df.iloc[:, 0].values.astype(float)
                    data = df.iloc[:, 1].values.astype(float)
                    ax = fig.add_subplot(111)
                    ax.plot(time, data, 'b-')
                    fcn1.dc_charge_load_profile_raw = data
                    ax.set_title('车流量曲线')
                    ax.set_xlabel('时间')
                    ax.set_ylabel('车流量')
                    container = self.traffic_curve_container
                    placeholder = self.traffic_curve_placeholder
                elif data_type == "load":
                    # load的Excel采集两列：时间和对应的数据
                    time = df.iloc[:24, 0].values.astype(int)
                    data = df.iloc[:24, 7].values.astype(float)
                    ax = fig.add_subplot(111)
                    ax.plot(time, data, 'b-')
                    # fcn1.solar_irradiance = data   现在的负荷用的是random，后面再说
                    ax.set_title('主要负荷曲线')
                    ax.set_xlabel('时间')
                    ax.set_ylabel('主要负荷')
                    container = self.load_curve_container
                    placeholder = self.load_curve_placeholder

                ax.grid(True)

                # 移除占位符
                if placeholder.parent() is not None:
                    placeholder.setParent(None)

                # 创建FigureCanvas并添加到容器
                canvas = FigureCanvas(fig)
                container.layout().addWidget(canvas)

                # 保存数据引用
                if data_type == "light":
                    self.light_data = df
                elif data_type == "traffic":
                    self.traffic_data = df
                elif data_type == "load":
                    self.load_data = df

                self.show_status(f"{data_type}数据导入成功")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"数据导入失败: {str(e)}")

    def clear_data(self, data_type):
        """清除导入的数据"""
        if data_type == "light":
            container = self.light_curve_container
            placeholder = self.light_curve_placeholder
            self.light_data = None
        elif data_type == "traffic":
            container = self.traffic_curve_container
            placeholder = self.traffic_curve_placeholder
            self.traffic_data = None
        elif data_type == "load":
            container = self.load_curve_container
            placeholder = self.load_curve_placeholder
            self.load_data = None

        # 清除容器中的所有控件
        layout = container.layout()
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        # 重新添加占位符
        layout.addWidget(placeholder)

        self.show_status(f"{data_type}数据已清除")
# ===========import_from_background 逻辑修改了一下===============
    def import_from_background(self):
        """从背景信息导入数据"""
        if not self.area_edit.text() or not self.vehicle_count_edit.text():
            QMessageBox.warning(self, "警告", "请先填写背景信息")
            return

        # 模拟从背景信息计算方案预期参数
        try:
            area = float(self.area_edit.text())
            vehicle_count = int(self.vehicle_count_edit.text())
            fcn1.park_space = area
            fcn1.car_number = vehicle_count
            # 基于简单规则计算预期参数
            charger_dc_min = max(10, min(100, int(vehicle_count * 0.2)))
            charger_dc_max = max(charger_dc_min, int(vehicle_count * 0.4))
            charger_ac_min = max(10, min(100, int(vehicle_count * 0.3)))
            charger_ac_max = max(charger_ac_min, int(vehicle_count * 0.5))

            # 光伏容量基于面积
            solar_dc_min = max(0.1, min(5.0, area / 5000))
            solar_dc_max = max(solar_dc_min, min(8.0, area / 4000))
            solar_ac_min = solar_dc_min * 0.8
            solar_ac_max = solar_dc_max * 0.8

            # 储能容量基于光伏容量
            storage_dc_min = max(0.1, min(3.0, solar_dc_min * 0.6))
            storage_dc_max = max(storage_dc_min, min(4.0, solar_dc_max * 0.5))
            storage_ac_min = storage_dc_min * 0.8
            storage_ac_max = storage_dc_max * 0.8

            # 更新UI
            self.charger_dc_min_edit.setValue(charger_dc_min)
            self.charger_dc_max_edit.setValue(charger_dc_max)
            self.charger_ac_min_edit.setValue(charger_ac_min)
            self.charger_ac_max_edit.setValue(charger_ac_max)
            self.solar_dc_min_edit.setValue(solar_dc_min)
            self.solar_dc_max_edit.setValue(solar_dc_max)
            self.solar_ac_min_edit.setValue(solar_ac_min)
            self.solar_ac_max_edit.setValue(solar_ac_max)
            self.storage_dc_min_edit.setValue(storage_dc_min)
            self.storage_dc_max_edit.setValue(storage_dc_max)
            self.storage_ac_min_edit.setValue(storage_ac_min)
            self.storage_ac_max_edit.setValue(storage_ac_max)

            self.background_imported = True
            self.show_status("已从背景信息导入方案预期参数")
        except ValueError:
            QMessageBox.critical(self, "错误", "背景信息格式不正确，请检查输入")

    def update_progress(self):
        """更新进度条"""
        count = 0
        
        # 检查背景信息
        if self.area_edit.text():
            count += 1
        if self.vehicle_count_edit.text():
            count += 1
        
        # 检查方案预期
        if self.charger_dc_min_edit.value() > 0 and self.charger_dc_max_edit.value() > 0:
            count += 1
        if self.charger_ac_min_edit.value() > 0 and self.charger_ac_max_edit.value() > 0:
            count += 1
        if self.solar_dc_min_edit.value() > 0 and self.solar_dc_max_edit.value() > 0:
            count += 1
        if self.solar_ac_min_edit.value() > 0 and self.solar_ac_max_edit.value() > 0:
            count += 1
        if self.storage_dc_min_edit.value() > 0 and self.storage_dc_max_edit.value() > 0:
            count += 1
        if self.storage_ac_min_edit.value() > 0 and self.storage_ac_max_edit.value() > 0:
            count += 1
        
        # 检查数据导入
        if self.light_data is not None:
            count += 1
        if self.traffic_data is not None:
            count += 1
        if self.load_data is not None:
            count += 1
        
        self.progress_value = count
        self.progress_bar.setValue(count)
        
        # 当所有数据都填写后，启用下一步按钮
        self.next_step_btn.setEnabled(count == self.progress_max)

# ===========next_step 传递了参数过去===============
    def next_step(self):
        """进入下一步"""
        if self.progress_value == self.progress_max:
            self.tabs.setCurrentIndex(1)  # 切换到优化配置标签页
            self.show_status("进入优化配置")
            # 构建 6 个决策变量的下限 xl_user 和上限 xu_user
            xl_user = np.array([
                self.solar_ac_min_edit.value(),
                self.solar_dc_min_edit.value(),
                self.storage_ac_min_edit.value(),
                self.storage_dc_min_edit.value(),
                self.charger_ac_min_edit.value(),
                self.charger_dc_min_edit.value()
            ])

            xu_user = np.array([
                self.solar_ac_max_edit.value(),
                self.solar_dc_max_edit.value(),
                self.storage_ac_max_edit.value(),
                self.storage_dc_max_edit.value(),
                self.charger_ac_max_edit.value(),
                self.charger_dc_max_edit.value()
            ])            
            fcn1.xl_user = xl_user
            fcn1.xu_user = xu_user
            fcn1.preprocess_inputs(5000)
        else:
            QMessageBox.warning(self, "警告", "请完成所有必填项")
# ===========start_optimization 调用了相关函数===============
    def start_optimization(self):
        """开始优化"""
        # 检查是否有选中的指标
        selected_metrics = [name for name, checkbox in self.metrics_checkboxes.items() if checkbox.isChecked()]
        if not selected_metrics:
            QMessageBox.warning(self, "警告", "请至少选择一个优化指标")
            return
        
        # 模拟优化过程
        self.show_status("正在进行优化计算...")
        fig, F = fcn1.run_OG_optimization()
        # 保存到 self.optimization_results
        self.optimization_results = F

        # 填充下拉框
        self.scheme_selector.clear()
        self.scheme_selector.addItem("请选择方案")
        for i in range(len(F)):
            self.scheme_selector.addItem(f"方案 {i+1}")
        # 随机生成一些优化结果
        metrics = list(self.metrics_checkboxes.keys())
        scores = {metric: round(random.uniform(6.0, 9.5), 1) for metric in metrics}
        
        # 更新结果表格
        self.result_table.setRowCount(len(metrics))
        for i, metric in enumerate(metrics):
            self.result_table.setItem(i, 0, QTableWidgetItem(metric))
            self.result_table.setItem(i, 1, QTableWidgetItem(str(scores[metric])))
        
        self.result_table.resizeColumnsToContents()
        
        # 更新曲线占位符
        self.curve_placeholder.setText("优化曲线示意图\n（点击任意位置查看分数）")
        
        self.show_status("优化计算完成")

    def handle_curve_click(self, event):
        """处理曲线点击事件"""
        # 检查是否有优化结果
        if self.result_table.rowCount() == 0:
            QMessageBox.information(self, "提示", "请先进行优化计算")
            return

        # 显示优化结果对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("优化结果详情")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)
        
        # 添加结果表格
        result_table = QTableWidget()
        result_table.setColumnCount(2)
        result_table.setHorizontalHeaderLabels(["指标名称", "分数（满分10.0）"])
        
        # 复制数据
        result_table.setRowCount(self.result_table.rowCount())
        for i in range(self.result_table.rowCount()):
            result_table.setItem(i, 0, QTableWidgetItem(self.result_table.item(i, 0).text()))
            result_table.setItem(i, 1, QTableWidgetItem(self.result_table.item(i, 1).text()))
        
        result_table.resizeColumnsToContents()
        layout.addWidget(result_table)
        
        # 添加按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box)
        
        dialog.exec_()
# ===========show_detail_config 调用了相关函数===============
    def show_detail_config(self):
        """显示详细配置"""
        QMessageBox.information(self, "详细配置", "这里可以显示详细的系统配置信息")       
        fcn1.show_OG_selected_solution()
# ===========confirm_scheme 调用了相关函数===============
    def confirm_scheme(self):
        """确定方案"""
        reply = QMessageBox.question(self, "确认", "确定使用当前方案吗？", 
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            fcn1.OG_flag=1
            self.show_status("方案已确认")
            self.tabs.setCurrentIndex(2)  # 切换到能量管理标签页
        else :
            fcn1.OG_flag=0
            fcn1.show_OG_selected_solution()

    def set_energy_management_params(self):
        """设置能量管理参数"""
        if self.mode_combo.currentIndex() == 0:  # 离网模式
            params = {
                "mode": "offgrid",
                "photovoltaic_absorption": self.photovoltaic_absorption_spin.value(),
                "load_satisfaction": self.load_satisfaction_spin.value(),
                "storage_loss": self.storage_loss_offgrid_spin.value()
            }
        else:  # 并网模式
            params = {
                "mode": "grid",
                "carbon_emission": self.carbon_emission_spin.value(),
                "economic_indicator": self.economic_indicator_spin.value(),
                "storage_loss": self.storage_loss_grid_spin.value()
            }
        
        self.energy_management_settings = params
        self.clear_params_btn.setEnabled(True)
        self.start_management_btn.setEnabled(True)
        self.show_status("能量管理参数已设置")

    def clear_energy_management_params(self):
        """清空能量管理参数"""
        self.energy_management_settings = None
        self.clear_params_btn.setEnabled(False)
        self.start_management_btn.setEnabled(False)
        
        # 重置UI
        if self.mode_combo.currentIndex() == 0:  # 离网模式
            self.photovoltaic_absorption_spin.setValue(34)
            self.load_satisfaction_spin.setValue(33)
            self.storage_loss_offgrid_spin.setValue(33)
        else:  # 并网模式
            self.carbon_emission_spin.setValue(34)
            self.economic_indicator_spin.setValue(33)
            self.storage_loss_grid_spin.setValue(33)
        
        self.show_status("能量管理参数已清空")

    def start_energy_management(self):
        """开始能量管理"""
        if not self.energy_management_settings:
            QMessageBox.warning(self, "警告", "请先设置能量管理参数")
            return
        
        # 显示处理中消息
        self.show_status("正在进行能量管理计算...")
        
        # 模拟计算过程
        import time
        time.sleep(1)  # 模拟计算时间
        
        # 更新结果显示
        self.show_status("能量管理计算完成")
        self.tabs.setCurrentIndex(3)  # 切换到结果查看标签页

    def view_results(self):
        """查看结果"""
        if not self.energy_management_settings:
            QMessageBox.warning(self, "警告", "请先完成能量管理计算")
            return
        
        # 显示结果
        result_text = "能量管理结果:\n\n"
        
        if self.energy_management_settings["mode"] == "offgrid":
            result_text += f"光伏消纳率: {self.energy_management_settings['photovoltaic_absorption']}%\n"
            result_text += f"电负荷满足率: {self.energy_management_settings['load_satisfaction']}%\n"
            result_text += f"储能损耗: {self.energy_management_settings['storage_loss']}%\n"
        else:
            result_text += f"碳排放量: {self.energy_management_settings['carbon_emission']}%\n"
            result_text += f"经济指标: {self.energy_management_settings['economic_indicator']}%\n"
            result_text += f"储能损耗: {self.energy_management_settings['storage_loss']}%\n"
        
        # 随机生成一些额外结果
        result_text += "\n额外指标:\n"
        result_text += f"系统效率: {round(random.uniform(75, 95), 1)}%\n"
        result_text += f"能量平衡率: {round(random.uniform(90, 99), 1)}%\n"
        result_text += f"系统可靠性: {round(random.uniform(85, 98), 1)}%\n"
        
        self.results_display.setText(result_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EnergyOptimizationPlatform()
    window.show()
    sys.exit(app.exec_())