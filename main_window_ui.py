# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/main.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 720)
        MainWindow.setMinimumSize(QtCore.QSize(1280, 720))
        MainWindow.setMaximumSize(QtCore.QSize(1280, 720))
        font = QtGui.QFont()
        font.setPointSize(16)
        MainWindow.setFont(font)
        MainWindow.setStyleSheet("background-color: #F2EFE7;")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.name1 = QtWidgets.QLabel(self.centralwidget)
        self.name1.setEnabled(True)
        self.name1.setGeometry(QtCore.QRect(20, 0, 361, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.name1.setFont(font)
        self.name1.setStyleSheet("color: #2973B2;")
        self.name1.setObjectName("name1")
        self.name2 = QtWidgets.QLabel(self.centralwidget)
        self.name2.setEnabled(True)
        self.name2.setGeometry(QtCore.QRect(30, 60, 331, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.name2.setFont(font)
        self.name2.setStyleSheet("color: #2973B2;")
        self.name2.setObjectName("name2")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(520, 0, 20, 661))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(10, 160, 511, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.bt_start = QtWidgets.QPushButton(self.centralwidget)
        self.bt_start.setGeometry(QtCore.QRect(400, 0, 111, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.bt_start.setFont(font)
        self.bt_start.setStyleSheet("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("ui\\resources/diskette.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bt_start.setIcon(icon)
        self.bt_start.setObjectName("bt_start")
        self.bt_load = QtWidgets.QPushButton(self.centralwidget)
        self.bt_load.setGeometry(QtCore.QRect(400, 60, 111, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.bt_load.setFont(font)
        self.bt_load.setStyleSheet("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("ui\\resources/folder.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bt_load.setIcon(icon1)
        self.bt_load.setObjectName("bt_load")
        self.folder_name = QtWidgets.QLabel(self.centralwidget)
        self.folder_name.setGeometry(QtCore.QRect(10, 120, 441, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.folder_name.setFont(font)
        self.folder_name.setFrameShape(QtWidgets.QFrame.Box)
        self.folder_name.setObjectName("folder_name")
        self.bt_load_2 = QtWidgets.QPushButton(self.centralwidget)
        self.bt_load_2.setGeometry(QtCore.QRect(470, 120, 41, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.bt_load_2.setFont(font)
        self.bt_load_2.setStyleSheet("\n"
"")
        self.bt_load_2.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("ui\\resources/reset.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bt_load_2.setIcon(icon2)
        self.bt_load_2.setObjectName("bt_load_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 180, 501, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(130, 260, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(390, 260, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(260, 260, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setFrameShape(QtWidgets.QFrame.Box)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 300, 501, 131))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.gridLayoutWidget.setFont(font)
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("")
        self.label_6.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 1, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setStyleSheet("")
        self.label_9.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 2, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("")
        self.label_5.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 0, 0, 1, 1)
        self.cppg_snr = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.cppg_snr.setFont(font)
        self.cppg_snr.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.cppg_snr.setAlignment(QtCore.Qt.AlignCenter)
        self.cppg_snr.setObjectName("cppg_snr")
        self.gridLayout.addWidget(self.cppg_snr, 1, 1, 1, 1)
        self.ippg_snr = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.ippg_snr.setFont(font)
        self.ippg_snr.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.ippg_snr.setAlignment(QtCore.Qt.AlignCenter)
        self.ippg_snr.setObjectName("ippg_snr")
        self.gridLayout.addWidget(self.ippg_snr, 0, 1, 1, 1)
        self.spg_snr = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.spg_snr.setFont(font)
        self.spg_snr.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.spg_snr.setAlignment(QtCore.Qt.AlignCenter)
        self.spg_snr.setObjectName("spg_snr")
        self.gridLayout.addWidget(self.spg_snr, 2, 1, 1, 1)
        self.ippg_hr = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.ippg_hr.setFont(font)
        self.ippg_hr.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.ippg_hr.setAlignment(QtCore.Qt.AlignCenter)
        self.ippg_hr.setObjectName("ippg_hr")
        self.gridLayout.addWidget(self.ippg_hr, 0, 2, 1, 1)
        self.cppg_hr = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.cppg_hr.setFont(font)
        self.cppg_hr.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.cppg_hr.setAlignment(QtCore.Qt.AlignCenter)
        self.cppg_hr.setObjectName("cppg_hr")
        self.gridLayout.addWidget(self.cppg_hr, 1, 2, 1, 1)
        self.ippg_freq = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.ippg_freq.setFont(font)
        self.ippg_freq.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.ippg_freq.setAlignment(QtCore.Qt.AlignCenter)
        self.ippg_freq.setObjectName("ippg_freq")
        self.gridLayout.addWidget(self.ippg_freq, 0, 3, 1, 1)
        self.cppg_freq = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.cppg_freq.setFont(font)
        self.cppg_freq.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.cppg_freq.setAlignment(QtCore.Qt.AlignCenter)
        self.cppg_freq.setObjectName("cppg_freq")
        self.gridLayout.addWidget(self.cppg_freq, 1, 3, 1, 1)
        self.spg_hr = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.spg_hr.setFont(font)
        self.spg_hr.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.spg_hr.setAlignment(QtCore.Qt.AlignCenter)
        self.spg_hr.setObjectName("spg_hr")
        self.gridLayout.addWidget(self.spg_hr, 2, 2, 1, 1)
        self.spg_freq = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.spg_freq.setFont(font)
        self.spg_freq.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.spg_freq.setAlignment(QtCore.Qt.AlignCenter)
        self.spg_freq.setObjectName("spg_freq")
        self.gridLayout.addWidget(self.spg_freq, 2, 3, 1, 1)
        self.label_25 = QtWidgets.QLabel(self.centralwidget)
        self.label_25.setGeometry(QtCore.QRect(10, 260, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_25.setFont(font)
        self.label_25.setFrameShape(QtWidgets.QFrame.Box)
        self.label_25.setAlignment(QtCore.Qt.AlignCenter)
        self.label_25.setObjectName("label_25")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(10, 440, 511, 16))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.formLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(10, 500, 501, 170))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.label_14 = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_14.setFont(font)
        self.label_14.setStyleSheet("")
        self.label_14.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_14.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_14.setObjectName("label_14")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_14)
        self.label_18 = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_18.setFont(font)
        self.label_18.setStyleSheet("")
        self.label_18.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_18.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_18.setObjectName("label_18")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_18)
        self.label_19 = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_19.setFont(font)
        self.label_19.setStyleSheet("")
        self.label_19.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_19.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_19.setObjectName("label_19")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_19)
        self.frame_rate = QtWidgets.QComboBox(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.frame_rate.setFont(font)
        self.frame_rate.setObjectName("frame_rate")
        self.frame_rate.addItem("")
        self.frame_rate.addItem("")
        self.frame_rate.addItem("")
        self.frame_rate.addItem("")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.frame_rate)
        self.video_length = QtWidgets.QSpinBox(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.video_length.setFont(font)
        self.video_length.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.video_length.setMinimum(5)
        self.video_length.setMaximum(60)
        self.video_length.setObjectName("video_length")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.video_length)
        self.exposure_time = QtWidgets.QSpinBox(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.exposure_time.setFont(font)
        self.exposure_time.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.exposure_time.setMinimum(100)
        self.exposure_time.setMaximum(20000)
        self.exposure_time.setObjectName("exposure_time")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.exposure_time)
        self.label_20 = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_20.setFont(font)
        self.label_20.setStyleSheet("")
        self.label_20.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_20.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_20.setObjectName("label_20")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_20)
        self.label_24 = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_24.setFont(font)
        self.label_24.setStyleSheet("")
        self.label_24.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_24.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_24.setObjectName("label_24")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_24)
        self.size_ippg = QtWidgets.QComboBox(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.size_ippg.setFont(font)
        self.size_ippg.setObjectName("size_ippg")
        self.size_ippg.addItem("")
        self.size_ippg.addItem("")
        self.size_ippg.addItem("")
        self.size_ippg.addItem("")
        self.size_ippg.addItem("")
        self.size_ippg.addItem("")
        self.size_ippg.addItem("")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.size_ippg)
        self.size_spg = QtWidgets.QComboBox(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.size_spg.setFont(font)
        self.size_spg.setObjectName("size_spg")
        self.size_spg.addItem("")
        self.size_spg.addItem("")
        self.size_spg.addItem("")
        self.size_spg.addItem("")
        self.size_spg.addItem("")
        self.size_spg.addItem("")
        self.size_spg.addItem("")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.size_spg)
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(10, 460, 361, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_13.setFont(font)
        self.label_13.setFrameShape(QtWidgets.QFrame.Box)
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.personal_data = QtWidgets.QLabel(self.centralwidget)
        self.personal_data.setGeometry(QtCore.QRect(10, 220, 501, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.personal_data.setFont(font)
        self.personal_data.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.personal_data.setObjectName("personal_data")
        self.td = QtWidgets.QLabel(self.centralwidget)
        self.td.setGeometry(QtCore.QRect(540, 630, 531, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.td.setFont(font)
        self.td.setStyleSheet("color: #48A6A7;")
        self.td.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.td.setObjectName("td")
        self.apply_setting_camera = QtWidgets.QPushButton(self.centralwidget)
        self.apply_setting_camera.setGeometry(QtCore.QRect(380, 460, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.apply_setting_camera.setFont(font)
        self.apply_setting_camera.setStyleSheet("")
        self.apply_setting_camera.setObjectName("apply_setting_camera")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(1080, 630, 191, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.personal_data_2 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.personal_data_2.setFont(font)
        self.personal_data_2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.personal_data_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.personal_data_2.setObjectName("personal_data_2")
        self.horizontalLayout.addWidget(self.personal_data_2)
        self.port = QtWidgets.QLabel(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.port.setFont(font)
        self.port.setStyleSheet("color:green;")
        self.port.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.port.setAlignment(QtCore.Qt.AlignCenter)
        self.port.setObjectName("port")
        self.horizontalLayout.addWidget(self.port)
        self.result_group = QtWidgets.QGroupBox(self.centralwidget)
        self.result_group.setGeometry(QtCore.QRect(540, -10, 741, 631))
        self.result_group.setTitle("")
        self.result_group.setObjectName("result_group")
        self.freq_ippg = QtWidgets.QLabel(self.result_group)
        self.freq_ippg.setGeometry(QtCore.QRect(0, 220, 361, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.freq_ippg.setFont(font)
        self.freq_ippg.setFrameShape(QtWidgets.QFrame.Box)
        self.freq_ippg.setAlignment(QtCore.Qt.AlignCenter)
        self.freq_ippg.setObjectName("freq_ippg")
        self.freq_spg = QtWidgets.QLabel(self.result_group)
        self.freq_spg.setGeometry(QtCore.QRect(360, 220, 371, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.freq_spg.setFont(font)
        self.freq_spg.setFrameShape(QtWidgets.QFrame.Box)
        self.freq_spg.setAlignment(QtCore.Qt.AlignCenter)
        self.freq_spg.setObjectName("freq_spg")
        self.spg_cppg = QtWidgets.QLabel(self.result_group)
        self.spg_cppg.setGeometry(QtCore.QRect(360, 10, 371, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.spg_cppg.setFont(font)
        self.spg_cppg.setFrameShape(QtWidgets.QFrame.Box)
        self.spg_cppg.setAlignment(QtCore.Qt.AlignCenter)
        self.spg_cppg.setObjectName("spg_cppg")
        self.time_delay = QtWidgets.QLabel(self.result_group)
        self.time_delay.setGeometry(QtCore.QRect(0, 430, 731, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.time_delay.setFont(font)
        self.time_delay.setFrameShape(QtWidgets.QFrame.Box)
        self.time_delay.setAlignment(QtCore.Qt.AlignCenter)
        self.time_delay.setObjectName("time_delay")
        self.ippg_cppg = QtWidgets.QLabel(self.result_group)
        self.ippg_cppg.setGeometry(QtCore.QRect(0, 10, 361, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.ippg_cppg.setFont(font)
        self.ippg_cppg.setFrameShape(QtWidgets.QFrame.Box)
        self.ippg_cppg.setAlignment(QtCore.Qt.AlignCenter)
        self.ippg_cppg.setObjectName("ippg_cppg")
        self.ippg_cppg_plt = QtWidgets.QWidget(self.result_group)
        self.ippg_cppg_plt.setGeometry(QtCore.QRect(0, 50, 361, 161))
        self.ippg_cppg_plt.setObjectName("ippg_cppg_plt")
        self.spg_cppg_plt = QtWidgets.QWidget(self.result_group)
        self.spg_cppg_plt.setGeometry(QtCore.QRect(360, 50, 371, 161))
        self.spg_cppg_plt.setObjectName("spg_cppg_plt")
        self.freq_spg_plt = QtWidgets.QWidget(self.result_group)
        self.freq_spg_plt.setGeometry(QtCore.QRect(360, 260, 371, 161))
        self.freq_spg_plt.setObjectName("freq_spg_plt")
        self.freq_ippg_plt = QtWidgets.QWidget(self.result_group)
        self.freq_ippg_plt.setGeometry(QtCore.QRect(0, 260, 361, 161))
        self.freq_ippg_plt.setObjectName("freq_ippg_plt")
        self.time_delay_plt = QtWidgets.QWidget(self.result_group)
        self.time_delay_plt.setGeometry(QtCore.QRect(0, 470, 731, 161))
        self.time_delay_plt.setObjectName("time_delay_plt")
        self.result_group.raise_()
        self.name1.raise_()
        self.name2.raise_()
        self.line.raise_()
        self.line_2.raise_()
        self.bt_start.raise_()
        self.bt_load.raise_()
        self.folder_name.raise_()
        self.bt_load_2.raise_()
        self.label.raise_()
        self.label_2.raise_()
        self.label_3.raise_()
        self.label_4.raise_()
        self.gridLayoutWidget.raise_()
        self.label_25.raise_()
        self.line_3.raise_()
        self.formLayoutWidget.raise_()
        self.label_13.raise_()
        self.personal_data.raise_()
        self.td.raise_()
        self.apply_setting_camera.raise_()
        self.horizontalLayoutWidget.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 26))
        self.menubar.setObjectName("menubar")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menu = QtWidgets.QMenu(self.menuHelp)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionmeaning = QtWidgets.QAction(MainWindow)
        self.actionmeaning.setObjectName("actionmeaning")
        self.action = QtWidgets.QAction(MainWindow)
        self.action.setObjectName("action")
        self.menu.addAction(self.action)
        self.menuHelp.addAction(self.menu.menuAction())
        self.menuHelp.addAction(self.actionmeaning)
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.name1.setText(_translate("MainWindow", "Speckleplethysmography"))
        self.name2.setText(_translate("MainWindow", "Measurement Program"))
        self.bt_start.setText(_translate("MainWindow", "เริ่ม"))
        self.bt_load.setText(_translate("MainWindow", "โหลด"))
        self.folder_name.setText(_translate("MainWindow", "FOLDER__NAME__"))
        self.label.setText(_translate("MainWindow", "ข้อมูลทั่วไป"))
        self.label_2.setText(_translate("MainWindow", "SNR"))
        self.label_3.setText(_translate("MainWindow", "Frequency"))
        self.label_4.setText(_translate("MainWindow", "Heart Rate"))
        self.label_6.setText(_translate("MainWindow", "cPPG"))
        self.label_9.setText(_translate("MainWindow", "SPG"))
        self.label_5.setText(_translate("MainWindow", "iPPG"))
        self.cppg_snr.setText(_translate("MainWindow", "18 dB"))
        self.ippg_snr.setText(_translate("MainWindow", "18 dB"))
        self.spg_snr.setText(_translate("MainWindow", "18 dB"))
        self.ippg_hr.setText(_translate("MainWindow", "87 bpm"))
        self.cppg_hr.setText(_translate("MainWindow", "87 bpm"))
        self.ippg_freq.setText(_translate("MainWindow", "1.3002 Hz"))
        self.cppg_freq.setText(_translate("MainWindow", "1.3002 Hz"))
        self.spg_hr.setText(_translate("MainWindow", "87 bpm"))
        self.spg_freq.setText(_translate("MainWindow", "1.3002 Hz"))
        self.label_25.setText(_translate("MainWindow", "ค่าวัด"))
        self.label_14.setText(_translate("MainWindow", "Frame rate"))
        self.label_18.setText(_translate("MainWindow", "Exposure time (µS)"))
        self.label_19.setText(_translate("MainWindow", "Video length (S)"))
        self.frame_rate.setItemText(0, _translate("MainWindow", "60 FPS"))
        self.frame_rate.setItemText(1, _translate("MainWindow", "120 FPS"))
        self.frame_rate.setItemText(2, _translate("MainWindow", "180 FPS"))
        self.frame_rate.setItemText(3, _translate("MainWindow", "240 FPS"))
        self.label_20.setText(_translate("MainWindow", "Size iPPG"))
        self.label_24.setText(_translate("MainWindow", "Size SPG"))
        self.size_ippg.setItemText(0, _translate("MainWindow", "30x30 px"))
        self.size_ippg.setItemText(1, _translate("MainWindow", "50x50 px"))
        self.size_ippg.setItemText(2, _translate("MainWindow", "100x100 px"))
        self.size_ippg.setItemText(3, _translate("MainWindow", "150x150 px"))
        self.size_ippg.setItemText(4, _translate("MainWindow", "200x200 px"))
        self.size_ippg.setItemText(5, _translate("MainWindow", "250x250 px"))
        self.size_ippg.setItemText(6, _translate("MainWindow", "300x300 px"))
        self.size_spg.setItemText(0, _translate("MainWindow", "30x30 px"))
        self.size_spg.setItemText(1, _translate("MainWindow", "50x50 px"))
        self.size_spg.setItemText(2, _translate("MainWindow", "100x100 px"))
        self.size_spg.setItemText(3, _translate("MainWindow", "150x150 px"))
        self.size_spg.setItemText(4, _translate("MainWindow", "200x200 px"))
        self.size_spg.setItemText(5, _translate("MainWindow", "250x250 px"))
        self.size_spg.setItemText(6, _translate("MainWindow", "300x300 px"))
        self.label_13.setText(_translate("MainWindow", "ตั้งค่ากล้อง"))
        self.personal_data.setText(_translate("MainWindow", "อายุ 21 ปี น้ำหนัก 60 กิโลกรัม ส่วนสูง 172 BMI 20.96"))
        self.td.setText(_translate("MainWindow", "Average time delay between SPG and PPG signal: 0.002 S."))
        self.apply_setting_camera.setText(_translate("MainWindow", "ปรับใช้"))
        self.personal_data_2.setText(_translate("MainWindow", "pulse sensor:"))
        self.port.setText(_translate("MainWindow", "Found"))
        self.freq_ippg.setText(_translate("MainWindow", "Spectrum iPPG"))
        self.freq_spg.setText(_translate("MainWindow", "Spectrum SPG"))
        self.spg_cppg.setText(_translate("MainWindow", "SPG & cPPG"))
        self.time_delay.setText(_translate("MainWindow", "Time delay"))
        self.ippg_cppg.setText(_translate("MainWindow", "iPPG & cPPG"))
        self.menuHelp.setTitle(_translate("MainWindow", "ช่วยเหลือ"))
        self.menu.setTitle(_translate("MainWindow", "วิธีใช้งาน"))
        self.actionmeaning.setText(_translate("MainWindow", " คำนิยาม"))
        self.action.setText(_translate("MainWindow", "ไม่บอก"))
