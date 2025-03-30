# Project ST01
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter
from scipy import fftpack
from scipy.signal import find_peaks
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvas
import os
from PyQt5.QtWidgets import QAction
import shutil
from tkinter import *
from PyQt5 import QtGui


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        ######### MainWindow ###########
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1552, 974)
        MainWindow.setMinimumSize(QtCore.QSize(1552, 974))
        MainWindow.setMaximumSize(QtCore.QSize(1552, 974))
        MainWindow.setStyleSheet("background-color: rgba(252, 119, 34, 30);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        ############ Icon ############
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("ST05.png"),
                       QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setLayoutDirection(QtCore.Qt.LeftToRight)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        ######## QPushButton #########
        self.bt1 = QtWidgets.QPushButton(self.centralwidget)
        self.bt1.setGeometry(QtCore.QRect(540, 40, 171, 41))
        self.bt1.setStyleSheet("font: 14pt \"MS Shell Dlg 2\";\n"
                               "background-color: rgb(239, 114, 21);")
        self.bt1.setObjectName("bt1")

        self.bt2 = QtWidgets.QPushButton(self.centralwidget)
        self.bt2.setGeometry(QtCore.QRect(60, 180, 181, 41))
        self.bt2.setStyleSheet("font: 14pt \"MS Shell Dlg 2\";\n"
                               "background-color: rgb(239, 114, 21);")
        self.bt2.setObjectName("bt2")

        self.bt3 = QtWidgets.QPushButton(self.centralwidget)
        self.bt3.setGeometry(QtCore.QRect(570, 100, 121, 41))
        self.bt3.setStyleSheet("font: 10pt \"MS Shell Dlg 2\";\n"
                               "background-color: rgb(239, 114, 21);")
        self.bt3.setObjectName("bt3")

        ##########  Graph display ###############
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_3.setGeometry(
            QtCore.QRect(730, 50, 791, 861))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(
            self.verticalLayoutWidget_3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame = QtWidgets.QFrame(self.verticalLayoutWidget_3)
        self.frame.setStyleSheet("background-color: rgba(252, 119, 34, 30);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")

        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setGeometry(QtCore.QRect(400, 0, 391, 261))
        self.frame_2.setStyleSheet(
            "background-color: rgba(255, 231, 157, 100);")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.s3 = QtWidgets.QFrame(self.frame_2)
        self.s3.setMinimumSize(QtCore.QSize(391, 261))
        self.s3.setMaximumSize(QtCore.QSize(391, 261))
        self.s3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.s3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.s3.setObjectName("s3")
        self.hl3 = QtWidgets.QHBoxLayout(self.s3)
        self.hl3.setContentsMargins(0, 0, 0, 0)
        self.hl3.setObjectName("hl3")

        self.frame_4 = QtWidgets.QFrame(self.frame)
        self.frame_4.setEnabled(True)
        self.frame_4.setGeometry(QtCore.QRect(400, 270, 391, 261))
        self.frame_4.setStyleSheet(
            "background-color: rgba(255, 179, 124, 100);")
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.s4 = QtWidgets.QFrame(self.frame_4)
        self.s4.setMinimumSize(QtCore.QSize(391, 261))
        self.s4.setMaximumSize(QtCore.QSize(391, 261))
        self.s4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.s4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.s4.setObjectName("s4")
        self.hl4 = QtWidgets.QHBoxLayout(self.s4)
        self.hl4.setContentsMargins(0, 0, 0, 0)
        self.hl4.setObjectName("hl4")

        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setGeometry(QtCore.QRect(0, 270, 391, 261))
        self.frame_3.setStyleSheet(
            "background-color: rgba(255, 231, 157, 100);")
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.s5 = QtWidgets.QFrame(self.frame_3)
        self.s5.setMinimumSize(QtCore.QSize(391, 261))
        self.s5.setMaximumSize(QtCore.QSize(391, 261))
        self.s5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.s5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.s5.setObjectName("s5")
        self.hl5 = QtWidgets.QHBoxLayout(self.s5)
        self.hl5.setContentsMargins(0, 0, 0, 0)
        self.hl5.setObjectName("hl5")

        self.frame_5 = QtWidgets.QFrame(self.frame)
        self.frame_5.setGeometry(QtCore.QRect(0, 600, 391, 261))
        self.frame_5.setStyleSheet(
            "background-color: rgba(255, 188, 139, 100);")
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.s6 = QtWidgets.QFrame(self.frame_5)
        self.s6.setMinimumSize(QtCore.QSize(391, 261))
        self.s6.setMaximumSize(QtCore.QSize(391, 261))
        self.s6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.s6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.s6.setObjectName("s6")
        self.hl6 = QtWidgets.QHBoxLayout(self.s6)
        self.hl6.setContentsMargins(0, 0, 0, 0)
        self.hl6.setObjectName("hl6")

        self.frame_1 = QtWidgets.QFrame(self.frame)
        self.frame_1.setGeometry(QtCore.QRect(0, 0, 391, 261))
        self.frame_1.setStyleSheet(
            "background-color: rgba(255, 179, 124, 100);")
        self.frame_1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_1.setObjectName("frame_1")
        self.s2 = QtWidgets.QFrame(self.frame_1)
        self.s2.setMinimumSize(QtCore.QSize(391, 261))
        self.s2.setMaximumSize(QtCore.QSize(391, 261))
        self.s2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.s2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.s2.setObjectName("s2")
        self.hl2 = QtWidgets.QHBoxLayout(self.s2)
        self.hl2.setContentsMargins(0, 0, 0, 0)
        self.hl2.setObjectName("hl2")

        ########### Section ##########
        self.t1 = QtWidgets.QLabel(self.centralwidget)
        self.t1.setGeometry(QtCore.QRect(20, 20, 491, 71))
        self.t1.setStyleSheet("\n"
                              "background-color: rgb(139, 64, 0);")
        self.t1.setObjectName("t1")
        self.t1_2 = QtWidgets.QLabel(self.centralwidget)
        self.t1_2.setGeometry(QtCore.QRect(20, 100, 491, 61))
        self.t1_2.setStyleSheet("\n"
                                "background-color: rgb(177, 86, 15);")
        self.t1_2.setObjectName("t1_2")

        self.t2 = QtWidgets.QLabel(self.centralwidget)
        self.t2.setGeometry(QtCore.QRect(270, 180, 431, 41))
        self.t2.setStyleSheet("\n"
                              "background-color: rgb(255, 222, 137);")
        self.t2.setObjectName("t2")

        self.t3 = QtWidgets.QLabel(self.centralwidget)
        self.t3.setGeometry(QtCore.QRect(20, 230, 701, 41))
        self.t3.setStyleSheet("\n"
                              "background-color: rgb(96, 40, 0);")
        self.t3.setObjectName("t3")

        self.t5 = QtWidgets.QLabel(self.frame)
        self.t5.setGeometry(QtCore.QRect(0, 548, 391, 41))
        self.t5.setStyleSheet("background-color: rgb(177, 86, 15);")
        self.t5.setObjectName("t5")

        #####  Word meanings ###########
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(self.frame)
        self.verticalLayoutWidget_4.setGeometry(
            QtCore.QRect(460, 580, 311, 251))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(
            self.verticalLayoutWidget_4)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.p1 = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        self.p1.setObjectName("p1")
        self.verticalLayout_5.addWidget(self.p1)
        self.p2 = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        self.p2.setObjectName("p2")
        self.verticalLayout_5.addWidget(self.p2)
        self.p3 = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        self.p3.setObjectName("p3")
        self.verticalLayout_5.addWidget(self.p3)
        self.p4 = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        self.p4.setObjectName("p4")
        self.verticalLayout_5.addWidget(self.p4)
        self.p5 = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        self.p5.setObjectName("p5")
        self.verticalLayout_5.addWidget(self.p5)
        self.p6 = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        self.p6.setObjectName("p6")
        self.verticalLayout_5.addWidget(self.p6)
        self.p7 = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        self.p7.setObjectName("p7")
        self.verticalLayout_5.addWidget(self.p7)

        #### Display Infomation ########
        self.verticalLayout_3.addWidget(self.frame)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 270, 341, 571))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.l1 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.l1.setStyleSheet("background-color: rgb(177, 86, 15);")
        self.l1.setObjectName("l1")
        self.verticalLayout.addWidget(self.l1)
        self.l2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.l2.setObjectName("l2")
        self.verticalLayout.addWidget(self.l2)
        self.l3 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.l3.setObjectName("l3")
        self.verticalLayout.addWidget(self.l3)
        self.l4 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.l4.setObjectName("l4")
        self.verticalLayout.addWidget(self.l4)
        self.l5 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.l5.setObjectName("l5")
        self.verticalLayout.addWidget(self.l5)
        self.l6 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.l6.setStyleSheet("background-color: rgb(177, 86, 15);")
        self.l6.setObjectName("l6")
        self.verticalLayout.addWidget(self.l6)
        self.l7 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.l7.setObjectName("l7")
        self.verticalLayout.addWidget(self.l7)
        self.l8 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.l8.setObjectName("l8")
        self.verticalLayout.addWidget(self.l8)
        self.l9 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.l9.setObjectName("l9")
        self.verticalLayout.addWidget(self.l9)
        self.l10 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.l10.setObjectName("l10")
        self.verticalLayout.addWidget(self.l10)
        self.l11 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.l11.setObjectName("l11")
        self.verticalLayout.addWidget(self.l11)
        self.l12 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.l12.setObjectName("l12")
        self.verticalLayout.addWidget(self.l12)

        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(730, 0, 791, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(
            self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.t3_2 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.t3_2.setEnabled(True)
        self.t3_2.setStyleSheet("background-color: rgb(177, 86, 15);")
        self.t3_2.setObjectName("t3_2")
        self.horizontalLayout.addWidget(self.t3_2)
        self.t4 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.t4.setStyleSheet("background-color: rgb(177, 86, 15);")
        self.t4.setObjectName("t4")
        self.horizontalLayout.addWidget(self.t4)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(
            QtCore.QRect(370, 270, 351, 573))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(
            self.verticalLayoutWidget_2)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.r1 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.r1.setStyleSheet("background-color: rgb(204, 119, 34);")
        self.r1.setObjectName("r1")
        self.verticalLayout_4.addWidget(self.r1)
        self.r2 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.r2.setObjectName("r2")
        self.verticalLayout_4.addWidget(self.r2)
        self.r3 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.r3.setObjectName("r3")
        self.verticalLayout_4.addWidget(self.r3)
        self.r4 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.r4.setObjectName("r4")
        self.verticalLayout_4.addWidget(self.r4)
        self.r5 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.r5.setObjectName("r5")
        self.verticalLayout_4.addWidget(self.r5)
        self.r6 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.r6.setStyleSheet("background-color: rgb(204, 119, 34);")
        self.r6.setObjectName("r6")
        self.verticalLayout_4.addWidget(self.r6)
        self.r7 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.r7.setObjectName("r7")
        self.verticalLayout_4.addWidget(self.r7)
        self.r8 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.r8.setObjectName("r8")
        self.verticalLayout_4.addWidget(self.r8)
        self.r9 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.r9.setObjectName("r9")
        self.verticalLayout_4.addWidget(self.r9)
        self.r10 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.r10.setObjectName("r10")
        self.verticalLayout_4.addWidget(self.r10)
        self.r11 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.r11.setObjectName("r11")
        self.verticalLayout_4.addWidget(self.r11)
        self.r12 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.r12.setObjectName("r12")
        self.verticalLayout_4.addWidget(self.r12)

        self.lr1 = QtWidgets.QLabel(self.centralwidget)
        self.lr1.setGeometry(QtCore.QRect(20, 850, 681, 31))
        self.lr1.setObjectName("lr1")

        ########## QAction ###########
        self.openAction = QAction('&Open')
        self.openAction.setShortcut('Ctrl+O')
        self.openAction.setStatusTip('Open movie')
        self.openAction.triggered.connect(self.selectvideo)

        self.help = QAction('&Read_me')
        self.help.setShortcut('Ctrl+H')
        self.help.setStatusTip('Datesheet')
        self.help.triggered.connect(self.Read_me)

        self.exitAction = QAction('&Exit')
        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('Exit application')
        self.exitAction.triggered.connect(self.exitCall)

        ############  Menubar ##############
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1552, 26))
        self.menubar.setObjectName("menubar")

        self.fileMenu = self.menubar.addMenu('&File')
        self.fileMenu.addAction(self.openAction)
        self.fileMenu.addAction(self.help)
        self.fileMenu.addAction(self.exitAction)

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate(
            "MainWindow", "Speckleplethysmography Measurement Program "))

        ########## Set QPushButton #########
        self.bt1.setToolTip(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:8pt;\">Select the video you want to process.</span></p></body></html>"))
        self.bt1.setText(_translate("MainWindow", "Select video"))
        self.bt2.setToolTip(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:8pt;\">Start the processing of the program.</span></p></body></html>"))
        self.bt2.setText(_translate("MainWindow", "Start Process"))
        self.bt3.setToolTip(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:8pt;\">Please press \' q \' to cancel the video show.</span></p></body></html>"))
        self.bt3.setText(_translate("MainWindow", "Show Video"))

        ############# Set Section ########
        self.t1.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-family:\'Amasis MT Pro Black\'; font-size:22pt; font-weight:600; color:#ffffff;\">Speckleplethysmography </span></p></body></html>"))
        self.t1_2.setText(_translate(
            "MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-family:\'Amasis MT Pro Black\'; font-size:18pt; font-weight:600; color:#ffffff;\">Measurement Program </span></p></body></html>"))
        self.t3.setText(_translate(
            "MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt; font-weight:600; color:#ffffff;\">Information</span></p></body></html>"))
        self.t4.setText(_translate(
            "MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt; font-weight:600; color:#ffffff;\">PPG signal</span></p></body></html>"))
        self.t3_2.setText(_translate(
            "MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt; font-weight:600; color:#ffffff;\">SPG signal</span></p></body></html>"))
        self.t2.setText(_translate(
            "MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">Please select your video.</span></p></body></html>"))
        self.t5.setText(_translate(
            "MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:600; color:#ffffff;\">SPG &amp; PPG signal</span></p></body></html>"))

        ############## Set Word meaning ######
        self.p1.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; color:#ffffff;\">SPG = Speckleplethysmography </span></p></body></html>"))
        self.p2.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; color:#ffffff;\">PPG = Photoplethysmography </span></p></body></html>"))
        self.p3.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; color:#ffffff;\">SNR = Signal to Noise Ratio </span></p></body></html>"))
        self.p4.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; color:#ffffff;\">HRV = Heart Rate Variability </span></p></body></html>"))
        self.p5.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; color:#ffffff;\">H1 = first harmonic </span></p></body></html>"))
        self.p6.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; color:#ffffff;\">H2 = Second harmonic </span></p></body></html>"))
        self.p7.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; color:#ffffff;\">H3 = Third harmonic </span></p></body></html>"))

        ############# Set Data Display ############
        # SPG
        self.l1.setText(_translate(
            "MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; color:#ffffff;\">SPG signal</span></p></body></html>"))
        self.l2.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">SNR :</span></p></body></html>"))
        self.l3.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">HRV :</span></p></body></html>"))
        self.l4.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">Heart rate is :</span></p></body></html>"))
        self.l5.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">The frequency is :</span></p></body></html>"))
        self.l6.setText(_translate(
            "MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; color:#ffffff;\">Fourier transform</span></p></body></html>"))
        self.l7.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H1 :</span></p></body></html>"))
        self.l8.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H2 :</span></p></body></html>"))
        self.l9.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H3 :</span></p></body></html>"))
        self.l10.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H3/H1 :</span></p></body></html>"))
        self.l11.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H3/H2 :</span></p></body></html>"))
        self.l12.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H2/H1 :</span></p></body></html>"))
        # PPG
        self.r1.setText(_translate(
            "MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; color:#ffffff;\">PPG signal</span></p></body></html>"))
        self.r2.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">SNR :</span></p></body></html>"))
        self.r3.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">HRV :</span></p></body></html>"))
        self.r4.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">Heart rate is :</span></p></body></html>"))
        self.r5.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">The frequency is :</span></p></body></html>"))
        self.r6.setText(_translate(
            "MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; color:#ffffff;\">Fourier transform</span></p></body></html>"))
        self.r7.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H1 :</span></p></body></html>"))
        self.r8.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H2 :</span></p></body></html>"))
        self.r9.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H3 :</span></p></body></html>"))
        self.r10.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H3/H1 :</span></p></body></html>"))
        self.r11.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H3/H2 :</span></p></body></html>"))
        self.r12.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H2/H1 :</span></p></body></html>"))
        # Time delay
        self.lr1.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">Average time delay between SPG and PPG signal is :</span></p></body></html>"))

        ######### Connect QPushButton ########
        self.bt2.clicked.connect(self.showgrap)
        self.bt1.clicked.connect(self.selectvideo)
        self.bt3.clicked.connect(self.showvideo)

    ########   QAction  ###########

    def Read_me(self):
        class Window(Frame):
            def __init__(self, master=None):
                Frame.__init__(self, master)
                self.master = master
                self.pack(fill=BOTH, expand=1)

                text = Label(self, text='How to use                                                                                \n '
                             '1. Click '+"'Select video'" + ' to select the video to be processed.\n '
                             '2. Click ' + "'Select video'" + ' to calculate the result.                      \n'
                             '3. Click '+"'Show video'" + ' to show video.                                   \n')
                text.place(x=40, y=10)
                # text.pack()
        root = Tk()
        app = Window(root)
        root.wm_title("Read me")
        root.geometry("500x200")
        root.mainloop()
        # print('data')

    def exitCall(self):
        print('exit')
        sys.exit(app.exec_())

    ###### QPushButton Select video ######
    def selectvideo(self):
        self.setupUi(MainWindow)
        vdfname = QFileDialog.getOpenFileName(None, ("Selecciona los medios"),
                                              os.getcwd(),
                                              ("Video Files (*.avi *.mp4 *.flv)"))
        self.vdofname = vdfname[0]
        print(self.vdofname)
        cap = cv2.VideoCapture(self.vdofname)
        if (cap.isOpened() == False):
            self.t2.setText(
                "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">Not found!!! Please try again.</span></p></body></html>")
            print("Not found. Please try again")
        else:
            self.t2.setText(
                "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">Ready...</span></p></body></html>")

    ###### QPushButton Show video ######
    def showvideo(self):
        self.t2.setText(
            "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">Please select your video.</span></p></body></html>")
        cap = cv2.VideoCapture(self.vdofname)
        self.t2.setText(
            "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">Show Video ...</span></p></body></html>")
        if (cap.isOpened() == False):
            self.t2.setText(
                "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">Not found!!! Please try again.</span></p></body></html>")
            print("Error opening video stream or file")
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    self.t2.setText(
                        "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">Ready...</span></p></body></html>")
                    break
            else:
                self.t2.setText(
                    "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">Ready...</span></p></body></html>")
                break

        cap.release()
        cv2.destroyAllWindows()

    ###### QPushButton Start Process ######
    def showgrap(self):
        self.t2.setText(
            "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">Please select your video.</span></p></body></html>")
        cap = cv2.VideoCapture(self.vdofname)
        pathCurrent = os.getcwd()
        # Directory
        directory_spg = "Picture SPG"
        directory_ppg = "Picture PPG"
        # Parent Directory path
        parent_dir = pathCurrent
        # Path
        path_spg = os.path.join(parent_dir, directory_spg)
        path_ppg = os.path.join(parent_dir, directory_ppg)

        # Create the directory
        if os.path.exists(path_spg):
            shutil.rmtree(path_spg)
            os.mkdir(path_spg)
        else:
            os.mkdir(path_spg)

        if os.path.exists(path_ppg):
            shutil.rmtree(path_ppg)
            os.mkdir(path_ppg)
        else:
            os.mkdir(path_ppg)
        if (cap.isOpened() == False):
            print("Error for open file...")
        currentFrame = 1
        #
        while (cap.isOpened()):
            self.t2.setText(
                "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">Processing ...</span></p></body></html>")
            ret, frame = cap.read()
            if ret == True:
                xcenter = 300  # select picture center of axis X
                ycenter = 300  # select picture center of axis Y

                # 200*200
                size_spg = 200
                xx1_spg = int(xcenter - (size_spg/2))
                xx2_spg = int(xcenter + (size_spg/2))
                yy1_spg = int(ycenter - (size_spg/2))
                yy2_spg = int(ycenter + (size_spg/2))

                # 100*100
                size_ppg = 100
                xx1_ppg = int(xcenter - (size_ppg/2))
                xx2_ppg = int(xcenter + (size_ppg/2))
                yy1_ppg = int(ycenter - (size_ppg/2))
                yy2_ppg = int(ycenter + (size_ppg/2))

                spgcropped = frame[yy1_spg:yy2_spg,
                                   xx1_spg:xx2_spg]  # [y1:y2,x1:x2]
                ppgcropped = frame[yy1_ppg:yy2_ppg,
                                   xx1_ppg:xx2_ppg]  # [y1:y2,x1:x2]
                cv2.imwrite(os.path.join(path_spg, 'spg_frame' +
                            str(currentFrame) + '.png'), spgcropped)
                cv2.imwrite(os.path.join(path_ppg, 'ppg_frame' +
                            str(currentFrame) + '.png'), ppgcropped)
                # To stop duplicate images
                currentFrame += 1
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                    # Break the loop
            else:
                print("End of this file.")
                break
            # When everything done, release the video capture object
        cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()

        def AvgI(x, k, s):
            nx = x.shape
            nk = k.shape
            if (type(s) == int):
                s = s, s
            return np.array([[(x[i:i+nk[0], j:j+nk[1]]*k).sum()/nk[0]**2
                              for j in range(0, nx[1], s[1])]
                             for i in range(0, nx[0], s[0])])

        def convo2d(x1, k1):
            s1 = 5
            nx1 = x1.shape
            nk1 = k1.shape

            if (type(s1) == int):
                s1 = s1, s1
                return np.array([[((x1[i:i+nk1[0], j:j+nk1[1]]).std() / np.mean((x1[i:i+nk1[0], j:j+nk1[1]]*k1)))
                                  for j in range(0, nx1[1]-nk1[1]+1, s1[1])]
                                 for i in range(0, nx1[0]-nk1[0]+1, s1[0])])

        def convo2ds(x1):
            nx1 = x1.shape
            s1 = 5

            # return std/mean
            return np.array([[((x1[i:i+s1, j:j+s1]).std() / np.mean(x1[i:i+s1, j:j+s1]))
                            for j in range(0, nx1[1]-s1+1, s1)]
                            for i in range(0, nx1[0]-s1+1, s1)])

        # def filter SPG signal
        def butter_lowPass0(lowCut0, fs0, order0):
            nyq0 = 0.5 * fs0
            normal_cutoff0 = lowCut0 / nyq0
            b0, a0 = butter(order0, normal_cutoff0, btype='low', analog=False)
            return b0, a0

        def butter_lowPass_filter0(allpossitionspg, lowCut0, fs0, order0):
            b0, a0 = butter_lowPass0(lowCut0, fs0, order0)
            y0 = lfilter(b0, a0, allpossitionspg)
            return y0

        def butter_highPass00(highCut0, fs0, order0):
            nyq00 = 0.5 * fs0
            normal_cutoff00 = highCut0 / nyq00
            b00, a00 = butter(order0, normal_cutoff00,
                              btype='high', analog=False)
            return b00, a00

        def butter_highPass_filter00(spg002, highCut0, fs0, order0):
            b00, a00 = butter_highPass00(highCut0, fs0, order0)
            y00 = lfilter(b00, a00, spg002)
            return y00

        # def filter PPG signal
        def butter_lowPass1(lowCut1, sampling_rate1, order1):
            nyq1 = 0.5 * sampling_rate1
            normal_cutoff1 = lowCut1 / nyq1
            print("normal cutoff:", normal_cutoff1)
            b1, a1 = butter(order1, normal_cutoff1, btype='low', analog=False)
            return b1, a1

        def butter_lowPass_filter1(ppg, lowCut1, sampling_rate1, order1):
            b1, a1 = butter_lowPass1(lowCut1, sampling_rate1, order1)
            y1 = lfilter(b1, a1, ppg)
            return y1

        def butter_highPass11(highCut1, sampling_rate1, order1):
            nyq11 = 0.5 * sampling_rate1
            normal_cutoff11 = highCut1 / nyq11
            b11, a11 = butter(order1, normal_cutoff11,
                              btype='high', analog=False)
            return b11, a11

        def butter_highPass_filter11(ppg11, highCut1, sampling_rate1, order1):
            b11, a11 = butter_highPass11(highCut1, sampling_rate1, order1)
            y11 = lfilter(b11, a11, ppg11)
            return y11

        allpossitionspg = []
        allK = []
        ppg01 = []
        ppg2 = []
        spg01 = []
        currentFrame2 = 1
        if (currentFrame2 == 1):
            while (currentFrame2 < currentFrame):
                ppg_name_in = "ppg_frame" + str(currentFrame2) + ".png"
                # print('\ncreating ppg ...output ' + str(currentFrame2))
                k01 = size_ppg
                k02 = size_ppg

                # average identity
                x = cv2.imread(os.path.join(
                    path_ppg, ppg_name_in), 0)  # [:,:,0]
                # k = np.ones([k01, k02])  # create a 5x5 kernel of ones
                # s = k01
                # y2 = AvgI(x, k, s)
                # y001 = np.array(y2)
                # y002 = y001.flatten()
                # y003 = y002[0]

                y004 = np.mean(x)

                ppg01.append(str(y004))
                ppg2.append(y004)

                spg_name_in = "spg_frame" + str(currentFrame2) + ".png"
                x1 = cv2.imread(os.path.join(
                    path_spg, spg_name_in), 0)  # [:,:,0]
                k1 = np.ones([5, 5])
                contrast = convo2ds(x1)  # K #SPG signal array
                pointK = np.mean(contrast)
                allK.append(pointK)
                expt = 15*(10**(-3))  # T = 15ms #exposure time of camera
                spgsignalarray = 1/(2**expt*np.square(contrast))  # 1/(2TK^2)
                # Average over all pixel
                possitionspg = np.mean(spgsignalarray)
                allpossitionspg.append(possitionspg)
                spg01.append(str(possitionspg))
                currentFrame2 += 1
                maxx = currentFrame
                if currentFrame2 == maxx:
                    break

        # plot ppg2 and allpossitionspg
        plt.plot(ppg2)
        plt.plot(allpossitionspg)
        plt.show()

        ppg2 = ppg2[0:len(ppg2)]
        ppg01 = ppg01[0:len(ppg2)]
        # time = np.linspace(0, len(ppg2)/25, len(ppg2))
        # time = time

        # normalize PPG
        ppg = ppg2[0:len(ppg2)]
        ppg = ppg/max(ppg)
        ppg = ppg*1000-999

        # Using filter of SPG signal
        fs0 = 25*2  # Hz
        order0 = 3  # 5
        lowCut0 = 4  # 4
        highCut0 = 0.83  # 0.83

        # b0, a0 = butter_lowPass0(lowCut0, fs0, order0)
        spg002 = butter_lowPass_filter0(allpossitionspg, lowCut0, fs0, order0)
        # b00, a00 = butter_highPass00(highCut0, fs0, order0)
        # spg004 = butter_highPass_filter00(spg002, highCut0, fs0, order0)
        spg002 = spg002

        # Using filter of PPG signal
        sampling_rate1 = 25*2  # Hz
        order1 = 3  # 5
        lowCut1 = 4  # 4
        highCut1 = 0.83  # 0.83

        # b1, a1 = butter_lowPass1(lowCut1, sampling_rate1, order1)
        ppg11 = butter_lowPass_filter1(ppg, lowCut1, sampling_rate1, order1)
        # b11, a11 = butter_highPass11(highCut1, sampling_rate1, order1)
        ppg001 = butter_highPass_filter11(
            ppg11, highCut1, sampling_rate1, order1)

        # normallize PPG
        ppg3 = ppg001[0:len(ppg001)]
        ppg3 = ppg3/max(ppg3)
        ppg3 = ppg3*100-99
        ppg3 = ppg3 - np.mean(ppg3)
        ppg1 = -ppg3  # invert PPG

        # normallize SPG
        spg003 = spg002[0:len(spg002)]
        spg003 = spg003/max(spg003)
        spg003 = spg003*1000-999
        spg003 = spg003 - np.mean(spg003)

        # data_spg_final = spg003
        # data_ppg_final = ppg1
        # startTime = 1
        # stopTime = 10
        # startTime = startTime*25
        # stopTime = stopTime*25
        # data_spg_final = spg003[startTime:len(spg003)]
        # data_ppg_final = ppg1[startTime:len(ppg1)]
        data_spg_final = spg003
        data_ppg_final = ppg1

        # print('Processes complete')
        maxyspg = max(data_spg_final[25:len(data_spg_final)])
        maxyppg = max(data_ppg_final[25:len(data_ppg_final)])
        minyspg = min(data_spg_final[25:len(data_spg_final)])
        minyppg = min(data_ppg_final[25:len(data_ppg_final)])
        if (maxyspg > maxyppg):
            ymax01 = maxyspg + 30
        else:
            ymax01 = maxyppg + 30

        if (minyspg < minyppg):
            ymin01 = minyspg - 30
        else:
            ymin01 = minyppg - 30

        # Part of SNR (Signal to Noise Ratio)
        data_spg0 = data_spg_final[25:250]
        data_ppg0 = data_ppg_final[25:250]

        nsegments = 12
        segment_length = (2*data_spg0.shape[0]) // (nsegments + 1)
        fps = 25  # Sampling rate, or number of measurements per second
        fre_spg, psd_spg = signal.welch(data_spg0, fps, nperseg=segment_length)

        nsegments = 12
        segment_length = (2*data_ppg0.shape[0]) // (nsegments + 1)
        fps = 25  # Sampling rate, or number of measurements per second
        fre_ppg, psd_ppg = signal.welch(data_ppg0, fps, nperseg=segment_length)

        data_spg = abs(fre_spg)
        data_ppg = abs(fre_ppg)

        highCut_snr = 4
        lowCut_snr = 0.45
        power_spg = 0
        power_noise_spg = 0
        power_spg_count = 0
        power_noise_spg_count = 0
        power_spg_mean = 0
        power_noise_spg_mean = 0

        for something in range(len(fre_spg)):
            value_fre_spg = fre_spg[something]
            if (value_fre_spg <= lowCut_snr) or (value_fre_spg >= highCut_snr):
                power_noise_spg = power_noise_spg + psd_spg[something]
                power_noise_spg_count = power_noise_spg_count+1
            else:
                power_spg = power_spg + psd_spg[something]
                power_spg_count = power_spg_count+1

        power_spg_mean = power_spg/power_spg_count
        power_noise_spg_mean = power_noise_spg/power_noise_spg_count

        SNR_spg = 10*np.log(power_spg_mean/power_noise_spg_mean)

        power_ppg = 0
        power_noise_ppg = 0
        power_ppg_count = 0
        power_noise_ppg_count = 0
        power_ppg_mean = 0
        power_noise_ppg_mean = 0

        for something in range(len(fre_ppg)):
            value_fre_ppg = fre_ppg[something]
            if (value_fre_ppg <= lowCut_snr) or (value_fre_ppg >= highCut_snr):
                power_noise_ppg = power_noise_ppg + psd_ppg[something]
                power_noise_ppg_count = power_noise_ppg_count+1
            else:
                power_ppg = power_ppg + psd_ppg[something]
                power_ppg_count = power_ppg_count+1

        power_ppg_mean = power_ppg/power_ppg_count
        power_noise_ppg_mean = power_noise_ppg/power_noise_ppg_count

        SNR_ppg = 10*np.log(power_ppg_mean/power_noise_ppg_mean)

        # Part of Time delay # Part of Time delay # Part of Time delay # Part of Time delay
        data_spg = data_spg_final
        data_ppg = data_ppg_final

        start_num = 0
        start_mov = 0
        order_mov = 25
        val_mov = order_mov
        mov_avg_spg = []
        mov_avg_ppg = []
        array_range = np.arange(1, val_mov+1)
        if (start_num == 0):
            while (1 == 1):
                data_spg01 = data_spg[start_mov:val_mov]
                data_ppg01 = data_ppg[start_mov:val_mov]
                start_mov += order_mov
                val_mov += order_mov
                mean_data_spg = np.mean(data_spg01)
                mean_data_ppg = np.mean(data_ppg01)

                for i in array_range:
                    mov_avg_spg.append(mean_data_spg)
                    mov_avg_ppg.append(mean_data_ppg)

                if (val_mov > len(data_spg)+15):
                    break

        mov_avg_spg = np.array(mov_avg_spg[0:len(data_spg)])
        mov_avg_ppg = np.array(mov_avg_ppg[0:len(data_ppg)])

        peaks_spg, _ = find_peaks(data_spg, height=mov_avg_spg)
        peaks_ppg, _ = find_peaks(data_ppg, height=mov_avg_ppg)

        ###### Plot SPG and PPG ###########
        time001 = np.linspace(0, len(data_spg_final)/25, len(data_spg_final))

        def spg():
            fig, ax = plt.subplots()
            color = 'tab:red'
            ax.plot(time001, data_spg_final, color=color)
            ax.set_title('SPG Signal  ')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (Arb.Unit)')
            # ax.set_xlim(-f_s / 2, f_s / 2)
            ax.set_ylim(ymin01, ymax01)
            plt.legend(('SPG Signal', ''), loc='upper right')
            plt.grid(True, which='both')
            plt.show()
            return fig

        time002 = np.linspace(0, len(data_ppg_final)/25, len(data_ppg_final))

        def ppg():
            fig, ax = plt.subplots()
            color = 'blue'
            ax.plot(time002, data_ppg_final, color=color)
            ax.set_title('PPG Signal  ')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (Arb.Unit)')
            # ax.set_xlim(-f_s / 2, f_s / 2)
            ax.set_ylim(ymin01, ymax01)
            plt.legend(('PPG Signal', ''), loc='upper right')
            plt.grid(True, which='both')
            plt.show()
            return fig

        spg_plotG = spg()
        self.plotWidget2 = FigureCanvas(spg_plotG)
        self.hl2.addWidget(self.plotWidget2)

        ppg_plotG = ppg()
        self.plotWidget3 = FigureCanvas(ppg_plotG)
        self.hl3.addWidget(self.plotWidget3)

        list_peaks_spg = []
        self.t2.setText(
            "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">Unable to process !!! Try again.</span></p></body></html>")
        list_peaks_spg = [peaks_spg[0]]
        self.t2.setText(
            "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">Processing.... </span></p></body></html>")
        if (1 == 1):
            k = 0
            i = 0
            j = 1
            while (1 == 1):
                diff_spg = 0
                diff_spg = peaks_spg[j] - list_peaks_spg[k]

                if (diff_spg > 12 and diff_spg < 40):
                    list_peaks_spg.append(peaks_spg[j])
                    k += 1
                else:
                    pass

                i += 1
                j += 1
                max_val = len(peaks_spg)
                if (j == max_val):
                    break

        list_peaks_ppg = []
        list_peaks_ppg = [peaks_ppg[0]]
        if (1 == 1):
            k = 0
            i = 0
            j = 1
            while (1 == 1):
                diff_ppg = 0
                diff_ppg = peaks_ppg[j] - list_peaks_ppg[k]

                if (diff_ppg > 12 and diff_ppg < 50):
                    list_peaks_ppg.append(peaks_ppg[j])
                    k += 1
                else:
                    pass

                i += 1
                j += 1
                max_val = len(peaks_ppg)
                if (j == max_val):
                    break

        num_spg = int(len(list_peaks_spg))
        num_ppg = int(len(list_peaks_ppg))
        diff_for_append = abs(num_spg-num_ppg)
        len_for_append = np.arange(1, diff_for_append+1)
        for i in len_for_append:
            if (num_spg > num_ppg):
                list_peaks_ppg.append(0)
            else:
                list_peaks_spg.append(0)

        len_for_append_support = np.arange(1, 5+1)
        for j_sup in len_for_append_support:
            if (0 == 0):
                list_peaks_ppg.append(100)
                list_peaks_spg.append(0)

        i02 = 0
        j02 = 0
        num_max_val = int(len(list_peaks_spg))
        val_i = np.arange(1, num_max_val + 1)

        peaks_of_spg = []
        peaks_of_ppg = []
        for i in val_i:
            diff_peaks_01 = list_peaks_spg[i02] - list_peaks_ppg[j02]
            # 8 is difference of frame it accept
            if (abs(diff_peaks_01) < 10 or abs(diff_peaks_01) == 0):
                peaks_of_spg.append(list_peaks_spg[i02])
                peaks_of_ppg.append(list_peaks_ppg[j02])
                i02 += 1
                j02 += 1
            elif (list_peaks_spg[i02] > list_peaks_ppg[j02]):
                j02 += 1
                list_peaks_ppg.append(0)
                val_i += 1
            elif (list_peaks_ppg[j02] > list_peaks_spg[i02]):
                i02 += 1
                list_peaks_spg.append(0)
                val_i += 1
            else:
                pass

        peaks_of_spg_after_chek = []  # Reject zero value
        for i in peaks_of_spg:
            if (i != 0):
                peaks_of_spg_after_chek.append(i)

        peaks_of_ppg_after_chek = []
        for j in peaks_of_ppg:
            if (j != 0):
                peaks_of_ppg_after_chek.append(j)

        peaks_of_spg = peaks_of_spg_after_chek
        peaks_of_ppg = peaks_of_ppg_after_chek

        # difference of peaks of SPG and PPG

        all_difference_frame = []
        zip_object = zip(peaks_of_spg, peaks_of_ppg)
        for peaks_of_spg_i, peaks_of_ppg_i in zip_object:
            difference_frame = (peaks_of_ppg_i - peaks_of_spg_i)
            if (difference_frame >= 0):
                all_difference_frame.append(difference_frame)

        cenVal = int(len(all_difference_frame)/2)
        # startRec = 0
        startRec = cenVal-2
        valInfor = 0
        tdFrame = all_difference_frame
        # tdFrame = []
        # if(len(all_difference_frame)<=5):
        #     tdFrame = all_difference_frame
        # else:
        #     for g in all_difference_frame:
        #         valInfor+=1
        #         if(valInfor>startRec):
        #             tdFrame.append(g)
        #             if(len(tdFrame)==5):
        #                 break

        mean_diff = abs(np.mean(tdFrame))
        mean_diff_time = mean_diff/25
        mean_diff_time = mean_diff_time
        tdTime = np.array(tdFrame)/25

        # Part of HRV # Part of HRV # Part of HRV # Part of HRV # Part of HRV # Part of HRV

        peaks_of_spg_hrv = np.array(peaks_of_spg)
        peaks_of_ppg_hrv = np.array(peaks_of_ppg)

        out_spg_hrv = []
        out_ppg_hrv = []
        if (1 == 1):
            u = 0
            v = 1
            while (1 < 2):
                diff_spg_hrv = peaks_of_spg_hrv[v] - peaks_of_spg_hrv[u]
                diff_ppg_hrv = peaks_of_ppg_hrv[v] - peaks_of_ppg_hrv[u]
                if (diff_spg_hrv < 25):
                    out_spg_hrv.append(diff_spg_hrv/25)
                    # if(len(out_spg_hrv)== 5):
                    #     break
                else:
                    print("Reject difference value of SPG is: %d" %
                          diff_spg_hrv)
                if (diff_ppg_hrv < 25):
                    out_ppg_hrv.append(diff_ppg_hrv/25)
                    # if(len(out_ppg_hrv)== 5):
                    #     break
                else:
                    print("Reject difference value of iPPG is: %d" %
                          diff_ppg_hrv)
                u += 1
                v += 1
                if (v == len(peaks_of_spg_hrv)):
                    break

        # calculate Mesurements
        # HRV_SPG (Heart Rate Variability of SPG)
        hrv_spg = np.mean(out_spg_hrv)

        # BPM_SPG (Beat per minute of SPG)
        bpm_spg = 60/hrv_spg

        # HRV_PPG #HRV_PPG #HRV_PPG #HRV_PPG #HRV_PPG #HRV_PPG #HRV_PPG #HRV_PPG
        hrv_ppg = np.mean(out_ppg_hrv)

        # BPM_PPG #BPM_PPG #BPM_PPG #BPM_PPG #BPM_PPG #BPM_PPG #BPM_PPG #BPM_PPG
        bpm_ppg = 60/hrv_ppg

        # SPG Frequency
        fre_spg = bpm_spg/60

        # PPG Frequency
        fre_ppg = bpm_ppg/60

        # FFT (Fast Fourier Transform)
        data_spg_inv = -data_spg_final
        data_ppg_inv = -data_ppg_final
        start_num_inv = 0
        start_mov_inv = 0
        order_mov_inv = 15
        val_mov_inv = order_mov_inv
        mov_avg_spg_inv = []
        mov_avg_ppg_inv = []
        array_range_inv = np.arange(1, val_mov_inv+1)
        if (start_num_inv == 0):
            while (1 == 1):
                data_spg_inv01 = data_spg_inv[start_mov_inv:val_mov_inv]
                data_ppg_inv01 = data_ppg_inv[start_mov_inv:val_mov_inv]
                start_mov_inv += order_mov_inv
                val_mov_inv += order_mov_inv
                mean_data_spg01 = np.mean(data_spg_inv01)
                mean_data_ppg01 = np.mean(data_ppg_inv01)

                for i in array_range_inv:
                    mov_avg_spg_inv.append(mean_data_spg01)
                    mov_avg_ppg_inv.append(mean_data_ppg01)

                if (val_mov_inv > len(data_spg_inv)+15):
                    break

        mov_avg_spg_inv = np.array(mov_avg_spg_inv[0:len(data_spg_inv)])
        mov_avg_ppg_inv = np.array(mov_avg_ppg_inv[0:len(data_ppg_inv)])

        peaks_spg_inv, _ = find_peaks(data_spg_inv, height=mov_avg_spg_inv)
        peaks_ppg_inv, _ = find_peaks(data_ppg_inv, height=mov_avg_ppg_inv)

        list_peaks_spg_inv = 0
        list_peaks_spg_inv = [peaks_spg_inv[0]]

        if (1 == 1):
            k = 0
            i = 0
            j = 1
            while (1 == 1):
                diff_spg = 0
                diff_spg = peaks_spg_inv[j] - list_peaks_spg_inv[k]

                if (diff_spg > 10 and diff_spg < 50):
                    list_peaks_spg_inv.append(peaks_spg_inv[j])
                    k += 1
                else:
                    pass

                i += 1
                j += 1
                max_val = len(peaks_spg_inv)
                if (j == max_val):
                    break

        list_peaks_ppg_inv = 0
        list_peaks_ppg_inv = [peaks_ppg_inv[0]]
        if (1 == 1):
            k = 0
            i = 0
            j = 1
            while (1 == 1):
                diff_ppg = 0
                diff_ppg = peaks_ppg_inv[j] - list_peaks_ppg_inv[k]

                if (diff_ppg > 10 and diff_ppg < 50):
                    list_peaks_ppg_inv.append(peaks_ppg_inv[j])
                    k += 1
                else:
                    pass

                i += 1
                j += 1
                max_val = len(peaks_ppg_inv)
                if (j == max_val):
                    break

        i01 = 0
        j01 = 1
        h1s_spg = []
        h2s_spg = []
        h3s_spg = []
        h4s_spg = []
        h5s_spg = []
        h6s_spg = []

        h1s_ppg = []
        h2s_ppg = []
        h3s_ppg = []
        h4s_ppg = []
        h5s_ppg = []
        h6s_ppg = []

        fh1s_spg = []
        fh2s_spg = []
        fh3s_spg = []
        fh4s_spg = []
        fh5s_spg = []
        fh6s_spg = []

        fh1s_ppg = []
        fh2s_ppg = []
        fh3s_ppg = []
        fh4s_ppg = []
        fh5s_ppg = []
        fh6s_ppg = []

        h3h1_spg = []
        h3h2_spg = []
        h2h1_spg = []

        h3h1_ppg = []
        h3h2_ppg = []
        h2h1_ppg = []

        if (1 == 1):
            while (1 == 1):
                data1_spg = data_spg_final[list_peaks_spg_inv[i01]
                    :list_peaks_spg_inv[j01]]
                data1_ppg = data_ppg_final[list_peaks_ppg_inv[i01]
                    :list_peaks_ppg_inv[j01]]
                i01 += 1
                j01 += 1

                f_s = 25  # Sampling rate, or number of measurements per second
                fft_spg = fftpack.fft(data1_spg)
                freqs_spg = fftpack.fftfreq(len(data1_spg)) * f_s
                fft_spg[0] = 0
                fft_ppg = fftpack.fft(data1_ppg)
                freqs_ppg = fftpack.fftfreq(len(data1_ppg)) * f_s
                fft_ppg[0] = 0

                h1_spg = abs(fft_spg[1])
                h2_spg = abs(fft_spg[2])
                h3_spg = abs(fft_spg[3])
                h4_spg = abs(fft_spg[4])
                h5_spg = abs(fft_spg[5])
                h6_spg = abs(fft_spg[6])
                fh1_spg = abs(freqs_spg[1])
                fh2_spg = abs(freqs_spg[2])
                fh3_spg = abs(freqs_spg[3])
                fh4_spg = abs(freqs_spg[4])
                fh5_spg = abs(freqs_spg[5])
                fh6_spg = abs(freqs_spg[6])

                h1_ppg = abs(fft_ppg[1])
                h2_ppg = abs(fft_ppg[2])
                h3_ppg = abs(fft_ppg[3])
                h4_ppg = abs(fft_ppg[4])
                h5_ppg = abs(fft_ppg[5])
                h6_ppg = abs(fft_ppg[6])
                fh1_ppg = abs(freqs_ppg[1])
                fh2_ppg = abs(freqs_ppg[2])
                fh3_ppg = abs(freqs_ppg[3])
                fh4_ppg = abs(freqs_ppg[4])
                fh5_ppg = abs(freqs_ppg[5])
                fh6_ppg = abs(freqs_ppg[6])

                if (h1_spg > h2_spg and h2_spg > h3_spg and fh1_spg < 3.14 and fh1_spg > 0.8):
                    h1s_spg.append(h1_spg)
                    h2s_spg.append(h2_spg)
                    h3s_spg.append(h3_spg)
                    h4s_spg.append(h4_spg)
                    h5s_spg.append(h5_spg)
                    h6s_spg.append(h6_spg)
                    fh1s_spg.append(fh1_spg)
                    fh2s_spg.append(fh2_spg)
                    fh3s_spg.append(fh3_spg)
                    fh4s_spg.append(fh4_spg)
                    fh5s_spg.append(fh5_spg)
                    fh6s_spg.append(fh6_spg)

                    h3h1_spg.append(h3_spg/h1_spg)
                    h3h2_spg.append(h3_spg/h2_spg)
                    h2h1_spg.append(h2_spg/h1_spg)

                else:
                    pass

                if (h1_ppg > h2_ppg and h2_ppg > h3_ppg and fh1_ppg < 3.14 and fh1_ppg > 0.8):
                    h1s_ppg.append(h1_ppg)
                    h2s_ppg.append(h2_ppg)
                    h3s_ppg.append(h3_ppg)
                    h4s_ppg.append(h4_ppg)
                    h5s_ppg.append(h5_ppg)
                    h6s_ppg.append(h6_ppg)
                    fh1s_ppg.append(fh1_ppg)
                    fh2s_ppg.append(fh2_ppg)
                    fh3s_ppg.append(fh3_ppg)
                    fh4s_ppg.append(fh4_ppg)
                    fh5s_ppg.append(fh5_ppg)
                    fh6s_ppg.append(fh6_ppg)

                    h3h1_ppg.append(h3_ppg/h1_ppg)
                    h3h2_ppg.append(h3_ppg/h2_ppg)
                    h2h1_ppg.append(h2_ppg/h1_ppg)

                else:
                    pass

                if (j01 == int(len(list_peaks_ppg_inv)) or j01 == int(len(list_peaks_spg_inv))):
                    break

        avg_h1_spg = np.mean(h1s_spg)
        avg_h2_spg = np.mean(h2s_spg)
        avg_h3_spg = np.mean(h3s_spg)
        avg_h4_spg = np.mean(h4s_spg)
        avg_h5_spg = np.mean(h5s_spg)
        avg_h6_spg = np.mean(h6s_spg)
        avg_fh1f_spg = np.mean(fh1s_spg)
        avg_fh2f_spg = np.mean(fh2s_spg)
        avg_fh3f_spg = np.mean(fh3s_spg)
        avg_fh4f_spg = np.mean(fh4s_spg)
        avg_fh5f_spg = np.mean(fh5s_spg)
        avg_fh6f_spg = np.mean(fh6s_spg)

        avg_h1_ppg = np.mean(h1s_ppg)
        avg_h2_ppg = np.mean(h2s_ppg)
        avg_h3_ppg = np.mean(h3s_ppg)
        avg_h4_ppg = np.mean(h4s_ppg)
        avg_h5_ppg = np.mean(h5s_ppg)
        avg_h6_ppg = np.mean(h6s_ppg)
        avg_fh1f_ppg = np.mean(fh1s_ppg)
        avg_fh2f_ppg = np.mean(fh2s_ppg)
        avg_fh3f_ppg = np.mean(fh3s_ppg)
        avg_fh4f_ppg = np.mean(fh4s_ppg)
        avg_fh5f_ppg = np.mean(fh5s_ppg)
        avg_fh6f_ppg = np.mean(fh6s_ppg)

        avgH3H1_spg = np.mean(h3h1_spg)  # h3_h1_spg
        avgH3H2_spg = np.mean(h3h2_spg)  # h3_h2_spg
        avgH2H1_spg = np.mean(h2h1_spg)  # h2_h1_spg

        avgH3H1_ppg = np.mean(h3h1_ppg)  # h3_h1_ppg
        avgH3H2_ppg = np.mean(h3h2_ppg)  # h3_h2_ppg
        avgH2H1_ppg = np.mean(h2h1_ppg)  # h2_h1_ppg

        ####### Plot FFT #############
        def fftsp():
            fig, ax = plt.subplots()
            ax.stem(-avg_fh1f_spg, avg_h1_spg, 'red', markerfmt=" ")
            ax.stem(-avg_fh2f_spg, avg_h2_spg, 'red', markerfmt=" ")
            ax.stem(-avg_fh3f_spg, avg_h3_spg, 'red', markerfmt=" ")
            ax.stem(-avg_fh4f_spg, avg_h4_spg, 'red', markerfmt=" ")
            ax.stem(-avg_fh5f_spg, avg_h5_spg, 'red', markerfmt=" ")
            ax.stem(-avg_fh6f_spg, avg_h6_spg, 'red', markerfmt=" ")

            markerline, stemlines, baseline = plt.stem(
                avg_fh1f_spg, avg_h1_spg, 'red', markerfmt='go', label='H1')
            markerline, stemlines, baseline = plt.stem(
                avg_fh2f_spg, avg_h2_spg, 'red', markerfmt='yo', label='H2')
            markerline, stemlines, baseline = plt.stem(
                avg_fh3f_spg, avg_h3_spg, 'red', markerfmt='o', label='H3')
            plt.legend()
            ax.stem(avg_fh4f_spg, avg_h4_spg, 'red', markerfmt=" ")
            ax.stem(avg_fh5f_spg, avg_h5_spg, 'red', markerfmt=" ")
            ax.stem(avg_fh6f_spg, avg_h6_spg, 'red', markerfmt=" ")
            ax.set_title('SPG signal')
            ax.set_xlabel('Frequency[Hz]')
            ax.set_ylabel('Magnitude (Arb.Unit)')
            ax.set_xlim(-f_s / 2, f_s / 2)
            ax.set_ylim(-1, avg_h1_spg+50)
            plt.show()
            return fig

        def fftas():
            fig, ax = plt.subplots()
            ax.stem(-avg_fh1f_ppg, avg_h1_ppg, 'blue', markerfmt=" ")
            ax.stem(-avg_fh2f_ppg, avg_h2_ppg, 'blue', markerfmt=" ")
            ax.stem(-avg_fh3f_ppg, avg_h3_ppg, 'blue', markerfmt=" ")
            ax.stem(-avg_fh4f_ppg, avg_h4_ppg, 'blue', markerfmt=" ")
            ax.stem(-avg_fh5f_ppg, avg_h5_ppg, 'blue', markerfmt=" ")
            ax.stem(-avg_fh6f_ppg, avg_h6_ppg, 'blue', markerfmt=" ")

            markerline, stemlines, baseline = plt.stem(
                avg_fh1f_ppg, avg_h1_ppg, 'blue', markerfmt='go', label='H1')
            markerline, stemlines, baseline = plt.stem(
                avg_fh2f_ppg, avg_h2_ppg, 'blue', markerfmt='yo', label='H2')
            markerline, stemlines, baseline = plt.stem(
                avg_fh3f_ppg, avg_h3_ppg, 'blue', markerfmt='o', label='H3')
            plt.legend()

            ax.stem(avg_fh4f_ppg, avg_h4_ppg, 'blue', markerfmt=" ")
            ax.stem(avg_fh5f_ppg, avg_h5_ppg, 'blue', markerfmt=" ")
            ax.stem(avg_fh6f_ppg, avg_h6_ppg, 'blue', markerfmt=" ")
            ax.set_title('PPG signal')
            ax.set_xlabel('Frequency[Hz]')
            ax.set_ylabel('Magnitude (Arb.Unit)')
            ax.set_xlim(-f_s / 2, f_s / 2)
            ax.set_ylim(-1, avg_h1_ppg+50)
            plt.show()
            return fig

        ##### Clear Data image ###
        shutil.rmtree(path_ppg)
        shutil.rmtree(path_spg)

        ###### Link Grap Display #######
        # spg_plotG = spg()
        # self.plotWidget2 = FigureCanvas(spg_plotG)
        # self.hl2.addWidget(self.plotWidget2)

        # ppg_plotG = ppg()
        # self.plotWidget3 = FigureCanvas(ppg_plotG)
        # self.hl3.addWidget(self.plotWidget3)

        fftspg_spg = fftas()
        self.plotWidget4 = FigureCanvas(fftspg_spg)
        self.hl4.addWidget(self.plotWidget4)

        fftppg = fftsp()
        self.plotWidget5 = FigureCanvas(fftppg)
        self.hl5.addWidget(self.plotWidget5)

        # spg_ppg = spg_and_ppg()
        # self.plotWidget6 = FigureCanvas(spg_ppg)
        # self.hl6.addWidget(self.plotWidget6)

        ######## Link Text Display ##############
        # SPG
        self.l2.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">SNR : %.3f dB</span></p></body></html>" % (SNR_spg))
        self.l3.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">HRV : %.3f s</span></p></body></html>" % (hrv_spg))
        self.l4.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">Heart rate : %d BPM</span></p></body></html>" % (bpm_spg))
        self.l5.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">Frequency : %.4f Hz</span></p></body></html>" % (fre_spg))

        self.l7.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H1 : %.4f</span></p></body></html>" % (avg_h1_spg))
        self.l8.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H2 : %.4f</span></p></body></html>" % (avg_h2_spg))
        self.l9.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H3 : %.4f</span></p></body></html>" % (avg_h3_spg))
        self.l10.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H3/H1 : %.4f</span></p></body></html>" % (avgH3H1_spg))
        self.l11.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H3/H2 : %.4f</span></p></body></html>" % (avgH3H2_spg))
        self.l12.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H2/H1 : %.4f</span></p></body></html>" % (avgH2H1_spg))
        # PPG
        self.r2.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">SNR : %.4f dB</span></p></body></html>" % (SNR_ppg))
        self.r3.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">HRV : %.4f s</span></p></body></html>" % (hrv_ppg))
        self.r4.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">Heart rate : %d BPM</span></p></body></html>" % (bpm_ppg))
        self.r5.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">Frequency : %.4f Hz</span></p></body></html>" % (fre_ppg))

        self.r7.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H1 : %.4f</span></p></body></html>" % (avg_h1_ppg))
        self.r8.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H2 : %.4f</span></p></body></html>" % (avg_h2_ppg))
        self.r9.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H3 : %.4f</span></p></body></html>" % (avg_h3_ppg))
        self.r10.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H3/H1 : %.4f</span></p></body></html>" % (avgH3H1_ppg))
        self.r11.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H3/H2 : %.4f</span></p></body></html>" % (avgH3H2_ppg))
        self.r12.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">H2/H1 : %.4f</span></p></body></html>" % (avgH2H1_ppg))

        self.lr1.setText(
            "<html><head/><body><p><span style=\" font-size:12pt; color:#ffffff;\">Average time delay between SPG and PPG signal is : %.4f s</span></p></body></html>" % (np.mean(tdTime)))

        def spg_and_ppg():
            fig, ax = plt.subplots()
            color1 = 'red'
            color2 = 'blue'
            ax.plot(data_spg, color=color1)
            ax.plot(data_ppg, color=color2)
            ax.set_title('SPG & PPG Signal')
            ax.set_xlabel('Frame (25 Frame / second)')
            ax.set_ylabel('Amplitude (Arb.Unit)')
            ax.set_ylim(ymin01, ymax01)
            ax.plot(peaks_of_spg, data_spg[peaks_of_spg], "o")
            ax.plot(peaks_of_ppg, data_ppg[peaks_of_ppg], "o")
            plt.legend(('SPG Signal', 'PPG Signal'), loc='best')
            plt.grid(True, which='both')
            plt.show()
            return fig

        spg_ppg = spg_and_ppg()
        self.plotWidget6 = FigureCanvas(spg_ppg)
        self.hl6.addWidget(self.plotWidget6)

        self.t2.setText(
            "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">Process complete</span></p></body></html>")


##### Show GUI ########
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
