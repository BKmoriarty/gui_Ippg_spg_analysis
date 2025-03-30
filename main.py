import sys
import os
import yaml
from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QFileDialog, QMessageBox, QWidget, QVBoxLayout
)
from PyQt5.QtCore import QLocale
from PyQt5.uic import loadUi

from harmonicAnalyzer import HarmonicAnalyzer
from main_full_ui import Ui_MainWindow
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtWidgets, QtCore, QtGui
import threading
from time import sleep
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from recordVideo import RecordVideo
from checkCamera import RecordingVideoPulse, check_pulse_is_connected
from SignalExtractionAndAnalysis import Analysis_PPG_SPG


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100, left=0.05, right=0.95, top=0.95, bottom=0.05, xlabel='', ylabel=''):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.grid(True, which='both')
        # Add these lines to make the plot fill the canvas
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        fig.tight_layout()
        fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
        super().__init__(fig)


def convert_frame_rate(str_fps):
    switcher = {
        "60 FPS": 60,
        "120 FPS": 120,
        "180 FPS": 180,
        "240 FPS": 240,
    }

    return switcher.get(str_fps, 120)


def convert_roi(str_roi):
    switcher = {
        "30x30 px": 30,
        "50x50 px": 50,
        "100x100 px": 100,
        "150x150 px": 150,
        "200x200 px": 200,
        "250x250 px": 250,
        "300x300 px": 300,
    }

    return switcher.get(str_roi, 200)


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None, ):
        super().__init__(parent)
        self.setupUi(self)

        self.setup_result_plt()
        self.setup_preview()
        self.setup_plots()

        self.connectSignalsSlots()

        self.record = RecordVideo(
            fps=convert_frame_rate(self.saved_frame_rate),
            exposure_time=int(self.saved_exposure_time),
            width=320,
            height=320,
            filename="VIDEO",
            time_record=int(self.saved_video_length)
        )

        self.record.change_pixmap_signal.connect(self.update_image)
        self.record.update_plot_signal.connect(self.update_plots)

        # Start the pulse checking thread
        self.pulse_thread = threading.Thread(
            target=self.check_pulse, daemon=True)
        self.pulse_thread.start()

        self.personal_name = ""
        self.personal_age = 0
        self.personal_gender = ""
        self.personal_weight = 0.0
        self.personal_height = 0
        self.personal_congenital_disease = False
        self.personal_non_congenital_disease = False
        self.personal_congenital_disease_detail = ""

        self.current_folder_show = ""
        self.current_data_show = []

    def check_pulse(self):
        while True:
            port = check_pulse_is_connected()
            if port:
                self.port.setText("Found")
                self.port.setStyleSheet("color: green;")
            else:
                self.port.setText("Not Found")
                self.port.setStyleSheet("color: red;")
            sleep(1)

    def closeEvent(self, event):
        # Ask for confirmation before closing
        confirmation = QMessageBox.question(
            self, "Confirmation", "Are you sure you want to close the application?", QMessageBox.Yes | QMessageBox.No)

        if confirmation == QMessageBox.Yes:
            event.accept()  # Close the app
            self.record.handle_close()
        else:
            event.ignore()  # Don't close the app

    def setup_preview(self):
        self.preview_group = QtWidgets.QGroupBox(self.centralwidget)
        self.preview_group.setGeometry(QtCore.QRect(540, -10, 1151, 691))
        self.preview_group.setTitle("")
        self.preview_group.setObjectName("preview_group")
        self.label_31 = QtWidgets.QLabel(self.preview_group)
        self.label_31.setGeometry(QtCore.QRect(0, 360, 1151, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_31.setFont(font)
        self.label_31.setFrameShape(QtWidgets.QFrame.Box)
        self.label_31.setAlignment(QtCore.Qt.AlignCenter)
        self.label_31.setObjectName("label_31")
        self.label_26 = QtWidgets.QLabel(self.preview_group)
        self.label_26.setGeometry(QtCore.QRect(0, 10, 301, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_26.setFont(font)
        self.label_26.setFrameShape(QtWidgets.QFrame.Box)
        self.label_26.setAlignment(QtCore.Qt.AlignCenter)
        self.label_26.setObjectName("label_26")
        self.label_27 = QtWidgets.QLabel(self.preview_group)
        self.label_27.setGeometry(QtCore.QRect(310, 10, 841, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_27.setFont(font)
        self.label_27.setFrameShape(QtWidgets.QFrame.Box)
        self.label_27.setAlignment(QtCore.Qt.AlignCenter)
        self.label_27.setObjectName("label_27")
        self.image_real_time = QtWidgets.QLabel(self.preview_group)
        self.image_real_time.setGeometry(QtCore.QRect(0, 50, 300, 300))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.image_real_time.setFont(font)
        self.image_real_time.setFrameShape(QtWidgets.QFrame.Box)
        self.image_real_time.setAlignment(QtCore.Qt.AlignCenter)
        self.image_real_time.setObjectName("image_real_time")
        self.freq_real_time = QtWidgets.QWidget(self.preview_group)
        self.freq_real_time.setGeometry(QtCore.QRect(310, 50, 1151, 301))
        self.freq_real_time.setObjectName("freq_real_time")
        self.ippg_real_time = QtWidgets.QWidget(self.preview_group)
        self.ippg_real_time.setGeometry(QtCore.QRect(0, 400, 841, 221))
        self.ippg_real_time.setObjectName("ippg_real_time")

        _translate = QtCore.QCoreApplication.translate
        self.label_31.setText(_translate("MainWindow", "iPPG Real time"))
        self.label_26.setText(_translate("MainWindow", "Image Real time"))
        self.label_27.setText(_translate(
            "MainWindow", "Spectrum iPPG Real time"))

    def connectSignalsSlots(self):
        self.bt_start.clicked.connect(self.open_personal)
        self.bt_load.clicked.connect(self.second_button)
        self.bt_load_2.clicked.connect(self.reload_button)

        self.bt_load_2.setEnabled(False)

        self.folder_name.mousePressEvent = self.open_file_explorer
        self.folder_name.setText(os.getcwd())

        self.personal_data.setText("")

        self.ippg_snr.setText("- dB")
        self.cppg_snr.setText("- dB")
        self.spg_snr.setText("- dB")

        self.ippg_hr.setText("- BPM")
        self.cppg_hr.setText("- BPM")
        self.spg_hr.setText("- BPM")

        self.ippg_freq.setText("- Hz")
        self.cppg_freq.setText("- Hz")
        self.spg_freq.setText("- Hz")

        self.ippg_freq_h2.setText("- Hz")
        self.cppg_freq_h2.setText("- Hz")
        self.spg_freq_h2.setText("- Hz")

        self.ippg_freq_h3.setText("- Hz")
        self.cppg_freq_h3.setText("- Hz")
        self.spg_freq_h3.setText("- Hz")

        self.saved_frame_rate = "120 FPS"
        self.saved_exposure_time = 8000
        self.saved_video_length = 30
        self.saved_size_ippg = "150x150 px"
        self.saved_size_spg = "150x150 px"

        self.frame_rate.setCurrentText(self.saved_frame_rate)
        self.exposure_time.setValue(self.saved_exposure_time)
        self.video_length.setValue(self.saved_video_length)
        self.size_ippg.setCurrentText(self.saved_size_ippg)
        self.size_spg.setCurrentText(self.saved_size_spg)

        self.apply_setting_camera.clicked.connect(self.apple_setting_camera)
        self.frame_rate.currentTextChanged.connect(self.save_settings)
        self.exposure_time.valueChanged.connect(self.save_settings)
        self.video_length.valueChanged.connect(self.save_settings)
        self.size_ippg.currentTextChanged.connect(self.save_settings)
        self.size_spg.currentTextChanged.connect(self.save_settings)

        self.onset.setText("")
        self.systolic.setText("")
        self.dicrotic.setText("")
        self.diastolic.setText("")
        self.complete.setText("")
        self.time_delay.setText("")
        self.ms.setText("")

        self.delta_t.setText("")
        self.si.setText("")
        self.ct.setText("")
        self.t_sys.setText("")
        self.t_dia.setText("")
        self.t_ratio.setText("")
        self.ipr.setText("")

        self.amp.setText("")
        self.ri.setText("")
        self.a1.setText("")
        self.a2.setText("")
        self.ipa.setText("")

        self.dw_75.setText("")
        self.dw_66.setText("")
        self.dw_50.setText("")
        self.dw_33.setText("")
        self.dw_25.setText("")
        self.dw_10.setText("")

        self.sw_75.setText("")
        self.sw_66.setText("")
        self.sw_50.setText("")
        self.sw_33.setText("")
        self.sw_25.setText("")
        self.sw_10.setText("")

        self.w_75.setText("")
        self.w_66.setText("")
        self.w_50.setText("")
        self.w_33.setText("")
        self.w_25.setText("")
        self.w_10.setText("")

        self.ds_75.setText("")
        self.ds_66.setText("")
        self.ds_50.setText("")
        self.ds_33.setText("")
        self.ds_25.setText("")
        self.ds_10.setText("")

        self.signal_ippg_show.setChecked(True)
        self.signal_cppg_show.setChecked(True)
        self.signal_spg_show.setChecked(True)

        self.freq_ippg_show.setChecked(True)
        self.freq_cppg_show.setChecked(True)
        self.freq_spg_show.setChecked(True)

        self.signal_ippg_show.toggled.connect(self.show_signal_all)
        self.signal_cppg_show.toggled.connect(self.show_signal_all)
        self.signal_spg_show.toggled.connect(self.show_signal_all)

        self.freq_ippg_show.toggled.connect(self.show_freq_all)
        self.freq_cppg_show.toggled.connect(self.show_freq_all)
        self.freq_spg_show.toggled.connect(self.show_freq_all)

        self.time_start.valueChanged.connect(self.show_signal_all)
        self.time_end.valueChanged.connect(self.show_signal_all)

        self.freq_start.valueChanged.connect(self.show_freq_all)
        self.freq_end.valueChanged.connect(self.show_freq_all)

        self.preview_group.setVisible(True)
        self.result_group.setVisible(False)

    def open_file_explorer(self, *arg, **kwargs):
        os.startfile(self.folder_name.text())

    def show_signal_all(self):
        ppg, spg, excel = self.current_data_show

        self.signal_canvas.axes.clear()
        self.signal_canvas.axes.grid(True, which='both')
        if self.signal_ippg_show.isChecked():
            self.signal_canvas.axes.plot(ppg[0], ppg[1], color='b')
            self.signal_canvas.axes.plot(ppg[0][ppg[2]], ppg[1][ppg[2]],
                                         color='b', marker='o', linestyle='')
        if self.signal_cppg_show.isChecked():
            self.signal_canvas.axes.plot(excel[0], excel[1], color='r')
            self.signal_canvas.axes.plot(excel[0][excel[2]], excel[1][excel[2]],
                                         color='r', marker='o', linestyle='')
        if self.signal_spg_show.isChecked():
            self.signal_canvas.axes.plot(spg[0], spg[1], color='g')
            self.signal_canvas.axes.plot(spg[0][spg[2]], spg[1][spg[2]],
                                         color='g', marker='o', linestyle='')
        self.signal_canvas.axes.set_xlabel("Time (s)")
        self.signal_canvas.axes.set_ylabel("Amplitude")
        self.signal_canvas.axes.set_xlim(
            [self.time_start.value(), self.time_end.value()])
        self.signal_canvas.draw()

    def show_freq_all(self):
        ppg, spg, excel = self.current_data_show

        self.freq_result_canvas.axes.clear()
        self.freq_result_canvas.axes.grid(True, which='both')
        if self.freq_ippg_show.isChecked():
            self.freq_result_canvas.axes.plot(ppg[3], ppg[4], color='b')
            for name, (peak_freq, peak_mag) in ppg[5].items():
                self.freq_result_canvas.axes.plot(
                    peak_freq, peak_mag, 'o', color='b')
        if self.freq_cppg_show.isChecked():
            self.freq_result_canvas.axes.plot(excel[3], excel[4], color='r')
            for name, (peak_freq, peak_mag) in excel[5].items():
                self.freq_result_canvas.axes.plot(
                    peak_freq, peak_mag, 'o', color='r')
        if self.freq_spg_show.isChecked():
            self.freq_result_canvas.axes.plot(spg[3], spg[4], color='g')
            for name, (peak_freq, peak_mag) in spg[5].items():
                self.freq_result_canvas.axes.plot(
                    peak_freq, peak_mag, 'o', color='g')
        self.freq_result_canvas.axes.set_xlim([0, 5])
        self.freq_result_canvas.axes.set_ylim(
            [0, max(max(excel[4] if self.freq_cppg_show.isChecked() else [0]), max(ppg[4] if self.freq_ippg_show.isChecked() else [0]), max(spg[4] if self.freq_spg_show.isChecked() else [0]))*1.1,])
        self.freq_result_canvas.axes.set_xlabel("Frequency (Hz)")
        self.freq_result_canvas.axes.set_ylabel("Amplitude")
        self.freq_result_canvas.axes.set_xlim(
            [self.freq_start.value(), self.freq_end.value()])
        self.freq_result_canvas.draw()

    def second_button(self):
        if self.bt_load.text() == "ยกเลิก":
            self.record.handle_close()
            self.frame_rate.setEnabled(True)
            self.bt_load.setText("โหลด")
            self.bt_load.setIcon(QIcon('ui/resources/folder.png'))

            # clear output
            self.time_canvas.axes.clear()
            self.time_canvas.axes.grid(True, which='both')
            self.time_canvas.draw()

            self.freq_canvas.axes.clear()
            self.freq_canvas.axes.grid(True, which='both')
            self.freq_canvas.axes.grid(
                True, which='both', linestyle='--', linewidth=0.5)
            self.freq_canvas.axes.axvline(x=0.8, color='r', linestyle='--',
                                          label='Lower Bound (0.8 Hz)')
            self.freq_canvas.axes.axvline(
                x=4, color='g', linestyle='--', label='Upper Bound (4 Hz)')
            self.freq_canvas.axes.set_ylim(0, 300)
            self.freq_canvas.draw()

            # self.ippg_hr.setText("- BPM")
            # self.ippg_freq.setText("- Hz")
            # self.ippg_snr.setText("- dB")

            self.bt_start.setText("เริ่ม")

        elif self.bt_load.text() == "โหลด":
            file = str(QFileDialog.getExistingDirectory(
                self, "Select Directory"))
            self.folder_name.setText(file)
            self.folder_name.setToolTip(file)

            self.check_is_valid_folder(file)

    def check_is_valid_folder(self, folder):
        required_folders = ["ippg", "spg", "cppg"]
        config_file = "config.yml"

        for folder_name in required_folders:
            folder_path = os.path.join(folder, folder_name)
            if not os.path.isdir(folder_path):
                QMessageBox.warning(
                    self, "Warning", f"Folder '{folder_name}' is missing in the selected directory.")
                return

        config_path = os.path.join(folder, config_file)
        if not os.path.isfile(config_path):
            QMessageBox.warning(
                self, "Warning", f"File '{config_file}' is missing in the selected directory.")
            return

        with open(f"{folder}/config.yml", "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        # QMessageBox.information(
        #     self, "Success", "The selected directory is valid.")

        ppg = [None] * 6

        ppg[0] = np.load(f"{folder}/ippg/ippg_filtered_time.npy")
        ppg[1] = np.load(f"{folder}/ippg/ippg_filtered_amp.npy")
        ppg[2] = np.load(f"{folder}/ippg/ippg_filtered_peak.npy")
        ppg[3] = np.load(f"{folder}/ippg/filtered_ippg_fft_freq.npy")
        ppg[4] = np.load(f"{folder}/ippg/filtered_ippg_fft_amp.npy")
        ppg[5] = np.load(
            f"{folder}/ippg/filtered_ippg_fft_peak.npy", allow_pickle=True).item()

        ppg[5] = dict(ppg[5])

        spg = [None] * 6

        spg[0] = np.load(f"{folder}/spg/spg_filtered_spg_time.npy")
        spg[1] = np.load(f"{folder}/spg/spg_filtered_spg_amp.npy")
        spg[2] = np.load(f"{folder}/spg/spg_filtered_spg_peak.npy")
        spg[3] = np.load(f"{folder}/spg/filtered_spg_fft_freq.npy")
        spg[4] = np.load(f"{folder}/spg/filtered_spg_fft_amp.npy")
        spg[5] = np.load(
            f"{folder}/spg/filtered_spg_fft_peak.npy", allow_pickle=True).item()

        spg[5] = dict(spg[5])

        excel = [None] * 6

        excel[0] = np.load(f"{folder}/cppg/cppg_filtered_excel_time.npy")
        excel[1] = np.load(f"{folder}/cppg/cppg_filtered_excel_amp.npy")
        excel[2] = np.load(f"{folder}/cppg/cppg_filtered_excel_peak.npy")
        excel[3] = np.load(f"{folder}/cppg/filtered_excel_fft_freq.npy")
        excel[4] = np.load(f"{folder}/cppg/filtered_excel_fft_amp.npy")
        excel[5] = np.load(
            f"{folder}/cppg/filtered_excel_fft_peak.npy", allow_pickle=True).item()

        excel[5] = dict(excel[5])

        self.current_data_show = [ppg, spg, excel]

        # show data
        self.preview_group.setVisible(False)
        self.result_group.setVisible(True)

        self.signal_plt.setVisible(True)
        self.freq_plt.setVisible(True)

        self.signal_canvas.axes.clear()
        self.signal_canvas.axes.grid(True, which='both')
        self.signal_canvas.axes.plot(excel[0], excel[1], color='r')
        self.signal_canvas.axes.plot(excel[0][excel[2]], excel[1][excel[2]],
                                     color='r', marker='o', linestyle='')
        self.signal_canvas.axes.plot(ppg[0], ppg[1], color='b')
        self.signal_canvas.axes.plot(ppg[0][ppg[2]], ppg[1][ppg[2]],
                                     color='b', marker='o', linestyle='')
        self.signal_canvas.axes.plot(spg[0], spg[1], color='g')
        self.signal_canvas.axes.plot(spg[0][spg[2]], spg[1][spg[2]],
                                     color='g', marker='o', linestyle='')
        self.signal_canvas.axes.set_xlabel("Time (s)")
        self.signal_canvas.axes.set_ylabel("Amplitude")
        self.signal_canvas.draw()

        self.freq_result_canvas.axes.clear()
        self.freq_result_canvas.axes.grid(True, which='both')
        self.freq_result_canvas.axes.plot(excel[3], excel[4], color='r')
        for name, (peak_freq, peak_mag) in excel[5].items():
            self.freq_result_canvas.axes.plot(
                peak_freq, peak_mag, 'o', color='r')
            self.freq_result_canvas.axes.text(
                peak_freq, peak_mag, "TEST")
        self.freq_result_canvas.axes.plot(ppg[3], ppg[4], color='b')
        for name, (peak_freq, peak_mag) in ppg[5].items():
            self.freq_result_canvas.axes.plot(
                peak_freq, peak_mag, 'o', color='b')
        self.freq_result_canvas.axes.plot(spg[3], spg[4], color='g')
        for name, (peak_freq, peak_mag) in spg[5].items():
            self.freq_result_canvas.axes.plot(
                peak_freq, peak_mag, 'o', color='g')
        self.freq_result_canvas.axes.set_xlim([0, 5])
        self.freq_result_canvas.axes.set_ylim(
            [0, max(max(excel[3]), max(ppg[3]), max(spg[3]))*1.1,])
        self.freq_result_canvas.axes.set_xlabel("Frequency (Hz)")
        self.freq_result_canvas.axes.set_ylabel("Amplitude")
        self.freq_result_canvas.draw()

        self.ippg_hr.setText(
            f"{round(config['result']['Heart Rate iPPG'],2)} BPM")
        self.ippg_freq.setText(
            f"{round(config['result']['Mag H1 iPPG'],4)}")
        self.ippg_freq_h2.setText(
            f"{round(config['result']['Mag H2 iPPG'],4)}")
        self.ippg_freq_h3.setText(
            f"{round(config['result']['Mag H3 iPPG'],4)}")
        self.ippg_snr.setText(f"{round(config['result']['SNR iPPG'],4)} dB")

        self.spg_hr.setText(
            f"{round(config['result']['Heart Rate SPG'],2)} BPM")
        self.spg_freq.setText(
            f"{round(config['result']['Mag H1 SPG'],4)}")
        self.spg_freq_h2.setText(
            f"{round(config['result']['Mag H2 SPG'],4)}")
        self.spg_freq_h3.setText(
            f"{round(config['result']['Mag H3 SPG'],4)}")
        self.spg_snr.setText(f"{round(config['result']['SNR SPG'],4)} dB")

        self.cppg_hr.setText(
            f"{round(config['result']['Heart Rate cPPG'],2)} BPM")
        self.cppg_freq.setText(
            f"{round(config['result']['Mag H1 cPPG'],4)}")
        self.cppg_freq_h2.setText(
            f"{round(config['result']['Mag H2 cPPG'],4)}")
        self.cppg_freq_h3.setText(
            f"{round(config['result']['Mag H3 cPPG'],4)}")
        self.cppg_snr.setText(f"{round(config['result']['SNR cPPG'],4)} dB")

        self.time_delay.setText(
            f"{round(config['result']['Average Time Delay'],4)} S.")

        self.bt_load_2.setEnabled(True)

        self.personal_name = config['personal']['name']
        self.personal_age = config['personal']['age']
        self.personal_gender = config['personal']['gender']
        self.personal_weight = config['personal']['weight']
        self.personal_height = config['personal']['height']
        self.personal_congenital_disease = config['personal']['congenital_disease']
        self.personal_non_congenital_disease = not config['personal']['congenital_disease']
        self.personal_congenital_disease_detail = config['personal']['congenital_disease_detail']

        bmi = self.personal_weight / \
            ((self.personal_height / 100) ** 2)
        self.personal_data.setText(
            f"อายุ {self.personal_age} ปี น้ำหนัก {self.personal_weight} กิโลกรัม ส่วนสูง {self.personal_height} BMI {round(bmi, 2)}")
        self.personal_name = self.personal_name

        self.saved_frame_rate = f"{config['config']['fps']} FPS"
        self.saved_exposure_time = config['config']['exposure_time']
        self.saved_video_length = config['config']['time_record']
        self.saved_size_ippg = f"{config['config']['size_ppg']}x{config['config']['size_ppg']} px"
        self.saved_size_spg = f"{config['config']['size_spg']}x{config['config']['size_spg']} px"

        self.frame_rate.setCurrentText(self.saved_frame_rate)
        self.exposure_time.setValue(self.saved_exposure_time)
        self.video_length.setValue(self.saved_video_length)
        self.size_ippg.setCurrentText(self.saved_size_ippg)
        self.size_spg.setCurrentText(self.saved_size_spg)

        self.time_end.setValue(self.saved_video_length)
        self.time_end.setMaximum(self.saved_video_length)
        self.freq_end.setValue(5)
        self.freq_end.setMaximum(5)

        self.onset.setText(f"{config['result']['onset']}")
        self.systolic.setText(f"{config['result']['systolic']}")
        self.dicrotic.setText(f"{config['result']['dicrotic']}")
        self.diastolic.setText(f"{config['result']['diastolic']}")
        self.complete.setText(f"{config['result']['complete patterns']}")
        self.ms.setText(f"{config['result']['Average maximum slope']}")

        self.delta_t.setText(f"{config['result']['Average (t_dia - t_sys)']}")
        self.si.setText(
            f"{config['result']['Stiffness Index h/(t_dia - t_sys)']}")
        self.ct.setText(
            f"{config['result']['Average Crest Time (t_sys - t_0)']}")
        self.t_sys.setText(
            f"{config['result']['Average t_sys (t(dic)-t(0))']}")
        self.t_dia.setText(
            f"{config['result']['Average t_dia (t(0)-t(dic))']}")
        self.t_ratio.setText(
            f"{config['result']['t_ratio (t_sys-t(0)) / (t_dia-t_dic)']}")
        self.ipr.setText(f"{config['result']['Average IPR']}")

        self.amp.setText(f"{config['result']['Average pulse amplitude']}")
        self.ri.setText(f"{config['result']['Average reflection index']}")
        self.a1.setText(f"{config['result']['Average systolic area']}")
        self.a2.setText(f"{config['result']['Average diastolic area']}")
        self.ipa.setText(f"{config['result']['IPA inflection point']}")

        self.dw_75.setText(f"{config['result']['Average dw_75']}")
        self.dw_66.setText(f"{config['result']['Average dw_66']}")
        self.dw_50.setText(f"{config['result']['Average dw_50']}")
        self.dw_33.setText(f"{config['result']['Average dw_33']}")
        self.dw_25.setText(f"{config['result']['Average dw_25']}")
        self.dw_10.setText(f"{config['result']['Average dw_10']}")

        self.sw_75.setText(f"{config['result']['Average sw_75']}")
        self.sw_66.setText(f"{config['result']['Average sw_66']}")
        self.sw_50.setText(f"{config['result']['Average sw_50']}")
        self.sw_33.setText(f"{config['result']['Average sw_33']}")
        self.sw_25.setText(f"{config['result']['Average sw_25']}")
        self.sw_10.setText(f"{config['result']['Average sw_10']}")

        self.w_75.setText(f"{config['result']['Average w_75']}")
        self.w_66.setText(f"{config['result']['Average w_66']}")
        self.w_50.setText(f"{config['result']['Average w_50']}")
        self.w_33.setText(f"{config['result']['Average w_33']}")
        self.w_25.setText(f"{config['result']['Average w_25']}")
        self.w_10.setText(f"{config['result']['Average w_10']}")

        self.ds_75.setText(f"{config['result']['Average dw_75/sw_75']}")
        self.ds_66.setText(f"{config['result']['Average dw_66/sw_66']}")
        self.ds_50.setText(f"{config['result']['Average dw_50/sw_50']}")
        self.ds_33.setText(f"{config['result']['Average dw_33/sw_33']}")
        self.ds_25.setText(f"{config['result']['Average dw_25/sw_25']}")
        self.ds_10.setText(f"{config['result']['Average dw_10/sw_10']}")

        self.current_folder_show = folder

    def apple_setting_camera(self):
        self.record.change_exposure_time(self.saved_exposure_time)
        self.record.change_size_ippg(convert_roi(self.saved_size_ippg))

    def save_settings(self):
        self.saved_frame_rate = self.frame_rate.currentText()
        self.saved_exposure_time = self.exposure_time.value()
        self.saved_video_length = self.video_length.value()
        self.saved_size_ippg = self.size_ippg.currentText()
        self.saved_size_spg = self.size_spg.currentText()

    def run_recording(self):
        video_path, serial_path = f"{self.folder_name.text()}/video-0000.avi", f"{self.folder_name.text()}/serial.xlsx"
        personal = {
            "name": self.personal_name,
            "age": self.personal_age,
            "gender": self.personal_gender,
            "weight": self.personal_weight,
            "height": self.personal_height,
            "congenital_disease": self.personal_congenital_disease,
            "congenital_disease_detail": self.personal_congenital_disease_detail
        }
        analysis = Analysis_PPG_SPG(
            video_path, serial_path, convert_roi(self.saved_size_ippg), convert_roi(self.saved_size_spg), self.saved_exposure_time, convert_frame_rate(self.saved_frame_rate), time_record=self.saved_video_length, personal=personal)
        ppg, spg, excel,  avg_time_delay = analysis.main()

        self.check_is_valid_folder(self.folder_name.text())
        return ppg, spg, excel, avg_time_delay

    def reload_button(self):
        recording_thread = threading.Thread(
            target=self.run_recording, daemon=True)
        recording_thread.start()

    def open_personal(self):
        self.time_delay.setText("")

        if self.bt_start.text() == "เริ่ม":
            personal = {
                "name": self.personal_name,
                "age": self.personal_age,
                "gender": self.personal_gender,
                "weight": self.personal_weight,
                "height": self.personal_height,
                "congenital_disease": self.personal_congenital_disease,
                "congenital_disease_detail": self.personal_congenital_disease_detail
            }

            dialog = Personal_UI(self, personal)
            dialog.exec()

            if dialog.state_btn == 'accepted':
                self.personal_name = dialog.personal_name
                self.personal_age = dialog.personal_age
                self.personal_gender = dialog.personal_gender
                self.personal_weight = dialog.personal_weight
                self.personal_height = dialog.personal_height
                self.personal_congenital_disease = dialog.personal_congenital_disease
                self.personal_non_congenital_disease = dialog.personal_non_congenital_disease
                self.personal_congenital_disease_detail = dialog.personal_congenital_disease_detail

                bmi = dialog.personal_weight / \
                    ((dialog.personal_height / 100) ** 2)
                self.personal_data.setText(
                    f"อายุ {dialog.personal_age} ปี น้ำหนัก {dialog.personal_weight} กิโลกรัม ส่วนสูง {dialog.personal_height} BMI {round(bmi, 2)}")
                self.personal_name = dialog.personal_name

                # start
                self.frame_rate.setEnabled(False)
                self.bt_load.setText("ยกเลิก")
                self.bt_load.setIcon(QIcon('ui/resources/close.png'))

                self.signal_plt.setVisible(False)
                self.freq_plt.setVisible(False)

                self.preview_group.setVisible(True)

                self.bt_start.setText("บันทึก")
                self.start_preview()

        elif self.bt_start.text() == "บันทึก":
            self.record.handle_close()

            # START
            port = check_pulse_is_connected()
            if port is None:
                QMessageBox.warning(
                    self, "Warning", "No pulse device found. Please connect a pulse device and try again.")
                return

            self.recordingVideoPulse = RecordingVideoPulse(
                port_name=port,
                exposure_time=int(self.saved_exposure_time),
                name=self.personal_name,
                fps=convert_frame_rate(self.saved_frame_rate),
                size=(320, 320),
                length=int(self.saved_video_length)
            )

            countdown = Countdown_UI(self, self.saved_video_length)

            def run_recording():
                video_path, serial_path = self.recordingVideoPulse.run()
                self.recordingVideoPulse.handle_close()
                # video_path, serial_path = "storage/2025-02-13 18_26_19 tee 8000 120 (320, 320)/video-0000.avi", "storage/2025-02-13 18_26_19 tee 8000 120 (320, 320)/serial.xlsx"

                # clear screen to show result
                self.preview_group.setVisible(False)
                self.result_group.setVisible(True)

                self.signal_plt.setVisible(True)
                self.freq_plt.setVisible(True)

                personal = {
                    "name": self.personal_name,
                    "age": self.personal_age,
                    "gender": self.personal_gender,
                    "weight": self.personal_weight,
                    "height": self.personal_height,
                    "congenital_disease": self.personal_congenital_disease,
                    "congenital_disease_detail": self.personal_congenital_disease_detail
                }

                print(personal)

                countdown.setTextDialog("กำลัง\nวิเคราะห์\nวิดีโอ")
                analysis = Analysis_PPG_SPG(
                    video_path, serial_path, convert_roi(self.saved_size_ippg), convert_roi(self.saved_size_spg), self.recordingVideoPulse.ExposureTime, self.recordingVideoPulse.fps, time_record=self.saved_video_length, personal=personal)
                ppg, spg, excel, avg_time_delay = analysis.main()

                countdown.close_pop_up()
                return ppg, spg, excel,  avg_time_delay

            self.recording_result = None
            recording_thread = threading.Thread(
                target=lambda: setattr(self, 'recording_result', run_recording()), daemon=True)
            recording_thread.start()

            countdown.exec()

            video_path, serial_path = self.recordingVideoPulse.get_paths()

            folder = "/".join(video_path.split("/")[:-1])
            self.folder_name.setText(folder)
            self.folder_name.setToolTip(folder)

            self.check_is_valid_folder(folder)

            self.bt_start.setText('เริ่ม')

            self.frame_rate.setEnabled(True)
            self.bt_load.setText("โหลด")
            self.bt_load.setIcon(QIcon('ui/resources/folder.png'))

            self.bt_load_2.setEnabled(True)

    def setup_result_plt(self):

        # signal
        self.plot_signal_widget = QWidget(self.result_group)
        self.plot_signal_widget.setGeometry(QtCore.QRect(0, 50, 1151, 300))
        self.plot_signal_layout = QVBoxLayout(self.plot_signal_widget)
        self.plot_signal_layout.setContentsMargins(
            0, 0, 0, 0)

        self.signal_canvas = MplCanvas(
            self, left=0.07, right=0.97, top=0.95, bottom=0.15, xlabel="Time (s)", ylabel="Amplitude")
        self.plot_signal_layout.addWidget(self.signal_canvas)

        # freq ippg
        self.plt_freq_widget = QWidget(self.result_group)
        self.plt_freq_widget.setGeometry(QtCore.QRect(0, 400, 1151, 300))
        self.plot_freq_layout = QVBoxLayout(self.plt_freq_widget)
        self.plot_freq_layout.setContentsMargins(
            0, 0, 0, 0)

        self.freq_result_canvas = MplCanvas(
            self, left=0.07, right=0.97, top=0.95, bottom=0.2, xlabel="Frequency (Hz)", ylabel="Amplitude")
        self.plot_freq_layout.addWidget(self.freq_result_canvas)

        # mock test

    def start_preview(self):
        print("start preview")
        self.record.init_camera()
        record_thread = threading.Thread(
            target=self.record.display_images, daemon=True)
        record_thread.start()

    def get_center_crop_position(self, w, h, crop_size):
        height, width = w, h

        # Ensure crop size is not larger than the image
        crop_size = min(crop_size, min(height, width))

        # Calculate the center of the image
        center_x, center_y = width // 2, height // 2

        # Calculate the top-left and bottom-right corners of the square
        x1 = center_x - crop_size // 2
        y1 = center_y - crop_size // 2
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        return (x1, y1, x2, y2)

    def update_image(self, cv_img):
        # Convert the image to QPixmap
        qt_img = QPixmap.fromImage(cv_img)

        # Create a QPainter object to draw on the image
        painter = QtGui.QPainter(qt_img)
        painter.setPen(QtGui.QPen(QtCore.Qt.red, 3, QtCore.Qt.SolidLine))

        # Draw a rectangle (square) on the image
        size_ippg = convert_roi(self.saved_size_ippg)  # one int ex 30
        size_spg = convert_roi(self.saved_size_spg)  # one int ex 30
        height, width = cv_img.height(), cv_img.width()
        x1_ppg, y1_ppg, x2_ppg, y2_ppg = self.get_center_crop_position(
            height, width, size_ippg)
        x1_spg, y1_spg, x2_spg, y2_spg = self.get_center_crop_position(
            height, width, size_spg)
        # print(x1_ppg, y1_ppg, x2_ppg, y2_ppg)
        rect_ippg = QtCore.QRect(
            x1_ppg, y1_ppg, x2_ppg-x1_ppg, y2_ppg-y1_ppg)  # x, y, width, height
        rect_spg = QtCore.QRect(x1_spg, y1_spg, x2_spg-x1_spg,
                                y2_spg-y1_spg)  # x, y, width, height

        # Draw rectangles with different colors
        painter.setPen(QtGui.QPen(QtCore.Qt.red, 3, QtCore.Qt.SolidLine))
        painter.drawRect(rect_ippg)
        painter.setPen(QtGui.QPen(QtCore.Qt.blue, 3, QtCore.Qt.SolidLine))
        painter.drawRect(rect_spg)

        # End the painter
        painter.end()

        # Set the modified image to the label
        self.image_real_time.setPixmap(qt_img)

    def setup_plots(self):
        self.plot_widget = QWidget(self.preview_group)
        self.plot_widget.setGeometry(QtCore.QRect(0, 400, 1151, 290))
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.plot_layout.setContentsMargins(
            0, 0, 0, 0)  # Remove layout margins

        self.time_canvas = MplCanvas(
            self, left=0.08, right=0.97, top=0.95, bottom=0.18, xlabel="Time (s)", ylabel="Amplitude")
        self.plot_layout.addWidget(self.time_canvas)

        self.plot_widget_freq = QWidget(self.preview_group)
        self.plot_widget_freq.setGeometry(QtCore.QRect(310, 50, 841, 301))
        self.plot_layout_freq = QVBoxLayout(self.plot_widget_freq)
        self.plot_layout_freq.setContentsMargins(
            0, 0, 0, 0)  # Remove layout margins

        self.freq_canvas = MplCanvas(
            self, left=0.08, right=0.95, top=0.95, bottom=0.18, xlabel="Frequency (Hz)", ylabel="Amplitude")
        self.plot_layout_freq.addWidget(self.freq_canvas)

    def update_plots(self, time_data, freq_data):
        self.time_canvas.axes.clear()
        self.time_canvas.axes.grid(True, which='both')
        self.time_canvas.axes.plot(time_data, color='b')
        self.time_canvas.axes.set_xlabel("Time (s)")
        self.time_canvas.axes.set_ylabel("Amplitude")
        self.time_canvas.draw()

        self.freq_canvas.axes.clear()
        self.freq_canvas.axes.grid(True, which='both')
        self.freq_canvas.axes.plot(freq_data[0], freq_data[1], color='r')
        self.freq_canvas.axes.grid(
            True, which='both', linestyle='--', linewidth=0.5)
        self.freq_canvas.axes.axvline(x=0.8, color='r', linestyle='--',
                                      label='Lower Bound (0.8 Hz)')
        self.freq_canvas.axes.axvline(
            x=4, color='g', linestyle='--', label='Upper Bound (4 Hz)')
        self.freq_canvas.axes.set_xlabel("Frequency (Hz)")
        self.freq_canvas.axes.set_ylabel("Amplitude")
        self.freq_canvas.axes.set_xlim(0, min(10, max(freq_data[0])))
        self.freq_canvas.axes.set_ylim(0, 300)
        self.freq_canvas.draw()

        self.ippg_hr.setText(f"{round(freq_data[2]*60,2)} BPM")
        self.ippg_freq.setText(f"{round(freq_data[2],4)} Hz")
        self.ippg_snr.setText(f"{round(freq_data[3],4)} dB")


class Personal_UI(QDialog):
    def __init__(self, parent=None, personal={}):
        super().__init__(parent)
        loadUi("ui/personal.ui", self)

        self.personal_name = personal['name']
        self.personal_age = personal['age']
        self.personal_gender = personal['gender']
        self.personal_weight = personal['weight']
        self.personal_height = personal['height']
        self.personal_congenital_disease = personal['congenital_disease']
        self.personal_non_congenital_disease = not personal['congenital_disease']
        self.personal_congenital_disease_detail = personal['congenital_disease_detail']

        self.per_name.setText(self.personal_name)
        self.per_age.setValue(self.personal_age)
        self.per_gender.setCurrentText(self.personal_gender)
        self.per_weight_2.setValue(int(self.personal_weight))
        self.per_height.setValue(self.personal_height)
        self.per_congenital_disease.setChecked(
            self.personal_congenital_disease)
        self.per_non_congenital_disease.setChecked(
            self.personal_non_congenital_disease)
        self.per_congenital_disease_detail.setText(
            self.personal_congenital_disease_detail)

        self.state_btn = ""

        self.bt_personal.accepted.connect(self.open_personal)
        self.bt_personal.rejected.connect(self.close_personal)
        self.per_gender.currentTextChanged.connect(self.personal_gender_change)
        self.per_congenital_disease.setChecked(False)
        self.per_non_congenital_disease.setChecked(True)
        self.per_non_congenital_disease.toggled.connect(
            self.toggle_congenital_disease)
        self.per_congenital_disease_detail.setEnabled(False)

    def close_personal(self):
        self.state_btn = "rejected"
        self.close
        self.reject()

    def open_personal(self):
        self.personal_name = self.per_name.text()
        self.personal_age = int(self.per_age.text())
        self.personal_gender = self.per_gender.currentText()
        self.personal_weight = float(self.per_weight_2.text())
        self.personal_height = int(self.per_height.text())
        self.personal_congenital_disease = self.per_congenital_disease.isChecked()
        self.personal_non_congenital_disease = self.per_non_congenital_disease.isChecked()
        self.personal_congenital_disease_detail = self.per_congenital_disease_detail.text()

        # print(self.personal_name)
        # print(self.personal_age)
        # print(self.personal_gender)
        # print(self.personal_weight)
        # print(self.personal_height)
        # print(self.personal_congenital_disease)
        # print(self.personal_non_congenital_disease)
        # print(self.personal_congenital_disease_detail)

        if not self.personal_name or not self.personal_age or not self.personal_gender or not self.personal_weight or not self.personal_height:
            QMessageBox.warning(self, "Input Error",
                                "All fields must be filled out before saving.")
            return

        # Proceed with saving the data
        self.state_btn = 'accepted'
        self.accept()

    def personal_gender_change(self, s):
        self.personal_gender = s

    def toggle_congenital_disease(self):
        if self.per_non_congenital_disease.isChecked() == True:
            self.per_congenital_disease_detail.setEnabled(False)
        else:
            self.per_congenital_disease_detail.setEnabled(True)


class Countdown_UI(QDialog):
    def __init__(self, parent=None, video_length=30):
        super().__init__(parent)
        self.video_length = video_length
        loadUi("ui/time_count.ui", self)

        self.countdown_thread = threading.Thread(
            target=self.run_countdown, daemon=True)
        self.countdown_thread.start()

    def run_countdown(self):
        for i in range(self.video_length):
            self.countdown.setText(f"เหลืออีก\n{self.video_length-i} \nวินาที")
            sleep(1)

        self.countdown.setText("เสร็จสิ้น")
        sleep(1)
        self.countdown.setText("กำลัง\nบันทึกข้อมูล")

    def setTextDialog(self, message):
        self.countdown.setText(message)

    def close_pop_up(self):
        self.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))
    win = Window()
    win.show()
    sys.exit(app.exec())
