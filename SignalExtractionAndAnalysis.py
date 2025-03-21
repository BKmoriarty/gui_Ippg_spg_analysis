import torch
from time import process_time
import datetime
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.widgets import MultiCursor
import cv2
import os
import yaml
from harmonicAnalyzer import HarmonicAnalyzer
from pressureAnalyzer import PressureAnalyzer
from save_excel import DataExporter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ImageCropper:
    def __init__(self, image):
        self.cropping = False
        self.x_start, self.y_start, self.x_end, self.y_end = 0, 0, 0, 0
        self.image = image
        self.oriImage = self.image.copy()

    def mouse_crop(self, event, x, y, flags, param):
        # grab references to the global variables
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x_start, self.y_start, self.x_end, self.y_end = x, y, x, y
            self.cropping = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.cropping:
                self.x_end, self.y_end = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.x_end, self.y_end = x, y
            self.cropping = False
            refPoint = [(self.x_start, self.y_start), (self.x_end, self.y_end)]
            if len(refPoint) == 2:
                roi = self.oriImage[refPoint[0][1]:refPoint[1]
                                    [1], refPoint[0][0]:refPoint[1][0]]
                print('Size', self.x_end - self.x_start,
                      self.y_end - self.y_start)
                cv2.imshow("Cropped", roi)

    def start_cropping(self):
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.mouse_crop)
        while True:
            i = self.image.copy()
            if not self.cropping:
                cv2.imshow("image", self.image)
            elif self.cropping:
                cv2.rectangle(i, (self.x_start, self.y_start),
                              (self.x_end, self.y_end), (255, 0, 0), 2)
                cv2.imshow("image", i)
            # if enter is pressed, break from the loop
            # and pass values x_start, y_start, x_end, y_end to the main function
            if cv2.waitKey(1) & 0xFF == 13:
                break
        cv2.destroyAllWindows()
        return self.x_start, self.y_start, self.x_end, self.y_end


class Analysis_PPG_SPG:

    def __init__(self, video_path, excel_path, size_ppg, size_spg, exposure_time=2200, fps=30, cache=False, cut_time_delay=0.2, time_record=30, personal={}):
        self.size_ppg = size_ppg
        self.size_spg = size_spg
        self.video_path = video_path
        self.dir_path = video_path.split('/')[1]
        self.excel_path = excel_path
        self.size_block = 5
        self.exposure_time = exposure_time
        self.fps = fps
        self.cache = cache
        self.cut_time_delay = cut_time_delay
        self.time_record = time_record
        self.personal = personal

    def __del__(self):
        # cv2.destroyAllWindows()
        # plt.close('all')
        pass

    def detect_finger_center(self, frame):
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise and improve contour detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Thresholding to create a binary image (finger is assumed to be the brightest part)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If contours are detected, find the largest one (assuming it's the finger)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Calculate the center of the bounding box
            center_x = x + w // 2
            center_y = y + h // 2

            return (center_x, center_y, w, h)
        else:
            return None

    def printProgressBar(self, iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                         (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
        # Print New Line on Complete
        if iteration == total:
            print()

    def crop_center(self, frame, crop_width, crop_height):
        height, width, _ = frame.shape
        start_x = width // 2 - crop_width // 2
        start_y = height // 2 - crop_height // 2
        return start_x, start_y, crop_width, crop_height

    def get_center_crop_position(self, image_shape, crop_size):
        height, width = image_shape[:2]

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

    def cal_contrast(self, frame):
        # Calculate the contrast of the frame
        shape = frame.shape

        return np.array([[((frame[i:i+self.size_block, j:j+self.size_block]).std() / np.mean(frame[i:i+self.size_block, j:j+self.size_block]))
                          for j in range(0, shape[1]-self.size_block+1, self.size_block)]
                         for i in range(0, shape[0]-self.size_block+1, self.size_block)])

    def cal_contrast_gpu(self, frame):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert frame to GPU tensor
        frame_tensor = torch.from_numpy(frame).float().to(device)
        shape = frame_tensor.shape

        # Initialize output tensor
        height = (shape[0] - self.size_block + 1) // self.size_block
        width = (shape[1] - self.size_block + 1) // self.size_block
        result = torch.zeros((height, width), device=device)

        # Unfold the frame into blocks
        blocks = frame_tensor.unfold(0, self.size_block, self.size_block).unfold(
            1, self.size_block, self.size_block)

        # Calculate mean and std for each block
        means = blocks.mean(dim=(2, 3))
        # Added unbiased=True to match numpy
        stds = blocks.std(dim=(2, 3), unbiased=True)

        # Calculate contrast
        # Add small epsilon to avoid division by zero
        result = stds / (means + 1e-6)

        return result.cpu().numpy()

    def extract_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)

        _, frame = cap.read()

        # crop = ImageCropper(frame)
        # x1, y1, x2, y2 = crop.start_cropping()

        # x1_ppg, y1_ppg, x2_ppg, y2_ppg = x1, y1, x2, y2

        x1_ppg, y1_ppg, x2_ppg, y2_ppg = self.get_center_crop_position(
            frame.shape, self.size_ppg)

        x1_spg, y1_spg, x2_spg, y2_spg = self.get_center_crop_position(
            frame.shape, self.size_spg)

        # color_ppg = (0, 255, 0)
        # color_spg = (255, 0, 0)
        # thickness = 2

        # cv2.rectangle(frame, (x1_ppg, y1_ppg),
        #               (x2_ppg, y2_ppg), color_ppg, thickness)
        # cv2.rectangle(frame, (x1_spg, y1_spg),
        #               (x2_spg, y2_spg), color_spg, thickness)
        # cv2.imshow('Image with Centered Square', frame)

        # fig, ax01 = plt.subplots(1, 1)
        # ax01.imshow(frame)
        # ax01.set_title('Raw Image')

        # fig, ax02 = plt.subplots(1, 1)
        # ax02.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        # ax02.set_title('Grayscale Image')

        # fig, ax03 = plt.subplots(1, 1)
        # ax03.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y1:y2, x1:x2])
        # ax03.set_title('Cropped Image')

        # frame_crop = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[
        #     y1_spg:y2_spg, x1_spg:x2_spg]

        # fig, ax04 = plt.subplots(1, 1)
        # mean_frame = np.mean(frame_crop)
        # # show the mean pixel intensity in the ROI
        # ax04.imshow(np.ones((frame_crop.shape[0], frame_crop.shape[1]))
        #             * mean_frame, cmap='gray')
        # ax04.set_title(f'Mean Frame: {mean_frame:.2f}')

        # fig, ax = plt.subplots(1, 1)
        # contrast = self.cal_contrast(frame_crop)
        # ax.imshow(contrast, cmap='hot')
        # ax.set_title('Contrast')

        return cap, [(x1_ppg, y1_ppg, x2_ppg, y2_ppg), (x1_spg, y1_spg, x2_spg, y2_spg)]

    def extract_signal(self, cap, position):

        signal_ppg = []
        mean_contrast_frame = []
        mean_exposure_frame = []
        # exposure_time = 1500  # us

        x1_ppg, y1_ppg, x2_ppg, y2_ppg = position[0]
        x1_spg, y1_spg, x2_spg, y2_spg = position[1]

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 1
        i = 0

        use_GPU = False
        # Check GPU availability
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            use_GPU = True

        self.printProgressBar(0, length, prefix='Progress:',
                              suffix='Complete', length=50)
        start_process = process_time()

        # video = cv2.VideoWriter(
        #     "roi.avi", cv2.VideoWriter_fourcc(*'XVID'), self.fps, (self.size_ppg, self.size_ppg))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Extract the ROI from the grayscale frame
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_ppg = frame[y1_ppg:y2_ppg, x1_ppg:x2_ppg]
            roi_spg = frame[y1_spg:y2_spg, x1_spg:x2_spg]

            roi_ppg = cv2.cvtColor(roi_ppg, cv2.COLOR_BGR2GRAY)
            roi_spg = cv2.cvtColor(roi_spg, cv2.COLOR_BGR2GRAY)

            # Calculate the mean pixel intensity in the ROI
            signal_intensity = np.mean(roi_ppg)
            signal_ppg.append(signal_intensity)

            contrast = self.cal_contrast_gpu(
                roi_spg) if use_GPU else self.cal_contrast(roi_spg)
            mean_contrast_frame.append(np.mean(contrast))  # mean contrast

            # Calculate mean exposure using the formula: 1 / (2 * T * K^2)
            T = self.exposure_time
            K = np.mean(contrast)
            epsilon = 1e-10
            mean_exposure = 1 / (2 * T * (np.square(K) + epsilon))
            mean_exposure_frame.append(mean_exposure)

            # video.write(frame)

            if (i == 0):
                fig, ax01 = plt.subplots(1, 1, figsize=(4, 4))
                ax01.imshow(frame[y1_ppg:y2_ppg, x1_ppg:x2_ppg])
                fig.tight_layout()
                fig.savefig(
                    f'{"/".join(self.video_path.split("/")[:-1])}/frame_{i}.png')

                fig, ax02 = plt.subplots(1, 1)
                ax02.imshow(roi_ppg, cmap='hot', )

                fig, ax03 = plt.subplots(1, 1)
                ax03.imshow(signal_intensity * np.ones(
                    (roi_ppg.shape[0], roi_ppg.shape[1])), cmap='hot')

                fig, ax04 = plt.subplots(1, 1)
                ax04.imshow(contrast, cmap='hot', )

                # fig, ax05 = plt.subplots(1, 1)
                # ax05.imshow(exposure, cmap='hot', )

            end_process = process_time()
            i = i+1
            self.printProgressBar(i + 1, length, prefix='Progress:',
                                  suffix=f'time: {(end_process-start_process):.2f} seconds. Complete', length=50)

        # video.release()
        end_process = process_time()
        i = i+1
        self.printProgressBar(i + 1, length, prefix='Progress',
                              suffix=f'time: {(end_process-start_process):.2f} seconds. Complete', length=50)
        cap.release()

        if not signal_ppg:
            print("Warning: No signal values were extracted from the video.")
        else:
            print("Signal values extracted successfully.", end='')

        return np.array(signal_ppg), np.array(mean_contrast_frame), np.array(mean_exposure_frame)

    def bandpass_filter(self, signal, lowcut, highcut, fs, order=3):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, signal)
        return y

    # Function to create a bandpass filter

    def lowpass_filter(self, signal, cutoff, fs, order=3):
        nyquist = 0.5 * fs
        normalized_cutoff = cutoff / nyquist
        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.99  # Ensure it's below 1
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, signal)
        return y

    def highpass_filter(self, signal, cutoff, fs, order=3):
        nyquist = 0.5 * fs
        normalized_cutoff = cutoff / nyquist
        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.99  # Ensure it's below 1
        b, a = butter(order, normalized_cutoff, btype='high', analog=False)
        y = filtfilt(b, a, signal)
        return y

    # Perform FFT on the signal

    def perform_fft(self, signal_ppg, frame_rate):
        N = len(signal_ppg)
        yf = fft(signal_ppg)
        xf = fftfreq(N, 1 / frame_rate)
        xf = xf[:N//2]
        yf = np.abs(np.array(yf[:N//2]))
        return xf, yf

    def find_harmonics(self, freqs, power_spectrum, fundamental_freq, max_harmonic=5):
        harmonics = []
        for i in range(1, max_harmonic + 1):
            harmonic_freq = i * fundamental_freq
            if harmonic_freq > 10:  # Limit to 10 Hz
                break
            idx = np.argmin(np.abs(freqs - harmonic_freq))
            harmonics.append((harmonic_freq, power_spectrum[idx]))
        return harmonics

    def calculate_snr_per_harmonic(self, signal, sample_rate, harmonics):
        snrs = []
        for freq, _ in harmonics:
            # Assuming a small window around each harmonic
            signal_freq_range = (freq - 0.1, freq + 0.1)
            noise_freq_ranges = [(0, signal_freq_range[0]),
                                 (signal_freq_range[1], 10)]
            snr, _, _ = self.calculate_snr(
                signal, sample_rate, signal_freq_range, noise_freq_ranges)
            snrs.append(snr)
        return snrs

    def define_fft(self, data,):
        fft_data = np.fft.fft(data)
        fft_data = np.abs(fft_data)
        fft_data = fft_data / len(data)
        fft_data = fft_data[0:len(data)//2]
        return fft_data

    def load_excel(self, file_name_excel):
        # Read the Excel file
        df = pd.read_excel(file_name_excel)
        # Convert the date column to datetime format if it's not already
        df['Time'] = pd.to_datetime(df['Data'], format='%H:%M:%S.%f')
        # df['Time'] = pd.to_datetime(df['Data'])

        # Define your start and end date
        # start_date = pd.to_datetime('17:41:48.318345', format='%H:%M:%S.%f')
        # end_date = pd.to_datetime('17:42:08.318345', format='%H:%M:%S.%f')

        end_time = df['Time'].max()
        start_time = end_time - pd.Timedelta(seconds=self.time_record)

        start_date = start_time
        end_date = end_time

        # Filter the data based on the date range
        filtered_df = df[(df['Time'] >= start_date) & (df['Time'] <= end_date)]

        # Calculate the time differences as a pandas Series
        time_excels = (
            pd.Series(filtered_df['Time']) - filtered_df['Time'].min())
        time_excel = time_excels.dt.total_seconds()
        amplitude_excel = pd.Series(filtered_df['Value']).values

        # print(time_excel)

        return [time_excel, amplitude_excel]

    def load_excel_sync(self, file_name_excel):
        # Read the Excel file
        df = pd.read_excel(file_name_excel)
        # Convert the date column to datetime format if it's not already
        # df['Time'] = pd.to_datetime(df['Data'], format='%H:%M:%S.%f')
        # df['Time'] = pd.to_datetime(df['Data'])

        # Define your start and end date
        # start_date = pd.to_datetime('17:41:48.318345', format='%H:%M:%S.%f')
        # end_date = pd.to_datetime('17:42:08.318345', format='%H:%M:%S.%f')

        # end_time = df['Time'].max()
        # start_time = end_time - pd.Timedelta(seconds=self.time_record)

        # start_date = start_time
        # end_date = end_time

        # # Filter the data based on the date range
        # filtered_df = df[(df['Time'] >= start_date) & (df['Time'] <= end_date)]

        # # Calculate the time differences as a pandas Series
        # time_excels = (
        #     pd.Series(filtered_df['Time']) - filtered_df['Time'].min())
        # time_excel = time_excels.dt.total_seconds()
        # amplitude_excel = pd.Series(filtered_df['Value']).values

        # print(time_excel)

        return [df['Hex Data'].values, df['Value'].values]

    def filter_signal(self, signal, fps, highcut, lowcut):
        # filter the signal
        filtered_ppg = self.lowpass_filter(signal, lowcut, fps)
        filtered_ppg = self.highpass_filter(filtered_ppg, highcut, fps)
        filtered_signal_fft = self.perform_fft(filtered_ppg, fps)

        # [b, a] = butter(4, 0.5, btype='highpass', fs=fps)
        # filtered_ppg = filtfilt(b, a, filtered_ppg)

        return [filtered_ppg, filtered_signal_fft]

    def calculate_snr(self, data, sample_rate, signal_freq_range, noise_freq_ranges):
        # Perform FFT
        n = len(data)
        yf = fft(data)
        xf = fftfreq(n, 1/sample_rate)

        # Calculate power
        power = np.abs(np.array(yf))**2 / n

        # Find signal power
        signal_power = np.sum(
            power[(xf >= signal_freq_range[0]) & (xf <= signal_freq_range[1])])

        # Find noise power (sum of both noise ranges, capped at 10 Hz)
        noise_power = np.sum(
            power[(xf >= noise_freq_ranges[0][0]) & (xf <= min(noise_freq_ranges[0][1], 10))])
        noise_power += np.sum(power[(xf >= noise_freq_ranges[1][0])
                                    & (xf <= min(noise_freq_ranges[1][1], 10))])

        # Calculate SNR
        snr = 10 * np.log10(signal_power / noise_power)

        return snr, xf, power

    def calculate_snr_hybrid_method(self, data, sample_rate, heart_rate_freq_range):
        # Perform FFT
        n = len(data)
        yf = fft(data)
        xf = fftfreq(n, 1/sample_rate)

        # Calculate power
        power = np.abs(np.array(yf))**2 / n

        # Frequency domain: Find peak frequency within heart rate range
        heart_rate_indices = (xf >= heart_rate_freq_range[0]) & (
            xf <= heart_rate_freq_range[1])
        peak_index = np.argmax(power[heart_rate_indices])
        peak_power = power[heart_rate_indices][peak_index]

        # Time domain: Calculate mean and variance
        mean_signal = np.mean(data)
        variance_signal = np.var(data)

        # Define signal and noise areas
        signal_area = peak_power / 2 + variance_signal
        noise_area = np.sum(power) - signal_area

        # print('Signal area: ', signal_area)
        # print('Noise area: ', noise_area)

        # Calculate SNR
        snr = 10 * np.log10(signal_area / noise_area)

        return snr, xf, power

    def snr_cal(self, fre_ppg, psd_ppg, lowCut_snr, highCut_snr):
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

        SNR_ppg = 10*np.log(power_ppg_mean/power_noise_ppg_mean)  # unit is dB

        return SNR_ppg

    def find_peak_freq(self, data, time):

        distance = self.fps * (30/88)

        peaks, _ = find_peaks(data, height=None,
                              threshold=None, distance=distance)

        # Calculate heart rate (if peaks represent heartbeats)
        time_diff = np.diff(time[peaks])
        heart_rate = 60 / np.mean(time_diff)  # unit is bpm

        return peaks, heart_rate

    def find_one_peaks(self, data):
        i_peaks, _ = find_peaks(data)

        i_max_peak = i_peaks[np.argmax(data[i_peaks])]

        return i_max_peak

    def find_second_peaks(self, data):
        i_peaks, _ = find_peaks(data)

        if len(i_peaks) < 2:
            return None

        # Sort peaks by their amplitude in descending order
        sorted_peaks = sorted(i_peaks, key=lambda x: data[x], reverse=True)

        # Return the second highest peak
        return sorted_peaks[1]

    def find_peak_freq_excel(self, data, time):
        # Find peaks
        # Adjust these parameters as needed for your specific data
        peaks, _ = find_peaks(data, height=None,
                              threshold=None, distance=60)

        peaks = peaks[1:]

        # print(peaks)
        # print(time[peaks])

        # Calculate heart rate (if peaks represent heartbeats)
        time_diff = np.diff(time[peaks])
        heart_rate = 60 / np.mean(time_diff)

        return peaks, heart_rate

    def Derivative(self, xlist, ylist):
        yprime = np.diff(ylist)/np.diff(xlist)
        xprime = []
        for i in range(len(yprime)):
            xtemp = (xlist[i+1]+xlist[i])/2
            xprime = np.append(xprime, xtemp)
        return xprime, yprime

    def main(self):
        # if (self.cache):
        #     try:
        #         signal_ppg = np.load(
        #             f'storage/{self.dir_path}/signal_ppg.npy')
        #         mean_exposure_frame = np.load(
        #             f'storage/{self.dir_path}/mean_exposure_frame.npy')
        #         mean_contrast_frame = np.load(
        #             f'storage/{self.dir_path}/mean_contrast_frame.npy')
        #     except:
        #         print("Error: Cache files not found. Extracting signal from video.")
        #         cap, position = self.extract_video_frames(self.video_path)
        #         signal_ppg, mean_contrast_frame, mean_exposure_frame = self.extract_signal(
        #             cap, position)
        #         np.save(f'storage/{self.dir_path}/signal_ppg.npy', signal_ppg)
        #         np.save(
        #             f'storage/{self.dir_path}/mean_exposure_frame.npy', mean_exposure_frame)
        #         np.save(
        #             f'storage/{self.dir_path}/mean_contrast_frame.npy', mean_contrast_frame)
        # else:
        #     cap, position = self.extract_video_frames(self.video_path)
        #     signal_ppg, mean_contrast_frame, mean_exposure_frame = self.extract_signal(
        #         cap, position
        #     )
        #     np.save(f'storage/{self.dir_path}/signal_ppg.npy', signal_ppg)
        #     np.save(
        #         f'storage/{self.dir_path}/mean_exposure_frame.npy', mean_exposure_frame)
        #     np.save(
        #         f'storage/{self.dir_path}/mean_contrast_frame.npy', mean_contrast_frame)

        cap, position = self.extract_video_frames(self.video_path)
        signal_ppg, mean_contrast_frame, mean_exposure_frame = self.extract_signal(
            cap, position
        )

        ranges = 0, 10
        # =========================== PPG ===========================
        print('.', end='')
        signal_freq_range_ppg = (0.83, 4)  # Hz (typical range for heart rate)
        noise_freq_range_ppg = [
            (0, signal_freq_range_ppg[0]), (signal_freq_range_ppg[1], 15)]

        fps = self.fps
        signal_values_fft = self.perform_fft(signal_ppg, fps)

        fs = signal_ppg.size / self.time_record
        time_ppg = np.arange(signal_ppg.size) / fs

        filtered_ppg, filtered_signal_fft1 = self.filter_signal(
            signal_ppg, fps,  signal_freq_range_ppg[0], signal_freq_range_ppg[1])

        filtered_ppg_reverse, filtered_signal_fft1_reverse = self.filter_signal(
            np.log(signal_ppg), fps,  signal_freq_range_ppg[0], signal_freq_range_ppg[1])

        # phase shift the signal 180 degree _reverse
        filtered_ppg_reverse = filtered_ppg_reverse

        # 1/ln
        filtered_ppg = filtered_ppg/np.max(filtered_ppg)

        peaks_filtered_signal, heart_rate_filtered_signal = self.find_peak_freq(
            filtered_ppg, time_ppg)

        max_filtered_ppg = self.find_one_peaks(filtered_signal_fft1[1])
        heart_rate_filtered_ppg = filtered_signal_fft1[0][max_filtered_ppg] * 60

        fig, ax5 = plt.subplots(1, 1, figsize=(6, 4))

        harminic_ippg = HarmonicAnalyzer(
            filtered_ppg, fs=self.fps, offset=0.1, verbose=False)

        fig, ax5 = harminic_ippg.plot_spectrum(fig, ax5)

        fig.tight_layout()
        fig.savefig(
            f'{"/".join(self.video_path.split("/")[:-1])}/iPPG_freq.png')

        fig, ax6 = plt.subplots(1, 1, figsize=(6, 4))

        # ax6.plot(time_ppg, filtered_ppg_reverse, color='y',
        #          label='iPPG Filtered Reverse Signal')
        ax6.plot(time_ppg, filtered_ppg, color='b',
                 label='iPPG Filtered Signal')
        # ax6.plot(time_ppg, signal_ppg, color='y',
        #          label='iPPG Signal')

        ax6.plot(time_ppg[peaks_filtered_signal], filtered_ppg[peaks_filtered_signal],
                 color='b', label="Peaks iPPG", marker='o', linestyle='')
        ax6.legend()
        ax6.set_xlim(ranges)
        # ax6.set_ylim([np.min(filtered_ppg) * 1.4, np.max(filtered_ppg) * 1.4])
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Amplitude')

        fig.tight_layout()
        fig.savefig(
            f'{"/".join(self.video_path.split("/")[:-1])}/iPPG.png')

        fig, ax60 = plt.subplots(1, 1, figsize=(6, 4))
        ax60.plot(time_ppg, np.log(signal_ppg) * 25, color='b',
                  label='iPPG Natural logarithm Signal')
        ax60.legend()
        ax60.set_xlim(ranges)
        ax60.set_xlabel('Time (s)')
        ax60.set_ylabel('Amplitude')
        fig.tight_layout()
        fig.savefig(
            f'{"/".join(self.video_path.split("/")[:-1])}/iPPG_Natural_logarithm.png')

        print('.', end='')
        # =========================== ippg pressure ===================================

        pressure_ippg = PressureAnalyzer(
            time_data=time_ppg, pressure_data=filtered_ppg, sample_rate=self.fps)

        # excel_data = self.load_excel(self.excel_path)
        # pressure_ippg = PressureAnalyzer(
        #     time_data=excel_data[0], pressure_data=excel_data[1], sample_rate=135)

        pressure_ippg.find_pressure_features(
            min_distance=0.5, height_percentile=75, diastolic_prominence=0.001, onset_search_range=0.001)

        results_pressure_ippg = pressure_ippg.get_results()
        pattern_count = pressure_ippg.detect_pattern()
        time_delay_systolic_diastolic = pressure_ippg.get_time_delay_systolic_diastolic()
        crest_time = pressure_ippg.get_crest_time(cast_time_distance=0.5)
        time_delays_dd = pressure_ippg.get_time_delay_dicrotic_diastolic(
            notch_to_dia_distance=0.3)
        t_sys = pressure_ippg.get_t_sys(notch_time_distance=1)
        t_dia = pressure_ippg.get_t_dia(pulse_end_time_distance=0.5)
        dw_75 = pressure_ippg.get_diastolic_width(percent=75)
        dw_66 = pressure_ippg.get_diastolic_width(percent=66)
        dw_50 = pressure_ippg.get_diastolic_width(percent=50)
        dw_33 = pressure_ippg.get_diastolic_width(percent=33)
        dw_25 = pressure_ippg.get_diastolic_width(percent=25)
        dw_10 = pressure_ippg.get_diastolic_width(percent=10)
        sw_75 = pressure_ippg.get_systolic_width(percent=75)
        sw_66 = pressure_ippg.get_systolic_width(percent=66)
        sw_50 = pressure_ippg.get_systolic_width(percent=50)
        sw_33 = pressure_ippg.get_systolic_width(percent=33)
        sw_25 = pressure_ippg.get_systolic_width(percent=25)
        sw_10 = pressure_ippg.get_systolic_width(percent=10)
        ipr = pressure_ippg.get_ipr()
        pulse_amplitude = pressure_ippg.get_pulse_amplitude(time_distance=0.5)
        reflection_index = pressure_ippg.get_reflection_index(time_distance=1)
        systolic_area = pressure_ippg.get_systolic_area(time_distance=1)
        diastolic_area = pressure_ippg.get_diastolic_area(time_distance=1)
        normalized_max_slope = pressure_ippg.get_normalized_max_slope()

        fig, ax_pressure = plt.subplots(1, 1, figsize=(6, 4))

        fig, ax_pressure = pressure_ippg.plot_results(
            fig, ax_pressure)
        ax_pressure.set_xlim(ranges)
        fig.tight_layout()
        fig.savefig(
            f'{"/".join(self.video_path.split("/")[:-1])}/iPPG_pressure.png')
        print('.', end='')
        # =========================== SPG ===================================
        # Hz (typical range for heart rate)
        signal_freq_range_spg = (0.83, 4.5)
        noise_freq_range_spg = [
            (0, signal_freq_range_spg[0]), (signal_freq_range_spg[1], 15)]

        fps = self.fps
        time_spg = np.arange(mean_exposure_frame.size) / fps

        mean_exposure_frame_fft = self.perform_fft(mean_exposure_frame, fps)

        filtered_spg, filtered_spg_fft = self.filter_signal(
            mean_exposure_frame, fps,  signal_freq_range_spg[0], signal_freq_range_spg[1])

        peaks_filtered_spg, heart_rate_filtered_spg = self.find_peak_freq(
            filtered_spg, time_spg)

        max_filtered_spg = self.find_one_peaks(filtered_spg_fft[1])
        heart_rate_spg = filtered_spg_fft[0][max_filtered_spg]*60

        fig, ax2 = plt.subplots(1, 1, figsize=(6, 4))

        harminic_spg = HarmonicAnalyzer(
            filtered_spg, fs=self.fps, offset=0.1, verbose=False)

        fig, ax2 = harminic_spg.plot_spectrum(fig, ax2)

        fig.tight_layout()
        fig.savefig(
            f'{"/".join(self.video_path.split("/")[:-1])}/SPG_freq.png')

        fig, ax3 = plt.subplots(1, 1, figsize=(6, 4))

        ax3.plot(time_spg, filtered_spg, color='g',
                 label='SPG Filtered Signal')
        ax3.plot(time_spg[peaks_filtered_spg], filtered_spg[peaks_filtered_spg],
                 color='g', label="Peaks SPG", marker='o', linestyle='')
        ax3.legend()
        ax3.set_xlim(ranges)
        ax3.set_ylim([np.min(filtered_spg) * 1.4, np.max(filtered_spg) * 1.4])
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude')

        fig.tight_layout()
        fig.savefig(f'{"/".join(self.video_path.split("/")[:-1])}/SPG.png')

        print('.', end='')
        # =========================== import excel ===========================
        excel_data = self.load_excel_sync(self.excel_path)

        time_excel = (np.arange(excel_data[1].size) / fps)
        time_excel = time_excel-0.125

        # apply filter
        signal_freq_range_excel = (0.83, 4)
        noise_freq_range_excel = [
            (0, signal_freq_range_excel[0]), (signal_freq_range_excel[1], 15)]
        fps_excel = self.fps  # sampling rate means the number of samples per second

        filtered_excel_fft = self.perform_fft(excel_data[1], fps_excel)

        # filtered_excel = excel_data[1]

        filtered_excel, filtered_excel_fft = self.filter_signal(
            excel_data[1], fps_excel,  signal_freq_range_excel[0], signal_freq_range_excel[1])

        peaks_filtered_excel, heart_rate_filtered_excel = self.find_peak_freq(
            filtered_excel, time_excel)

        max_filtered_excel = self.find_one_peaks(
            filtered_excel_fft[1])
        heart_rate_excel = filtered_excel_fft[0][max_filtered_excel]*60

        fig, ax11 = plt.subplots(1, 1, figsize=(8, 4))

        harminic_cppg = HarmonicAnalyzer(
            filtered_excel, fs=self.fps, offset=0.1, verbose=False)

        fig, ax11 = harminic_cppg.plot_spectrum(fig, ax11)

        fig.tight_layout()
        fig.savefig(
            f'{"/".join(self.video_path.split("/")[:-1])}/cPPG_freq.png')

        fig, ax12 = plt.subplots(1, 1, figsize=(8, 4))
        ax12.plot(time_excel, filtered_excel/np.max(filtered_excel), color='r',
                  label='cPPG Filtered Signal')
        ax12.plot(time_excel[peaks_filtered_excel], (filtered_excel/np.max(filtered_excel))[peaks_filtered_excel],
                  color='r', label="Peaks cPPG", marker='o', linestyle='')
        ax12.legend()
        ax12.set_xlim(ranges)
        ax12.set_xlabel('Time (s)')
        ax12.set_ylabel('Amplitude')

        fig.tight_layout()
        fig.savefig(
            f'{"/".join(self.video_path.split("/")[:-1])}/cPPG.png')

        # =========================== Tiem delay cppg ippg ===========================

        # remove the first peak
        peaks_filtered_ippg = peaks_filtered_signal[1:]
        peaks_filtered_cppg = peaks_filtered_excel[1:]

        # Convert peak indices to time points
        # not include the first peak
        ippg_peak_times = time_ppg[peaks_filtered_ippg]
        cppg_peak_times = time_ppg[peaks_filtered_cppg]

        # print("\n".join(map(str, ppg_peak_times)))
        # print("\n".join(map(str, spg_peak_times)))

        delay_threshold = 0.5  # seconds

        # Calculate time differences between closest corresponding peaks
        peak_delays = []
        valid_ippg_peaks = []
        valid_cppg_peaks = []

        i, j = 0, 0
        while i < len(ippg_peak_times) and j < len(cppg_peak_times):
            # Find the time difference for corresponding peaks
            delay = cppg_peak_times[j] - ippg_peak_times[i]

            # if (delay < 0.5 and delay > -0.5):
            #     peak_delays.append(delay)

            # Only keep delays within the threshold
            # And only keep positive delays
            if delay <= delay_threshold and delay >= 0:
                # peak_delays.append(np.abs(delay))
                print(cppg_peak_times[j], "    ",
                      ippg_peak_times[i], "    ", delay)
                peak_delays.append(delay)
                # peak_delays.append(delay)
                valid_ippg_peaks.append(peaks_filtered_ippg[i])
                valid_cppg_peaks.append(peaks_filtered_cppg[j])

                # Move to the next pair of peaks
                i += 1
                j += 1
            else:
                # If delay is too large, move the pointer that is behind
                if ippg_peak_times[i] < cppg_peak_times[j]:
                    i += 1
                else:
                    j += 1

        # for i in peak_delays:
        #     print(i)
        avg_time_delay = np.mean(peak_delays) if peak_delays else np.nan

        # =========================== SNR ===========================
        print('.', end='')
        # mean
        snr_signal_values, frequencies_signal_values, power_signal_values = self.calculate_snr(
            signal_ppg, fps, signal_freq_range_ppg, noise_freq_range_ppg)

        # ppg
        snr_filtered_ppg, frequencies_filtered_ppg, power_filtered_ppg = self.calculate_snr(
            filtered_ppg, fps, signal_freq_range_ppg, noise_freq_range_ppg)

        # spg
        snr_spg, frequencies_spg, power_spg = self.calculate_snr(
            mean_exposure_frame, fps, signal_freq_range_spg, noise_freq_range_spg)

        snr_filtered_spg, frequencies_filtered_spg, power_filtered_spg = self.calculate_snr(
            filtered_spg, fps, signal_freq_range_spg, noise_freq_range_spg)

        # excel
        snr_excel, frequencies_excel, power_excel = self.calculate_snr(
            excel_data[1], fps_excel, signal_freq_range_excel, noise_freq_range_excel)

        snr_filtered_excel, frequencies_excel, power_excel = self.calculate_snr(
            filtered_excel, fps_excel, signal_freq_range_excel, noise_freq_range_excel)

        # =========================== Time Delay ===========================
        time_plot = np.arange(mean_exposure_frame.size) / fps

        # remove the first peak
        peaks_filtered_signal = peaks_filtered_signal[1:]
        peaks_filtered_spg = peaks_filtered_spg[1:]

        # Convert peak indices to time points
        # not include the first peak
        ppg_peak_times = time_plot[peaks_filtered_signal]
        spg_peak_times = time_plot[peaks_filtered_spg]

        # print("\n".join(map(str, ppg_peak_times)))
        # print("\n".join(map(str, spg_peak_times)))

        delay_threshold = self.cut_time_delay  # seconds

        # Calculate time differences between closest corresponding peaks
        peak_delays = []
        valid_spg_peaks = []
        valid_ppg_peaks = []
        i, j = 0, 0
        while i < len(spg_peak_times) and j < len(ppg_peak_times):
            # Find the time difference for corresponding peaks
            delay = ppg_peak_times[j] - spg_peak_times[i]

            # if (delay < 0.5 and delay > -0.5):
            #     peak_delays.append(delay)

            # Only keep delays within the threshold
            # And only keep positive delays
            if delay <= delay_threshold and delay >= 0:
                # peak_delays.append(np.abs(delay))
                # print(ppg_peak_times[j], "  ", spg_peak_times[i])
                peak_delays.append(delay)
                # peak_delays.append(delay)
                valid_spg_peaks.append(peaks_filtered_spg[i])
                valid_ppg_peaks.append(peaks_filtered_signal[j])

                # Move to the next pair of peaks
                i += 1
                j += 1
            else:
                # If delay is too large, move the pointer that is behind
                if spg_peak_times[i] < ppg_peak_times[j]:
                    i += 1
                else:
                    j += 1

        diff = filtered_ppg/np.max(filtered_ppg) - \
            filtered_spg/np.max(filtered_spg)
        avg_diff = np.mean(diff)

        print('.', end='')
        # Calculate the average time delay between SPG and PPG peaks
        avg_time_delay = np.mean(peak_delays) if peak_delays else np.nan

        # avg_time_delay, spg_peak_times, ppg_peak_times, peak_delays

        # fig, (ax13, ax14, ax15) = plt.subplots(3, 1, figsize=(12, 6))  #
        fig, ax13 = plt.subplots(1, 1, figsize=(8, 4))

        ax13.plot(time_plot, filtered_ppg/np.max(filtered_ppg),
                  color='b', label='iPPG Signal')
        ax13.plot(time_plot[peaks_filtered_signal], (filtered_ppg/np.max(filtered_ppg))[peaks_filtered_signal],
                  color='b', marker='o', linestyle='')
        ax13.plot(time_plot, filtered_spg/np.max(filtered_spg),
                  color='g', label="SPG Signal")
        ax13.plot(time_plot[peaks_filtered_spg], (filtered_spg/np.max(filtered_spg))[peaks_filtered_spg],
                  color='g', marker='o', linestyle='')
        # Plot vertical lines representing time delay between each corresponding peak
        colors = ['r']
        for i in range(min(len(valid_spg_peaks), len(valid_ppg_peaks))):
            color = colors[i % len(colors)]  # Cycle through colors
            # Plot vertical line connecting the peaks
            ax13.plot([time_plot[valid_spg_peaks[i]], time_plot[valid_ppg_peaks[i]]],
                      [(filtered_spg/np.max(filtered_spg))[valid_spg_peaks[i]],
                       (filtered_ppg/np.max(filtered_ppg))[valid_ppg_peaks[i]]],
                      f'{color}--', alpha=0.7)
            # Plot markers at the peaks
            ax13.plot(time_plot[valid_spg_peaks[i]], (filtered_spg/np.max(filtered_spg))[valid_spg_peaks[i]],
                      marker='o', color=color)
            ax13.plot(time_plot[valid_ppg_peaks[i]], (filtered_ppg/np.max(filtered_ppg))[valid_ppg_peaks[i]],
                      marker='o', color=color)
            # Plot lines on the y-axis 0 to 1 for each pair of peaks
            ax13.plot([time_plot[valid_spg_peaks[i]], time_plot[valid_ppg_peaks[i]]],
                      [0, 0], f'{color}-', alpha=0.5)
            ax13.plot([time_plot[valid_spg_peaks[i]], time_plot[valid_spg_peaks[i]]],
                      [0, 1], f'{color}-', alpha=0.5)
            ax13.plot([time_plot[valid_ppg_peaks[i]], time_plot[valid_ppg_peaks[i]]],
                      [0, 1], f'{color}-', alpha=0.5)
            ax13.plot([time_plot[valid_spg_peaks[i]], time_plot[valid_ppg_peaks[i]]],
                      [1, 1], f'{color}-', alpha=0.5)

        ax13.grid()
        ax13.legend()
        ax13.set_ylim([-1.4, 1.4])
        ax13.set_xlim([15, 20])
        ax13.set_title('Time Delay Between PPG and SPG Peaks')
        ax13.set_xlabel('Time (s)')
        ax13.set_ylabel('Normalized Amplitude')

        # # Plot time delay between valid peak pairs
        # colors = ['r', 'y', 'c', 'm']
        # for idx, delay in enumerate(peak_delays):
        #     color = colors[idx % len(colors)]
        #     ax14.plot(idx, delay, "o-", color=color)
        # # ax14.axhline(y=avg_time_delay, color='gray', linestyle='--',
        # #              label=f'Average Delay: {avg_time_delay:.4f} s')
        # ax14.set_ylim([-0.2, 0.2])
        # ax14.set_xlabel('Peak Pair Index')
        # ax14.set_ylabel('Time Delay (s)')
        # ax14.set_title('Time Delays Between Corresponding Peaks')
        # ax14.grid()
        # # ax14.legend()

        # # Plot difference between SPG and PPG signals
        # ax15.plot(time_plot, diff, color='b',
        #           label=f'Avg Diff: {avg_diff:.4f}')
        # ax15.axhline(y=avg_diff, color='gray', linestyle='--')

        # # ax15.legend()
        # ax15.grid()
        # ax15.set_title('Difference Between SPG and PPG Signals')
        # ax15.set_xlabel('Time (s)')
        # ax15.set_ylabel('Amplitude Difference')

        # save figure to file
        fig.tight_layout()
        fig.savefig(
            f'{"/".join(self.video_path.split("/")[:-1])}/time_delay.png')

        # =========================== Plotting ===========================
        fig, ax7 = plt.subplots(1, 1, figsize=(8, 4))

        ax7.plot(time_plot, filtered_ppg/np.max(filtered_ppg),
                 color='b', label='iPPG Signal')
        ax7.plot(time_plot[peaks_filtered_signal], (filtered_ppg/np.max(filtered_ppg))[peaks_filtered_signal],
                 color='b', marker='o', linestyle='')
        ax7.plot(time_excel, (filtered_excel/np.max(filtered_excel)),
                 color='r', label='cPPG Signal', linestyle='--')
        ax7.plot(time_excel[peaks_filtered_excel], (filtered_excel/np.max(filtered_excel))[peaks_filtered_excel],
                 color='r', marker='o', linestyle='')
        ax7.plot(time_plot, filtered_spg/np.max(filtered_spg),
                 color='g', label="SPG Signal")
        ax7.plot(time_plot[peaks_filtered_spg], (filtered_spg/np.max(filtered_spg))[peaks_filtered_spg],
                 color='g', marker='o', linestyle='')
        # ax7.grid()
        ax7.legend()
        ax7.set_xlim(ranges)
        ax7.set_ylim([-1.4, 1.4])
        ax7.set_title('Integrate Signal')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Amplitude')

        # offset_ppg = np.mean((filtered_ppg/np.max(filtered_ppg))
        #                      [peaks_filtered_signal])

        # # offset_excel = np.mean((filtered_excel/np.max(filtered_excel))
        # #                        [peaks_filtered_excel])

        # # offset_ppg_excel = 1 + np.abs(offset_ppg - offset_excel)

        # ax8.plot(time_plot, (filtered_ppg/np.max(filtered_ppg)),  # * offset_ppg_excel,
        #          color='b', label='iPPG Signal')
        # ax8.plot(time_plot[peaks_filtered_signal], (filtered_ppg/np.max(filtered_ppg))[peaks_filtered_signal],  # * offset_ppg_excel,
        #          color='b', label='Peaks iPPG', marker='o', linestyle='')
        # ax8.plot(excel_data[0], filtered_excel/np.max(filtered_excel),
        #          color='r', label='cPPG Signal', linestyle='--')
        # # ax8.plot(excel_data[0][peaks_filtered_excel], (filtered_excel/np.max(filtered_excel))[peaks_filtered_excel],
        # #          color='r', label='Peaks cPPG', marker='o', linestyle='')

        # ax8.grid()
        # # ax8.legend()
        # ax8.set_title('Integrate iPPG Signal')
        # ax8.set_xlabel('Time (s)')
        # ax8.set_ylabel('Amplitude')

        # ax9.plot(excel_data[0], filtered_excel/np.max(filtered_excel),
        #          color='r', label='cPPG Signal', linestyle='--')
        # # ax9.plot(excel_data[0][peaks_filtered_excel], (filtered_excel/np.max(filtered_excel))[peaks_filtered_excel],
        # #          color='r', label='Peaks cPPG', marker='o', linestyle='')
        # ax9.plot(time_plot, filtered_spg/np.max(filtered_spg),
        #          color='g', label="SPG Signal")
        # ax9.plot(time_plot[peaks_filtered_spg], (filtered_spg/np.max(filtered_spg))[peaks_filtered_spg],
        #          color='g', label="Peaks SPG", marker='o', linestyle='')
        # ax9.grid()
        # # ax9.legend()
        # ax9.set_title('Integrate SPG Signal')
        # ax9.set_xlabel('Time (s)')
        # ax9.set_ylabel('Amplitude')

        # # Add MultiCursor to sync movement across subplots
        # multi = MultiCursor(fig.canvas, (ax7, ax8, ax9), color='y', lw=1,
        #                     horizOn=True, vertOn=True)

        # save figure to file
        fig.tight_layout()
        fig.savefig(
            f'{"/".join(self.video_path.split("/")[:-1])}/integrate_signal.png')
        # }/{folder}/integrate_signal.png')

        # plt.show()
        print('.', end='')

        if self.personal == {}:
            with open(f'{"/".join(self.video_path.split("/")[:-1])}/config.yml', "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)

            self.personal = config['personal']

        # Save time_ppg, filtered_ppg, and filtered_signal_fft1 to files
        # Create ippg folder if it doesn't exist
        main_folder = "/".join(self.video_path.split("/")[:-1])
        ippg_folder = f'{main_folder}/ippg'
        if not os.path.exists(ippg_folder):
            os.makedirs(ippg_folder)

        np.save(
            f'{ippg_folder}/ippg_filtered_time.npy', time_ppg)
        np.save(
            f'{ippg_folder}/ippg_filtered_amp.npy', filtered_ppg/np.max(filtered_ppg))
        np.save(
            f'{ippg_folder}/ippg_filtered_peak.npy', peaks_filtered_signal)
        np.save(
            f'{ippg_folder}/filtered_ippg_fft_freq.npy', harminic_ippg.frequencies)
        np.save(
            f'{ippg_folder}/filtered_ippg_fft_amp.npy', harminic_ippg.magnitude)
        print(harminic_ippg.peaks)
        np.save(
            f'{ippg_folder}/filtered_ippg_fft_peak.npy', harminic_ippg.peaks)

        # Save time_spg, filtered_spg, and filtered_spg_fft to files
        spg_folder = f'{main_folder}/spg'
        if not os.path.exists(spg_folder):
            os.makedirs(spg_folder)

        np.save(
            f'{spg_folder}/spg_filtered_spg_time.npy', time_spg)
        np.save(
            f'{spg_folder}/spg_filtered_spg_amp.npy', filtered_spg/np.max(filtered_spg))
        np.save(
            f'{spg_folder}/spg_filtered_spg_peak.npy', peaks_filtered_spg)
        np.save(
            f'{spg_folder}/filtered_spg_fft_freq.npy', harminic_spg.frequencies)
        np.save(
            f'{spg_folder}/filtered_spg_fft_amp.npy', harminic_spg.magnitude)
        np.save(
            f'{spg_folder}/filtered_spg_fft_peak.npy', harminic_spg.peaks)

        # Save excel_data[0], filtered_excel, and filtered_excel_fft to files
        cppg_folder = f'{main_folder}/cppg'
        if not os.path.exists(cppg_folder):
            os.makedirs(cppg_folder)

        np.save(
            f'{cppg_folder}/cppg_filtered_excel_time.npy', time_excel)
        np.save(
            f'{cppg_folder}/cppg_filtered_excel_amp.npy', filtered_excel/np.max(filtered_excel))
        np.save(
            f'{cppg_folder}/cppg_filtered_excel_peak.npy', peaks_filtered_excel)
        np.save(
            f'{cppg_folder}/filtered_excel_fft_freq.npy', harminic_cppg.frequencies)
        np.save(
            f'{cppg_folder}/filtered_excel_fft_amp.npy', harminic_cppg.magnitude)
        np.save(
            f'{cppg_folder}/filtered_excel_fft_peak.npy', harminic_cppg.peaks)

        print('.')

        print(
            f"Heart rate cPPG: {heart_rate_excel:.2f}, iPPG: {heart_rate_filtered_ppg:.2f}, SPG: {heart_rate_spg:.2f} BPM")

        print(
            f"SNR cPPG: {snr_filtered_excel:.2f}, iPPG: {snr_filtered_ppg:.2f}, SPG: {snr_filtered_spg:.2f} dB")

        qulity_time_delay = 100 * \
            (1 - np.std(peak_delays) / np.max(peak_delays))
        qulity_time_delay_count = 100 - (1+len(results_pressure_ippg['systolic']['values'])-len(
            peak_delays))*100/len(results_pressure_ippg['systolic']['values'])

        print(
            f"Average SPG-PPG peaks: {avg_time_delay:.6f}, std: {np.std(peak_delays):.6f}, max: {np.max(peak_delays):.6f} seconds. quality: {(qulity_time_delay+qulity_time_delay_count)/2:.2f}%")

        print(f"Freq H3/H1 cPPG: {(harminic_cppg.third_mag/harminic_cppg.fundamental_mag).round(6)}, iPPG: {(harminic_ippg.third_mag/harminic_ippg.fundamental_mag).round(6)}, SPG: {(harminic_spg.third_mag/harminic_spg.fundamental_mag).round(6)}")
        print(f"Freq H3/H2 cPPG: {(harminic_cppg.third_mag/harminic_cppg.second_mag).round(6)}, iPPG: {(harminic_ippg.third_mag/harminic_ippg.second_mag).round(6)}, SPG: {(harminic_spg.third_mag/harminic_spg.second_mag).round(6)}")
        print(f"Freq H2/H1 cPPG: {(harminic_cppg.second_mag/harminic_cppg.fundamental_mag).round(6)}, iPPG: {(harminic_ippg.second_mag/harminic_ippg.fundamental_mag).round(6)}, SPG: {(harminic_spg.second_mag/harminic_spg.fundamental_mag).round(6)}")

        print(f"Onset: {len(results_pressure_ippg['onset']['values'])}, Systolic: {len(results_pressure_ippg['systolic']['values'])}, Dicrotic: {len(results_pressure_ippg['dicrotic']['values'])}, Diastolic: {len(results_pressure_ippg['diastolic']['values'])}, complete patterns: {pattern_count}")

        qulity_t_dia_t_sys = 100 * \
            (1 - np.std(time_delay_systolic_diastolic) /
             np.max(time_delay_systolic_diastolic))
        qulity_t_dia_t_sys_count = 100 - (1+len(results_pressure_ippg['systolic']['values'])-len(
            time_delay_systolic_diastolic))*100/len(results_pressure_ippg['systolic']['values'])

        print(
            f"Average (t_dia - t_sys): {np.mean(time_delay_systolic_diastolic):.4f}, std: {np.std(time_delay_systolic_diastolic):.4f}, max: {np.max(time_delay_systolic_diastolic):.4f} seconds. quality: {(qulity_t_dia_t_sys+qulity_t_dia_t_sys_count)/2:.2f}%")

        print(
            f"Stiffness Index h/(t_dia - t_sys): {(170/100)/np.mean(time_delay_systolic_diastolic):.4f}")

        qulity_crest_time = 100 * (1 - np.std(crest_time) / np.max(crest_time))
        qulity_crest_time_count = 100 - (1+len(results_pressure_ippg['systolic']['values'])-len(
            crest_time))*100/len(results_pressure_ippg['systolic']['values'])

        print(
            f"Average Crest Time (t_sys - t_0): {np.mean(crest_time):.4f}, std: {np.std(crest_time):.4f}, max: {np.max(crest_time):.4f} seconds. quality: {(qulity_crest_time+qulity_crest_time_count)/2:.2f}%")

        qulity_dd = 100 * (1 - np.std(time_delays_dd) / np.max(time_delays_dd))
        qulity_dd_count = 100 - (1+len(results_pressure_ippg['diastolic']['values'])-len(
            time_delays_dd))*100/len(results_pressure_ippg['diastolic']['values'])

        print(
            f"Average (t_dic - t_dia): {np.mean(time_delays_dd):.4f}, std: {np.std(time_delays_dd):.4f}, max: {np.max(time_delays_dd):.4f} seconds. quality: {(qulity_dd+qulity_dd_count)/2:.2f}%")

        print(
            f"t_ratio (t_sys-t(0)) / (t_dia-t_dic)): {np.mean(crest_time)/np.mean(time_delays_dd):.4f}")

        qulity_t_sys = 100 * \
            (1 - np.std(t_sys) / np.max(t_sys))
        qulity_t_sys_count = 100 - (1+len(results_pressure_ippg['systolic']['values'])-len(
            t_sys))*100/len(results_pressure_ippg['systolic']['values'])

        print(
            f"Average t_sys (t(dic)-t(0)): {np.mean(t_sys):.4f}, std: {np.std(t_sys):.4f}, max: {np.max(t_sys):.4f} seconds. quality: {(qulity_t_sys+qulity_t_sys_count)/2:.2f}%")

        qulity_t_dia = 100 * \
            (1 - np.std(t_dia) / np.max(t_dia))
        qulity_t_dia_count = 100 - (1+len(results_pressure_ippg['onset']['values'])-len(
            t_dia))*100/len(results_pressure_ippg['onset']['values'])

        print(
            f"Average t_dia (t(0)-t(dic)): {np.mean(t_dia):.4f}, std: {np.std(t_dia):.4f}, max: {np.max(t_dia):.4f} seconds. quality: {(qulity_t_dia+qulity_t_dia_count)/2:.2f}%")

        print(
            f"t_ratio (t_sys-t_0)/(t(dic)-t(0)): {np.mean(crest_time)/np.mean(t_sys):.4f}")

        qulity_dw_75 = 100 * \
            (1 - np.std(dw_75) / np.max(dw_75))
        qulity_dw_75_count = 100 - (1+len(results_pressure_ippg['onset']['values'])-len(
            dw_75))*100/len(results_pressure_ippg['onset']['values'])

        print(
            f"Average dw_75: {np.mean(dw_75):.4f}, std: {np.std(dw_75):.4f}, max: {np.max(dw_75):.4f} seconds. quality: {(qulity_dw_75+qulity_dw_75_count)/2:.2f}%")

        qulity_dw_66 = 100 * \
            (1 - np.std(dw_66) / np.max(dw_66))
        qulity_dw_66_count = 100 - (1+len(results_pressure_ippg['onset']['values'])-len(
            dw_66))*100/len(results_pressure_ippg['onset']['values'])

        print(
            f"Average dw_66: {np.mean(dw_66):.4f}, std: {np.std(dw_66):.4f}, max: {np.max(dw_66):.4f} seconds. quality: {(qulity_dw_66+qulity_dw_66_count)/2:.2f}%")

        qulity_dw_50 = 100 * \
            (1 - np.std(dw_50) / np.max(dw_50))
        qulity_dw_50_count = 100 - (1+len(results_pressure_ippg['onset']['values'])-len(
            dw_50))*100/len(results_pressure_ippg['onset']['values'])

        print(
            f"Average dw_50: {np.mean(dw_50):.4f}, std: {np.std(dw_50):.4f}, max: {np.max(dw_50):.4f} seconds. quality: {(qulity_dw_50+qulity_dw_50_count)/2:.2f}%")

        qulity_dw_33 = 100 * \
            (1 - np.std(dw_33) / np.max(dw_33))
        qulity_dw_33_count = 100 - (1+len(results_pressure_ippg['onset']['values'])-len(
            dw_33))*100/len(results_pressure_ippg['onset']['values'])

        print(
            f"Average dw_33: {np.mean(dw_33):.4f}, std: {np.std(dw_33):.4f}, max: {np.max(dw_33):.4f} seconds. quality: {(qulity_dw_33+qulity_dw_33_count)/2:.2f}%")

        qulity_dw_25 = 100 * \
            (1 - np.std(dw_25) / np.max(dw_25))
        qulity_dw_25_count = 100 - (1+len(results_pressure_ippg['onset']['values'])-len(
            dw_25))*100/len(results_pressure_ippg['onset']['values'])

        print(
            f"Average dw_25: {np.mean(dw_25):.4f}, std: {np.std(dw_25):.4f}, max: {np.max(dw_25):.4f} seconds. quality: {(qulity_dw_25+qulity_dw_25_count)/2:.2f}%")

        qulity_dw_10 = 100 * \
            (1 - np.std(dw_10) / np.max(dw_10))
        qulity_dw_10_count = 100 - (1+len(results_pressure_ippg['onset']['values'])-len(
            dw_10))*100/len(results_pressure_ippg['onset']['values'])

        print(
            f"Average dw_10: {np.mean(dw_10):.4f}, std: {np.std(dw_10):.4f}, max: {np.max(dw_10):.4f} seconds. quality: {(qulity_dw_10+qulity_dw_10_count)/2:.2f}%")

        qulity_slope = 100 * \
            (1 - np.std(normalized_max_slope) / np.max(normalized_max_slope))
        qulity_slope_count = 100 - (1+len(results_pressure_ippg['onset']['values'])-len(
            normalized_max_slope))*100/len(results_pressure_ippg['onset']['values'])

        qulity_sw_75 = 100 * \
            (1 - np.std(sw_75) / np.max(sw_75))
        qulity_sw_75_count = 100 - (1+len(results_pressure_ippg['onset']['values'])-len(
            sw_75))*100/len(results_pressure_ippg['onset']['values'])

        print(
            f"Average sw_75: {np.mean(sw_75):.4f}, std: {np.std(sw_75):.4f}, max: {np.max(sw_75):.4f} seconds. quality: {(qulity_sw_75+qulity_sw_75_count)/2:.2f}%")

        qulity_sw_66 = 100 * \
            (1 - np.std(sw_66) / np.max(sw_66))
        qulity_sw_66_count = 100 - (1+len(results_pressure_ippg['onset']['values'])-len(
            sw_66))*100/len(results_pressure_ippg['onset']['values'])

        print(
            f"Average sw_66: {np.mean(sw_66):.4f}, std: {np.std(sw_66):.4f}, max: {np.max(sw_66):.4f} seconds. quality: {(qulity_sw_66+qulity_sw_66_count)/2:.2f}%")

        qulity_sw_50 = 100 * \
            (1 - np.std(sw_50) / np.max(sw_50))
        qulity_sw_50_count = 100 - (1+len(results_pressure_ippg['onset']['values'])-len(
            sw_50))*100/len(results_pressure_ippg['onset']['values'])

        print(
            f"Average sw_50: {np.mean(sw_50):.4f}, std: {np.std(sw_50):.4f}, max: {np.max(sw_50):.4f} seconds. quality: {(qulity_sw_50+qulity_sw_50_count)/2:.2f}%")

        qulity_sw_33 = 100 * \
            (1 - np.std(sw_33) / np.max(sw_33))
        qulity_sw_33_count = 100 - (1+len(results_pressure_ippg['onset']['values'])-len(
            sw_33))*100/len(results_pressure_ippg['onset']['values'])

        print(
            f"Average sw_33: {np.mean(sw_33):.4f}, std: {np.std(sw_33):.4f}, max: {np.max(sw_33):.4f} seconds. quality: {(qulity_sw_33+qulity_sw_33_count)/2:.2f}%")

        qulity_sw_25 = 100 * \
            (1 - np.std(sw_25) / np.max(sw_25))
        qulity_sw_25_count = 100 - (1+len(results_pressure_ippg['onset']['values'])-len(
            sw_25))*100/len(results_pressure_ippg['onset']['values'])

        print(
            f"Average sw_25: {np.mean(sw_25):.4f}, std: {np.std(sw_25):.4f}, max: {np.max(sw_25):.4f} seconds. quality: {(qulity_sw_25+qulity_sw_25_count)/2:.2f}%")

        qulity_sw_10 = 100 * \
            (1 - np.std(sw_10) / np.max(sw_10))
        qulity_sw_10_count = 100 - (1+len(results_pressure_ippg['onset']['values'])-len(
            sw_10))*100/len(results_pressure_ippg['onset']['values'])

        print(
            f"Average sw_10: {np.mean(sw_10):.4f}, std: {np.std(sw_10):.4f}, max: {np.max(sw_10):.4f} seconds. quality: {(qulity_sw_10+qulity_sw_10_count)/2:.2f}%")

        print(f"Average w_75: {np.mean(sw_75)+np.mean(dw_75):.4f}")
        print(f"Average w_66: {np.mean(sw_66)+np.mean(dw_66):.4f}")
        print(f"Average w_50: {np.mean(sw_50)+np.mean(dw_50):.4f}")
        print(f"Average w_33: {np.mean(sw_33)+np.mean(dw_33):.4f}")
        print(f"Average w_10: {np.mean(sw_10)+np.mean(dw_10):.4f}")

        print(f"Average dw_75/sw_75: {np.mean(dw_75)/np.mean(sw_75):.4f}")
        print(f"Average dw_66/sw_66: {np.mean(dw_66)/np.mean(sw_66):.4f}")
        print(f"Average dw_50/sw_50: {np.mean(dw_50)/np.mean(sw_50):.4f}")
        print(f"Average dw_33/sw_33: {np.mean(dw_33)/np.mean(sw_33):.4f}")
        print(f"Average dw_10/sw_10: {np.mean(dw_10)/np.mean(sw_10):.4f}")

        qulity_ipr = 100 * \
            (1 - np.std(ipr) / np.max(ipr))
        qulity_ipr_count = 100 - (1+len(results_pressure_ippg['onset']['values'])-len(
            ipr))*100/len(results_pressure_ippg['onset']['values'])

        print(
            f"Average IPR: {np.mean(ipr):.4f}, std: {np.std(ipr):.4f}, max: {np.max(ipr):.4f} seconds. quality: {(qulity_ipr+qulity_ipr_count)/2:.2f}%")

        qulity_pulse_amplitude = 100 * \
            (1 - np.std(pulse_amplitude) / np.max(pulse_amplitude))
        qulity_pulse_amplitude_count = 100 - (1+len(results_pressure_ippg['onset']['values'])-len(
            pulse_amplitude))*100/len(results_pressure_ippg['onset']['values'])

        print(
            f"Average pulse amplitude: {np.mean(pulse_amplitude):.4f}, std: {np.std(pulse_amplitude):.4f}, max: {np.max(pulse_amplitude):.4f} seconds. quality: {(qulity_pulse_amplitude+qulity_pulse_amplitude_count)/2:.2f}%")

        qulity_reflection_index = 100 * \
            (1 - np.std(reflection_index) / np.max(reflection_index))
        qulity_reflection_index_count = 100 - (1+len(results_pressure_ippg['onset']['values'])-len(
            reflection_index))*100/len(results_pressure_ippg['onset']['values'])

        print(
            f"Average reflection index: {np.mean(reflection_index):.4f}, std: {np.std(reflection_index):.4f}, max: {np.max(reflection_index):.4f} seconds. quality: {(qulity_reflection_index+qulity_reflection_index_count)/2:.2f}%")

        qulity_systolic_area = 100 * \
            (1 - np.std(systolic_area) / np.max(systolic_area))
        qulity_systolic_area_count = 100 - (1+len(results_pressure_ippg['onset']['values'])-len(
            systolic_area))*100/len(results_pressure_ippg['onset']['values'])

        print(
            f"Average systolic area: {np.mean(systolic_area):.4f}, std: {np.std(systolic_area):.4f}, max: {np.max(systolic_area):.4f} seconds. quality: {(qulity_systolic_area+qulity_systolic_area_count)/2:.2f}%")

        qulity_diastolic_area = 100 * \
            (1 - np.std(diastolic_area) / np.max(diastolic_area))
        qulity_diastolic_area_count = 100 - (1+len(results_pressure_ippg['onset']['values'])-len(
            diastolic_area))*100/len(results_pressure_ippg['onset']['values'])

        print(
            f"Average diastolic area: {np.mean(diastolic_area):.4f}, std: {np.std(diastolic_area):.4f}, max: {np.max(diastolic_area):.4f} seconds. quality: {(qulity_diastolic_area+qulity_diastolic_area_count)/2:.2f}%")

        print(
            f"IPA inflection point: {np.mean(diastolic_area)/np.mean(systolic_area):.4f}")

        qulity_slope = 100 * \
            (1 - np.std(normalized_max_slope) / np.max(normalized_max_slope))
        qulity_slope_count = 100 - (1+len(results_pressure_ippg['onset']['values'])-len(
            normalized_max_slope))*100/len(results_pressure_ippg['onset']['values'])

        print(
            f"Average maximum slope x'(ms)/(x(sys)-x(0)): {np.mean(normalized_max_slope):.4f}, std: {np.std(normalized_max_slope):.4f}, max: {np.max(normalized_max_slope):.4f} seconds. quality: {(qulity_slope+qulity_slope_count)/2:.2f}%")

        print('\n')

        data_feature = {
            "name": self.personal['name'],
            "age": self.personal['age'],
            "gender": self.personal['gender'],
            "height": self.personal['height'],
            "weight": self.personal['weight'],
            "congenital_disease": self.personal['congenital_disease'],
            "congenital_disease_detail": self.personal['congenital_disease_detail'],

            "size_ppg": self.size_ppg,
            "size_spg": self.size_spg,
            "exposure_time": self.exposure_time,
            "fps": self.fps,
            "time_record": self.time_record,

            'SNR iPPG': float(snr_filtered_ppg.round(2)),
            'SNR SPG': float(snr_filtered_spg.round(2)),
            'SNR cPPG': float(snr_filtered_excel.round(2)),
            'Average Time Delay': float(avg_time_delay.round(6)),
            'Heart Rate iPPG': float(heart_rate_filtered_ppg.round(2)),
            'Heart Rate SPG': float(heart_rate_spg.round(2)),
            'Heart Rate cPPG': float(heart_rate_excel.round(2)),
            'Freq H1 cPPG': float((harminic_cppg.fundamental_freq).round(6)),
            'Freq H1 iPPG': float((harminic_ippg.fundamental_freq).round(6)),
            'Freq H1 SPG': float((harminic_spg.fundamental_freq).round(6)),
            'Freq H2 cPPG': float((harminic_cppg.second_freq).round(6)),
            'Freq H2 iPPG': float((harminic_ippg.second_freq).round(6)),
            'Freq H2 SPG': float((harminic_spg.second_freq).round(6)),
            'Freq H3 cPPG': float((harminic_cppg.third_freq).round(6)),
            'Freq H3 iPPG': float((harminic_ippg.third_freq).round(6)),
            'Freq H3 SPG': float((harminic_spg.third_freq).round(6)),
            'Mag H1 cPPG': float((harminic_cppg.fundamental_mag).round(6)),
            'Mag H1 iPPG': float((harminic_ippg.fundamental_mag).round(6)),
            'Mag H1 SPG': float((harminic_spg.fundamental_mag).round(6)),
            'Mag H2 cPPG': float((harminic_cppg.second_mag).round(6)),
            'Mag H2 iPPG': float((harminic_ippg.second_mag).round(6)),
            'Mag H2 SPG': float((harminic_spg.second_mag).round(6)),
            'Mag H3 cPPG': float((harminic_cppg.third_mag).round(6)),
            'Mag H3 iPPG': float((harminic_ippg.third_mag).round(6)),
            'Mag H3 SPG': float((harminic_spg.third_mag).round(6)),
            'onset': int(len(results_pressure_ippg['onset']['values'])),
            'systolic': int(len(results_pressure_ippg['systolic']['values'])),
            'dicrotic': int(len(results_pressure_ippg['dicrotic']['values'])),
            'diastolic': int(len(results_pressure_ippg['diastolic']['values'])),
            'complete patterns': int(pattern_count),
            'Average (t_dia - t_sys)': float(np.mean(time_delay_systolic_diastolic).round(6)),
            'Stiffness Index h/(t_dia - t_sys)': float(((170/100)/np.mean(time_delay_systolic_diastolic)).round(6)),
            'Average Crest Time (t_sys - t_0)': float(np.mean(crest_time).round(6)),
            'Average (t_dic - t_dia)': float(np.mean(time_delays_dd).round(6)),
            't_ratio (t_sys-t(0)) / (t_dia-t_dic)': float((np.mean(crest_time)/np.mean(time_delays_dd)).round(6)),
            'Average t_sys (t(dic)-t(0))': float(np.mean(t_sys).round(6)),
            'Average t_dia (t(0)-t(dic))': float(np.mean(t_dia).round(6)),
            'Average dw_75': float(np.mean(dw_75).round(6)),
            'Average dw_66': float(np.mean(dw_66).round(6)),
            'Average dw_50': float(np.mean(dw_50).round(6)),
            'Average dw_33': float(np.mean(dw_33).round(6)),
            'Average dw_25': float(np.mean(dw_25).round(6)),
            'Average dw_10': float(np.mean(dw_10).round(6)),
            'Average sw_75': float(np.mean(sw_75).round(6)),
            'Average sw_66': float(np.mean(sw_66).round(6)),
            'Average sw_50': float(np.mean(sw_50).round(6)),
            'Average sw_33': float(np.mean(sw_33).round(6)),
            'Average sw_25': float(np.mean(sw_25).round(6)),
            'Average sw_10': float(np.mean(sw_10).round(6)),
            'Average w_75': float((np.mean(sw_75)+np.mean(dw_75)).round(6)),
            'Average w_66': float((np.mean(sw_66)+np.mean(dw_66)).round(6)),
            'Average w_50': float((np.mean(sw_50)+np.mean(dw_50)).round(6)),
            'Average w_33': float((np.mean(sw_33)+np.mean(dw_33)).round(6)),
            'Average w_25': float((np.mean(sw_25)+np.mean(dw_25)).round(6)),
            'Average w_10': float((np.mean(sw_10)+np.mean(dw_10)).round(6)),
            'Average dw_75/sw_75': float((np.mean(dw_75)/np.mean(sw_75)).round(6)),
            'Average dw_66/sw_66': float((np.mean(dw_66)/np.mean(sw_66)).round(6)),
            'Average dw_50/sw_50': float((np.mean(dw_50)/np.mean(sw_50)).round(6)),
            'Average dw_33/sw_33': float((np.mean(dw_33)/np.mean(sw_33)).round(6)),
            'Average dw_25/sw_25': float((np.mean(dw_25)/np.mean(sw_25)).round(6)),
            'Average dw_10/sw_10': float((np.mean(dw_10)/np.mean(sw_10)).round(6)),
            'Average IPR': float(np.mean(ipr).round(6)),
            'Average pulse amplitude': float(np.mean(pulse_amplitude).round(6)),
            'Average reflection index': float(np.mean(reflection_index).round(6)),
            'Average systolic area': float(np.mean(systolic_area).round(6)),
            'Average diastolic area': float(np.mean(diastolic_area).round(6)),
            'IPA inflection point': float((np.mean(diastolic_area)/np.mean(systolic_area)).round(6)),
            'Average maximum slope': float(np.mean(normalized_max_slope).round(6)),

        }

        # save to excel
        df = pd.DataFrame([data_feature])
        df.to_excel(
            f'{"/".join(self.video_path.split("/")[:-1])}/data_feature.xlsx', index=False)

        # create config file for use
        data_config = {
            "personal": self.personal,
            "config": {
                "size_ppg": self.size_ppg,
                "size_spg": self.size_spg,
                "exposure_time": self.exposure_time,
                "fps": self.fps,
                "time_record": self.time_record
            },
            "result": {
                'SNR iPPG': float(snr_filtered_ppg.round(2)),
                'SNR SPG': float(snr_filtered_spg.round(2)),
                'SNR cPPG': float(snr_filtered_excel.round(2)),
                'Average Time Delay': float(avg_time_delay.round(6)),
                'Heart Rate iPPG': float(heart_rate_filtered_ppg.round(2)),
                'Heart Rate SPG': float(heart_rate_spg.round(2)),
                'Heart Rate cPPG': float(heart_rate_excel.round(2)),
                'Freq H1 cPPG': float((harminic_cppg.fundamental_freq).round(6)),
                'Freq H1 iPPG': float((harminic_ippg.fundamental_freq).round(6)),
                'Freq H1 SPG': float((harminic_spg.fundamental_freq).round(6)),
                'Freq H2 cPPG': float((harminic_cppg.second_freq).round(6)),
                'Freq H2 iPPG': float((harminic_ippg.second_freq).round(6)),
                'Freq H2 SPG': float((harminic_spg.second_freq).round(6)),
                'Freq H3 cPPG': float((harminic_cppg.third_freq).round(6)),
                'Freq H3 iPPG': float((harminic_ippg.third_freq).round(6)),
                'Freq H3 SPG': float((harminic_spg.third_freq).round(6)),
                'Mag H1 cPPG': float((harminic_cppg.fundamental_mag).round(6)),
                'Mag H1 iPPG': float((harminic_ippg.fundamental_mag).round(6)),
                'Mag H1 SPG': float((harminic_spg.fundamental_mag).round(6)),
                'Mag H2 cPPG': float((harminic_cppg.second_mag).round(6)),
                'Mag H2 iPPG': float((harminic_ippg.second_mag).round(6)),
                'Mag H2 SPG': float((harminic_spg.second_mag).round(6)),
                'Mag H3 cPPG': float((harminic_cppg.third_mag).round(6)),
                'Mag H3 iPPG': float((harminic_ippg.third_mag).round(6)),
                'Mag H3 SPG': float((harminic_spg.third_mag).round(6)),
                'onset': int(len(results_pressure_ippg['onset']['values'])),
                'systolic': int(len(results_pressure_ippg['systolic']['values'])),
                'dicrotic': int(len(results_pressure_ippg['dicrotic']['values'])),
                'diastolic': int(len(results_pressure_ippg['diastolic']['values'])),
                'complete patterns': int(pattern_count),
                'Average (t_dia - t_sys)': float(np.mean(time_delay_systolic_diastolic).round(6)),
                'Stiffness Index h/(t_dia - t_sys)': float(((170/100)/np.mean(time_delay_systolic_diastolic)).round(6)),
                'Average Crest Time (t_sys - t_0)': float(np.mean(crest_time).round(6)),
                'Average (t_dic - t_dia)': float(np.mean(time_delays_dd).round(6)),
                't_ratio (t_sys-t(0)) / (t_dia-t_dic)': float((np.mean(crest_time)/np.mean(time_delays_dd)).round(6)),
                'Average t_sys (t(dic)-t(0))': float(np.mean(t_sys).round(6)),
                'Average t_dia (t(0)-t(dic))': float(np.mean(t_dia).round(6)),
                'Average dw_75': float(np.mean(dw_75).round(6)),
                'Average dw_66': float(np.mean(dw_66).round(6)),
                'Average dw_50': float(np.mean(dw_50).round(6)),
                'Average dw_33': float(np.mean(dw_33).round(6)),
                'Average dw_25': float(np.mean(dw_25).round(6)),
                'Average dw_10': float(np.mean(dw_10).round(6)),
                'Average sw_75': float(np.mean(sw_75).round(6)),
                'Average sw_66': float(np.mean(sw_66).round(6)),
                'Average sw_50': float(np.mean(sw_50).round(6)),
                'Average sw_33': float(np.mean(sw_33).round(6)),
                'Average sw_25': float(np.mean(sw_25).round(6)),
                'Average sw_10': float(np.mean(sw_10).round(6)),
                'Average w_75': float((np.mean(sw_75)+np.mean(dw_75)).round(6)),
                'Average w_66': float((np.mean(sw_66)+np.mean(dw_66)).round(6)),
                'Average w_50': float((np.mean(sw_50)+np.mean(dw_50)).round(6)),
                'Average w_33': float((np.mean(sw_33)+np.mean(dw_33)).round(6)),
                'Average w_25': float((np.mean(sw_25)+np.mean(dw_25)).round(6)),
                'Average w_10': float((np.mean(sw_10)+np.mean(dw_10)).round(6)),
                'Average dw_75/sw_75': float((np.mean(dw_75)/np.mean(sw_75)).round(6)),
                'Average dw_66/sw_66': float((np.mean(dw_66)/np.mean(sw_66)).round(6)),
                'Average dw_50/sw_50': float((np.mean(dw_50)/np.mean(sw_50)).round(6)),
                'Average dw_33/sw_33': float((np.mean(dw_33)/np.mean(sw_33)).round(6)),
                'Average dw_25/sw_25': float((np.mean(dw_25)/np.mean(sw_25)).round(6)),
                'Average dw_10/sw_10': float((np.mean(dw_10)/np.mean(sw_10)).round(6)),
                'Average IPR': float(np.mean(ipr).round(6)),
                'Average pulse amplitude': float(np.mean(pulse_amplitude).round(6)),
                'Average reflection index': float(np.mean(reflection_index).round(6)),
                'Average systolic area': float(np.mean(systolic_area).round(6)),
                'Average diastolic area': float(np.mean(diastolic_area).round(6)),
                'IPA inflection point': float((np.mean(diastolic_area)/np.mean(systolic_area)).round(6)),
                'Average maximum slope': float(np.mean(normalized_max_slope).round(6)),

            }
        }

        with open(f'{"/".join(self.video_path.split("/")[:-1])}/config.yml', "w", encoding="utf-8") as file:
            yaml.dump(data_config, file, allow_unicode=True,
                      default_flow_style=False)

        return [time_ppg, filtered_ppg/np.max(filtered_ppg), filtered_signal_fft1[0], filtered_signal_fft1[1], snr_filtered_ppg, heart_rate_filtered_ppg], [time_spg, filtered_spg/np.max(filtered_spg), filtered_spg_fft[0], filtered_spg_fft[1], snr_filtered_spg, heart_rate_spg], [excel_data[0], filtered_excel/np.max(filtered_excel), filtered_excel_fft[0], filtered_excel_fft[1], snr_filtered_excel, heart_rate_excel], avg_time_delay
        # return data_feature


if __name__ == "__main__":

    # exposure_time = 6000  # us
    # size = 200
    # size_ppg = size  # W x H 200
    # size_spg = 100  # W x H 100
    # fps = 120  # Hz 88
    # cache = False
    # cut_time_delay = 0.2

    # folder = "01_TI/2025-01-28 13_46_36 tee 8203 120 (200, 200) rate-10"
    # video_path = f"อาสาสมัคร/{folder}/video-0000.avi"
    # serial_path = f"อาสาสมัคร/{folder}/serial.xlsx"

    # folder = "2025-01-28 13_49_40 tee 6000 120 (200, 200) rate-10"
    # video_path = f"storage/{folder}/video-0000.avi"
    # serial_path = f"storage/{folder}/serial.xlsx"

    # analysis = Analysis_PPG_SPG(
    #     video_path, serial_path, size_ppg, size_spg, exposure_time, fps, cache, cut_time_delay)
    # ppg, spg, excel, avg_time_delay = analysis.main()

    # plt.show()

    folder_list = [f for f in os.listdir(
        'อาสาสมัคร') if os.path.isdir(os.path.join('อาสาสมัคร', f))]

    # snr_ppg_list = []
    # snr_spg_list = []
    # snr_excel_list = []
    # avg_time_delay_list = []

    # data_save_excel = []

    # for folder in folder_list:
    #     folder_list_2 = [f for f in os.listdir(
    #         f'อาสาสมัคร/{folder}') if os.path.isdir(os.path.join(f'อาสาสมัคร/{folder}', f))]
    #     for folder_2 in folder_list_2:

    #         video_path = f"อาสาสมัคร/{folder}/{folder_2}/video-0000.avi"
    #         serial_path = f"อาสาสมัคร/{folder}/{folder_2}/serial.xlsx"
    #         print(video_path, serial_path)
    #         size_ppg = 200
    #         size_spg = 200
    #         exposure_time = int(folder_2.split(" ")[3])
    #         fps = 120
    #         cache = False
    #         cut_time_delay = 0.2
    #         print(folder, folder_2, exposure_time)
    #         analysis = Analysis_PPG_SPG(
    #             video_path, serial_path, size_ppg, size_spg, exposure_time, fps, cache, cut_time_delay)
    #         data_feature = analysis.main()
    #         data_save_excel.append(data_feature)

    # # save data featrue to excel
    # exporter = DataExporter()

    # exporter.add_multiple_records(data_save_excel)
    # exporter.save_to_excel("my_data.xlsx")

    folder_list = [f for f in os.listdir(
        'storage') if os.path.isdir(os.path.join('storage', f))]
    for folder in [folder_list[3]]:
        # for folder in folder_list[3:]:
        video_path = f"storage/{folder}/video-0000.avi"
        serial_path = f"storage/{folder}/serial.xlsx"
        print(video_path, serial_path)
        size_ppg = 150
        size_spg = 150
        exposure_time = int(folder.split(" ")[3], 10)
        fps = 120
        cache = False
        cut_time_delay = 0.2
        print(folder, folder, exposure_time)
        analysis = Analysis_PPG_SPG(
            video_path, serial_path, size_ppg, size_spg, exposure_time, fps, cache, cut_time_delay)
        ppg = analysis.main()

    # fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    # ax.plot(snr_ppg_list, label='SNR iPPG')
    # ax.plot(snr_spg_list, label='SNR SPG')
    # ax.plot(snr_excel_list, label='SNR cPPG')
    # ax.set_xlabel('Folder')
    # ax.set_ylabel('SNR (dB)')
    # ax.legend()
    # fig.tight_layout()

    # fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    # ax.plot(avg_time_delay_list, label='Time Delay')
    # ax.set_xlabel('Folder')
    # ax.set_ylabel('Time Delay (s)')
    # ax.legend()
    # fig.tight_layout()

    # plt.show()
