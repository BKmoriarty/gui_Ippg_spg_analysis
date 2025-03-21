# Save this as pressure_analyzer.py
import numpy as np
from scipy.signal import find_peaks, argrelextrema
import argparse
import matplotlib.pyplot as plt


class PressureAnalyzer:
    def __init__(self, time_data, pressure_data, sample_rate=100):
        """
        Initialize with time and pressure data arrays
        time_data: array of time points
        pressure_data: array of pressure values
        sample_rate: samples per second (default 100 Hz)
        """
        self.sample_rate = sample_rate
        self.time = np.array(time_data)
        self.pressure = np.array(pressure_data)

        # Initialize results
        self.systolic_idx = None
        self.onset_idx = None
        self.diastolic_idx = None
        self.dicrotic_idx = None

        self.first_derivative = None  # First derivative of pressure signal
        self.ms_idx = None  # Maximum slope indices
        self.second_derivative = None
        self.second_deriv_peaks_idx = None
        self.b_points_idx = None

    def calculate_first_derivative(self):
        """Calculate the first derivative of the pressure signal."""
        self.first_derivative = np.gradient(self.pressure, self.time)

    def calculate_second_derivative(self):
        """Calculate the second derivative of the pressure signal."""
        if self.first_derivative is None:
            self.calculate_first_derivative()
        if self.second_derivative is None:
            self.second_derivative = np.gradient(
                self.first_derivative, self.time)

    def find_pressure_features(self, min_distance=0.5, height_percentile=75, diastolic_prominence=0.1, onset_search_range=0.1,):
        """
        Find systolic peaks, diastolic peaks (second peak after systolic), and dicrotic notches
        min_distance: minimum time between peaks in seconds
        height_percentile: percentile threshold for systolic peaks
        diastolic_prominence: minimum prominence for diastolic peak detection
        """
        distance_samples = int(min_distance * self.sample_rate)
        # Adjustable search range
        onset_range_samples = int(onset_search_range * self.sample_rate)

        # Find systolic peaks
        self.systolic_idx, _ = find_peaks(self.pressure,
                                          height=np.percentile(
                                              self.pressure, height_percentile),
                                          distance=distance_samples)

        # Calculate derivatives if not already done
        self.calculate_first_derivative()
        self.calculate_second_derivative()

        # Finding dicrotic notches
        self.dicrotic_idx = []
        for i in range(len(self.systolic_idx)-1):
            segment = self.pressure[self.systolic_idx[i]                                    :self.systolic_idx[i+1]]
            local_minima = argrelextrema(segment, np.less)[0]
            if len(local_minima) > 0:
                notch_idx = local_minima[0] + self.systolic_idx[i]
                if (self.pressure[notch_idx] < self.pressure[self.systolic_idx[i]] - diastolic_prominence):
                    self.dicrotic_idx.append(notch_idx)
        self.dicrotic_idx = np.array(
            self.dicrotic_idx) if self.dicrotic_idx else np.array([])

        # Finding onset peaks with adjustable search range
        self.onset_idx = []
        for i in range(len(self.systolic_idx)):
            if i < len(self.systolic_idx) - 1:
                start_idx = self.dicrotic_idx[i] if i < len(
                    self.dicrotic_idx) else self.systolic_idx[i]
                # Adjust search start
                start_idx = max(0, start_idx - onset_range_samples)
                end_idx = min(len(self.pressure), self.systolic_idx[i + 1])

                segment = self.pressure[start_idx:end_idx]
                if len(segment) > 0:
                    min_idx = np.argmin(segment) + start_idx
                    self.onset_idx.append(min_idx)
            else:
                segment = self.pressure[self.systolic_idx[i]:]
                if len(segment) > 0:
                    min_idx = np.argmin(segment) + self.systolic_idx[i]
                    self.onset_idx.append(min_idx)
        self.onset_idx = np.array(self.onset_idx)

        # Find diastolic peaks (second peak after systolic, before end of cycle)
        self.diastolic_idx = []
        for i in range(len(self.systolic_idx)):
            if i < len(self.systolic_idx) - 1:
                # Look for peaks in the segment after the systolic peak up to the next systolic peak
                start_idx = self.systolic_idx[i]
                end_idx = self.systolic_idx[i + 1]
                segment = self.pressure[start_idx:end_idx]

                # Find all local maxima in the segment (excluding the systolic peak)
                local_maxima, _ = find_peaks(segment, distance=int(0.1 * self.sample_rate),  # Minimum 0.1s apart
                                             prominence=diastolic_prominence)

                if len(local_maxima) >= 1:
                    # Take the first significant peak after the systolic peak (second peak in the cycle)
                    # Adjust index to account for segment start
                    diastolic_idx = local_maxima[0] + start_idx
                    self.diastolic_idx.append(diastolic_idx)
            else:
                # For the last cycle, look after the systolic peak to the end
                segment = self.pressure[self.systolic_idx[i]:]
                local_maxima, _ = find_peaks(segment, distance=int(0.1 * self.sample_rate),
                                             prominence=diastolic_prominence)
                if len(local_maxima) >= 1:
                    diastolic_idx = local_maxima[0] + self.systolic_idx[i]
                    self.diastolic_idx.append(diastolic_idx)
        self.diastolic_idx = np.array(self.diastolic_idx, dtype=int)

        # Find maximum slope (ms) points in the first derivative
        self.ms_idx = []
        for i in range(len(self.systolic_idx)):
            if i == 0:
                start_idx = 0
            else:
                start_idx = self.systolic_idx[i-1]
            end_idx = self.systolic_idx[i]
            segment = self.first_derivative[start_idx:end_idx]
            if len(segment) > 0:
                ms_idx = np.argmax(segment) + start_idx
                self.ms_idx.append(ms_idx)
        self.ms_idx = np.array(self.ms_idx)

    def get_normalized_max_slope(self):
        """Calculate the normalized maximum slope x'(ms)/(x(sys) - x(0)) for each cycle."""
        if self.systolic_idx is None or self.onset_idx is None or self.ms_idx is None:
            raise RuntimeError("Run find_pressure_features() first")

        # Ensure lengths match (trim to shortest length to avoid index errors)
        min_length = min(len(self.ms_idx), len(
            self.systolic_idx), len(self.onset_idx))
        ms_idx = self.ms_idx[:min_length]
        sys_idx = self.systolic_idx[:min_length]
        onset_idx = self.onset_idx[:min_length]

        # Calculate normalized maximum slope
        max_slope_values = self.first_derivative[ms_idx]
        systolic_values = self.pressure[sys_idx]
        onset_values = self.pressure[onset_idx]
        amplitude_diff = systolic_values - onset_values

        # Avoid division by zero
        with np.errstate(divide='warn', invalid='warn'):
            normalized_max_slope = max_slope_values / amplitude_diff
            # Replace infinities or NaNs with a default value (e.g., 0) if needed
            normalized_max_slope = np.where(np.isfinite(
                normalized_max_slope), normalized_max_slope, 0)

        return normalized_max_slope

    def get_results(self):
        """Return the values at detected points"""

        if self.systolic_idx is None:
            raise RuntimeError("Run find_pressure_features() first")

        # Find second derivative peaks (default parameters)
        second_deriv_peaks_idx, second_deriv_peaks = self.find_second_derivative_peaks()
        b_indices, b_values = self.find_b_points()
        b_indices = b_indices.astype(int)

        return {
            'systolic': {
                'times': self.time[self.systolic_idx],
                'values': self.pressure[self.systolic_idx]
            },
            'onset': {
                'times': self.time[self.onset_idx],
                'values': self.pressure[self.onset_idx]
            },
            'diastolic': {
                'times': self.time[self.diastolic_idx],
                'values': self.pressure[self.diastolic_idx]
            },
            'dicrotic': {
                'times': self.time[self.dicrotic_idx],
                'values': self.pressure[self.dicrotic_idx]
            },
            'ms': {
                'times': self.time[self.ms_idx],
                'values': self.first_derivative[self.ms_idx]
            },
            'normalized_max_slope': self.get_normalized_max_slope(),
            'second_derivative': self.get_second_derivative(),
            'second_deriv_peaks': {'times': self.time[second_deriv_peaks_idx], 'values': second_deriv_peaks},
            'b_points': {'times': self.time[b_indices], 'values': b_values}
        }

    def plot_results(self, fig=None, ax=None):
        """Plot the waveform with detected features"""

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(self.time, self.pressure, 'b-', label='Pressure Waveform')
        ax.plot(self.time[self.systolic_idx], self.pressure[self.systolic_idx],
                'go', label='Systolic Peaks')
        ax.plot(self.time[self.onset_idx], self.pressure[self.onset_idx],
                'ko', label='Onset Peaks')
        ax.plot(self.time[self.diastolic_idx], self.pressure[self.diastolic_idx],
                'mo', label='Diastolic Peaks')
        ax.plot(self.time[self.dicrotic_idx], self.pressure[self.dicrotic_idx],
                'ro', label='Dicrotic Notches')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Pressure Waveform Analysis')
        ax.legend()
        ax.grid(True)

        return fig, ax

    def plot_first_derivative(self, fig=None, ax=None):
        if self.first_derivative is None:
            raise RuntimeError(
                "First derivative not calculated. Run find_pressure_features() first.")

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(self.time, self.first_derivative,
                'b-', label='First Derivative')
        ax.plot(self.time[self.ms_idx], self.first_derivative[self.ms_idx],
                'co', label='Max Slope (ms)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('dP/dt')
        ax.set_title('First Derivative Analysis')
        ax.legend()
        ax.grid(True)

        return fig, ax

    def plot_results_all(self, fig=None, ax=None):
        if fig is None or ax is None:
            fig, (ax1, ax2, ax3) = plt.subplots(
                3, 1, figsize=(12, 8), sharex=True)

        else:
            ax1, ax2, ax3 = ax

        if self.second_derivative is None:
            raise RuntimeError(
                "Second derivative not calculated. Run find_pressure_features() first.")
        if self.second_deriv_peaks_idx is None:
            self.find_second_derivative_peaks()
        if self.b_points_idx is None:
            self.find_b_points()

        # Plot original pressure waveform
        ax1.plot(self.time, self.pressure, 'g-', label='Pressure Waveform')
        ax1.plot(self.time[self.systolic_idx],
                 self.pressure[self.systolic_idx], 'go', label='Systolic Peaks')
        ax1.plot(self.time[self.onset_idx],
                 self.pressure[self.onset_idx], 'ko', label='Onset Peaks')
        ax1.plot(self.time[self.diastolic_idx],
                 self.pressure[self.diastolic_idx], 'mo', label='Diastolic Peaks')
        ax1.plot(self.time[self.dicrotic_idx],
                 self.pressure[self.dicrotic_idx], 'ro', label='Dicrotic Notches')
        ax1.set_ylabel('Pressure Amplitude')
        ax1.legend()
        ax1.grid(True)

        # Plot first derivative with ms points
        ax2.plot(self.time, self.first_derivative,
                 'b-', label='First Derivative')
        ax2.plot(self.time[self.ms_idx],
                 self.first_derivative[self.ms_idx], 'co', label='Max Slope (ms)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('dP/dt')
        ax2.legend()
        ax2.grid(True)

        # Plot second derivative
        ax3.plot(self.time, self.second_derivative,
                 'r-', label='Second Derivative')
        ax3.plot(self.time[self.second_deriv_peaks_idx],
                 self.second_derivative[self.second_deriv_peaks_idx], 'ro', label='Wave Peaks')
        ax3.plot(self.time[self.b_points_idx],
                 self.second_derivative[self.b_points_idx], 'go', label='b Points')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('d²P/dt²')
        ax3.set_title('Second Derivative Analysis')
        ax3.legend()
        ax3.grid(True)

        fig.suptitle('Pressure Waveform and First Derivative Analysis')
        return fig, (ax1, ax2)

    def find_second_derivative_peaks(self, height=None, distance=0.7):
        """
        Find wave peaks in the second derivative signal.

        Parameters:
        - height: Minimum height for peaks (optional, auto-set if None based on signal range)
        - distance: Minimum distance between peaks in seconds (default 0.1 s)

        Returns:
        - Tuple (peak_indices, peak_values) of detected peaks in the second derivative
        """
        if self.second_derivative is None:
            self.calculate_second_derivative()

        # Convert distance from seconds to samples
        distance_samples = int(distance * self.sample_rate)

        # Set height threshold if not provided (e.g., 10% of max absolute value)
        if height is None:
            height = 0.1 * np.max(np.abs(self.second_derivative))

        # Find peaks in the second derivative
        peak_indices, _ = find_peaks(
            self.second_derivative, height=height, distance=distance_samples)
        peak_values = self.second_derivative[peak_indices]

        self.second_deriv_peaks_idx = peak_indices
        return peak_indices, peak_values

    def find_b_points(self, time_distance=1.2):
        """
        Find the 'b' point (lowest point in second derivative) for each pulse cycle.

        Parameters:
        - time_distance: Maximum time difference for pairing onset and dicrotic (default 0.5 s)

        Returns:
        - Tuple (b_indices, b_values) of 'b' points per cycle
        """
        if self.second_derivative is None:
            self.calculate_second_derivative()
        if self.onset_idx is None or self.dicrotic_idx is None:
            raise RuntimeError("Run find_pressure_features() first")

        onset_sorted = self.onset_idx[np.argsort(self.time[self.onset_idx])]
        dicrotic_sorted = self.dicrotic_idx[np.argsort(
            self.time[self.dicrotic_idx])]

        b_indices = []
        b_values = []
        i, j = 0, 0
        n_onset = len(onset_sorted)
        n_dicrotic = len(dicrotic_sorted)

        while i < n_onset and j < n_dicrotic:
            onset_idx = onset_sorted[i]
            dicrotic_idx = dicrotic_sorted[j]
            onset_time = self.time[onset_idx]
            dicrotic_time = self.time[dicrotic_idx]

            if abs(dicrotic_time - onset_time) < time_distance and dicrotic_time > onset_time:
                # Segment second derivative from onset to dicrotic notch
                segment_second_deriv = self.second_derivative[onset_idx:dicrotic_idx + 1]

                # Find the lowest point (minimum)
                b_idx_rel = np.argmin(segment_second_deriv)
                b_idx = onset_idx + b_idx_rel
                b_value = self.second_derivative[b_idx]
                b_indices.append(b_idx)
                b_values.append(b_value)
                i += 1
                j += 1
            elif dicrotic_time - onset_time > time_distance:
                i += 1
            else:
                j += 1

        self.b_points_idx = np.array(b_indices)
        return np.array(b_indices), np.array(b_values)

    def get_time_delay_systolic_diastolic(self):
        """
        Calculate the time delay between systolic and diastolic peaks for each cycle
        Returns a list of time delays (t_dia - t_sys) in seconds
        """
        if self.systolic_idx is None or self.diastolic_idx is None:
            raise RuntimeError("Run find_pressure_features() first")

        # Sort indices by time
        systolic_sorted = self.systolic_idx[np.argsort(
            self.time[self.systolic_idx])]
        diastolic_sorted = self.diastolic_idx[np.argsort(
            self.time[self.diastolic_idx])]

        time_delays = []
        i, j = 0, 0  # Indices for systolic and diastolic peaks
        n_systolic = len(systolic_sorted)
        n_diastolic = len(diastolic_sorted)

        while i < n_systolic and j < n_diastolic:
            systolic_time = self.time[systolic_sorted[i]]
            diastolic_time = self.time[diastolic_sorted[j]]

            # Match diastolic peak to the nearest systolic peak in time
            # Reasonable time window (e.g., 1 second)
            if abs(diastolic_time - systolic_time) < 1.0:
                time_delay = diastolic_time - systolic_time
                time_delays.append(time_delay)
                i += 1
                j += 1
            elif diastolic_time < systolic_time:
                j += 1  # Move to next diastolic if it's before the current systolic
            else:
                i += 1  # Move to next systolic if it's before the current diastolic

        return time_delays

    def detect_pattern(self):
        """
        Detect the pattern Systolic Peak -> Dicrotic Notch -> Diastolic Peak -> Onset
        Returns the count of complete patterns found
        """
        self.pattern_count = 0

        # Ensure all indices are available and sorted
        if (self.systolic_idx is None or self.dicrotic_idx is None or
                self.diastolic_idx is None or self.onset_idx is None):
            raise RuntimeError("Run find_pressure_features() first")

        # Sort indices by time (in case they're not already sorted)
        systolic_sorted = self.systolic_idx[np.argsort(
            self.time[self.systolic_idx])]
        dicrotic_sorted = self.dicrotic_idx[np.argsort(
            self.time[self.dicrotic_idx])] if len(self.dicrotic_idx) > 0 else np.array([])
        diastolic_sorted = self.diastolic_idx[np.argsort(
            self.time[self.diastolic_idx])]
        onset_sorted = self.onset_idx[np.argsort(self.time[self.onset_idx])]

        # Match points across cycles
        i, j, k, l = 0, 0, 0, 0  # Indices for systolic, dicrotic, diastolic, onset
        n_systolic = len(systolic_sorted)
        n_dicrotic = len(dicrotic_sorted)
        n_diastolic = len(diastolic_sorted)
        n_onset = len(onset_sorted)

        # Minimum time gap between points in seconds (adjust as needed)
        min_time_gap = 0.1
        min_time_gap_samples = int(min_time_gap * self.sample_rate)

        while i < n_systolic:
            systolic_time = self.time[systolic_sorted[i]]

            # Find corresponding dicrotic notch (if any) after systolic peak with minimum time gap
            dicrotic_time = float('inf')
            while j < n_dicrotic and self.time[dicrotic_sorted[j]] <= systolic_time + min_time_gap_samples / self.sample_rate:
                j += 1
            if j < n_dicrotic:
                dicrotic_time = self.time[dicrotic_sorted[j]]
                j += 1

            # Find corresponding diastolic peak after dicrotic notch (or systolic if no dicrotic) with minimum time gap
            diastolic_time = float('inf')
            while k < n_diastolic and self.time[diastolic_sorted[k]] <= (dicrotic_time if dicrotic_time != float('inf') else systolic_time) + min_time_gap_samples / self.sample_rate:
                k += 1
            if k < n_diastolic:
                diastolic_time = self.time[diastolic_sorted[k]]
                k += 1

            # Find corresponding onset before systolic peak with minimum time gap
            onset_time = float('-inf')
            while l < n_onset and self.time[onset_sorted[l]] < systolic_time - min_time_gap_samples / self.sample_rate:
                onset_time = self.time[onset_sorted[l]]
                l += 1

            # Check if all points exist in the expected order with minimum time gaps
            if (dicrotic_time != float('inf') and diastolic_time != float('inf') and
                onset_time != float('-inf') and
                systolic_time > onset_time + min_time_gap_samples / self.sample_rate and
                dicrotic_time > systolic_time + min_time_gap_samples / self.sample_rate and
                    diastolic_time > dicrotic_time + min_time_gap_samples / self.sample_rate):
                self.pattern_count += 1

            i += 1

        return self.pattern_count

    def get_crest_time(self, cast_time_distance=0.5):
        """
        Calculate the Crest Time (CT) for each cycle, defined as t(sys) - t(0) (time from onset to systolic peak)
        Returns a list of crest times in seconds
        """
        if self.systolic_idx is None or self.onset_idx is None:
            raise RuntimeError("Run find_pressure_features() first")

        # Sort indices by time
        systolic_sorted = self.systolic_idx[np.argsort(
            self.time[self.systolic_idx])]
        onset_sorted = self.onset_idx[np.argsort(self.time[self.onset_idx])]

        crest_times = []
        i, j = 0, 0  # Indices for systolic and onset peaks
        n_systolic = len(systolic_sorted)
        n_onset = len(onset_sorted)

        while i < n_systolic and j < n_onset:
            systolic_time = self.time[systolic_sorted[i]]
            onset_time = self.time[onset_sorted[j]]

            # Match onset to the nearest systolic peak in time (within a reasonable window)
            # Reasonable time window (e.g., 1.5 seconds for CT)
            # print(i, j, systolic_time - onset_time)
            if abs(systolic_time - onset_time) < cast_time_distance:
                crest_time = systolic_time - onset_time
                # Ensure CT is positive (onset before systolic)
                if crest_time > 0:
                    # print("PUSH")
                    crest_times.append(crest_time)
                    i += 1
                    j += 1
                elif onset_time < systolic_time:
                    j += 1  # Move to next onset if it's before the current systolic
                else:
                    i += 1  # Move to next systolic if it's before the current onset
            elif (systolic_time - onset_time) > cast_time_distance:
                j += 1
            else:
                i += 1  # Move to next systolic if it's before the current onset
        return crest_times

    def get_time_delay_dicrotic_diastolic(self, notch_to_dia_distance=0.5):
        """
        Calculate the time delay between dicrotic notch and diastolic peak for each cycle
        Returns a list of time delays (t_dia - t_dicrotic) in seconds
        """
        if self.dicrotic_idx is None or self.diastolic_idx is None:
            raise RuntimeError("Run find_pressure_features() first")

        # Sort indices by time
        dicrotic_sorted = self.dicrotic_idx[np.argsort(
            self.time[self.dicrotic_idx])]
        diastolic_sorted = self.diastolic_idx[np.argsort(
            self.time[self.diastolic_idx])]

        time_delays = []
        i, j = 0, 0  # Indices for dicrotic notch and diastolic peaks
        n_dicrotic = len(dicrotic_sorted)
        n_diastolic = len(diastolic_sorted)

        while i < n_dicrotic and j < n_diastolic:
            dicrotic_time = self.time[dicrotic_sorted[i]]
            diastolic_time = self.time[diastolic_sorted[j]]

            # Match diastolic peak to the nearest dicrotic notch in time
            # Reasonable time window (e.g., 1 second)
            print(i, j, diastolic_time - dicrotic_time)
            if abs(diastolic_time - dicrotic_time) < notch_to_dia_distance:
                time_delay = diastolic_time - dicrotic_time
                # Ensure time delay is positive (diastolic after dicrotic)
                if time_delay > 0:
                    print("PUSH")
                    time_delays.append(time_delay)
                    i += 1
                    j += 1
                elif dicrotic_time < diastolic_time:
                    j += 1  # Move to next onset if it's before the current systolic
                else:
                    i += 1  # Move to next systolic if it's before the current onset
            elif (diastolic_time - dicrotic_time) < notch_to_dia_distance:
                j += 1  # Move to next diastolic if it's before the current dicrotic
            else:
                i += 1  # Move to next dicrotic if it's before the current diastolic

        return time_delays

    def Derivative(self, xlist, ylist):
        yprime = np.diff(ylist)/np.diff(xlist)
        xprime = []
        for i in range(len(yprime)):
            xtemp = (xlist[i+1]+xlist[i])/2
            xprime = np.append(xprime, xtemp)
        return xprime, yprime

    def get_t_sys(self, notch_time_distance=1):
        """
        Calculate the time to dicrotic notch (t_sys) for each cycle, defined as t(dic) - t(0).
        Returns a list of t_sys times in seconds.

        Parameters:
        - notch_time_distance: Maximum allowed time difference between onset and dicrotic notch (default 0.5s)
        """
        if self.onset_idx is None or self.dicrotic_idx is None:
            raise RuntimeError("Run find_pressure_features() first")

        # Sort indices by time
        onset_sorted = self.onset_idx[np.argsort(self.time[self.onset_idx])]
        dicrotic_sorted = self.dicrotic_idx[np.argsort(
            self.time[self.dicrotic_idx])]

        t_sys_times = []
        i, j = 0, 0  # Indices for onset and dicrotic peaks
        n_onset = len(onset_sorted)
        n_dicrotic = len(dicrotic_sorted)

        while i < n_onset and j < n_dicrotic:
            onset_time = self.time[onset_sorted[i]]
            dicrotic_time = self.time[dicrotic_sorted[j]]

            # Match onset to the nearest dicrotic notch within a reasonable window
            time_diff = dicrotic_time - onset_time
            print(i, j, time_diff)
            if abs(time_diff) < notch_time_distance:
                # Ensure t_sys is positive (dicrotic notch after onset)
                if time_diff > 0:
                    t_sys_times.append(time_diff)
                    i += 1
                    j += 1
                elif onset_time > dicrotic_time:
                    j += 1  # Move to next dicrotic if it's before the current onset
                else:
                    i += 1  # Move to next onset if it's after the current dicrotic
            elif time_diff > notch_time_distance:
                i += 1  # Move to next onset if dicrotic is too far ahead
            else:
                j += 1  # Move to next dicrotic if onset is too far ahead

        return np.array(t_sys_times)

    def get_t_dia(self, pulse_end_time_distance=0.5):
        """
        Calculate the time from dicrotic notch to pulse end (t_dia) for each cycle, defined as T - t(dic).
        T is the time of the next onset (pulse end).
        Returns a list of t_dia times in seconds.

        Parameters:
        - pulse_end_time_distance: Maximum allowed time difference between dicrotic notch and next onset (default 0.5s)
        """
        if self.dicrotic_idx is None or self.onset_idx is None:
            raise RuntimeError("Run find_pressure_features() first")

        # Sort indices by time
        dicrotic_sorted = self.dicrotic_idx[np.argsort(
            self.time[self.dicrotic_idx])]
        onset_sorted = self.onset_idx[np.argsort(self.time[self.onset_idx])]

        t_dia_times = []
        i, j = 0, 0  # i for dicrotic, j for onset (next pulse end)
        n_dicrotic = len(dicrotic_sorted)
        n_onset = len(onset_sorted)

        # Start j at 1 since we need the next onset after the dicrotic notch
        while i < n_dicrotic and j < n_onset - 1:
            dicrotic_time = self.time[dicrotic_sorted[i]]
            # T is the next onset
            next_onset_time = self.time[onset_sorted[j + 1]]

            time_diff = next_onset_time - dicrotic_time
            # print(i, j, time_diff)
            if abs(time_diff) < pulse_end_time_distance:
                # Ensure t_dia is positive (next onset after dicrotic notch)
                if time_diff > 0:
                    t_dia_times.append(time_diff)
                    i += 1
                    j += 1
                elif dicrotic_time < next_onset_time:
                    j += 1  # Move to next onset if it's before the current dicrotic
                else:
                    i += 1  # Move to next dicrotic if it's after the current next onset
            elif time_diff > pulse_end_time_distance:
                i += 1  # Move to next dicrotic if next onset is too far ahead
            else:
                j += 1  # Move to next onset if dicrotic is too far ahead

        return np.array(t_dia_times)

    def get_diastolic_width(self, percent=50, time_distance=0.5):
        """
        Calculate the Diastolic Width (DW##) for each cycle, defined as t(x = ##) - t(sys).
        t(x = ##) is the time where the signal falls by ##% of the amplitude on the falling limb.

        Parameters:
        - percent: Percentage of amplitude drop (e.g., 50 for DW50, range 0-100)
        - time_distance: Maximum allowed time difference for pairing (default 0.5s)

        Returns:
        - Array of DW## times in seconds
        """
        if self.systolic_idx is None or self.onset_idx is None:
            raise RuntimeError("Run find_pressure_features() first")

        if not 0 <= percent <= 100:
            raise ValueError("Percent must be between 0 and 100")

        systolic_sorted = self.systolic_idx[np.argsort(
            self.time[self.systolic_idx])]
        onset_sorted = self.onset_idx[np.argsort(self.time[self.onset_idx])]

        dw_times = []
        i, j = 0, 0
        n_systolic = len(systolic_sorted)
        n_onset = len(onset_sorted)

        while i < n_systolic and j < n_onset:
            sys_idx = systolic_sorted[i]
            onset_idx = onset_sorted[j]
            sys_time = self.time[sys_idx]
            onset_time = self.time[onset_idx]

            if abs(sys_time - onset_time) < time_distance and sys_time > onset_time:
                sys_value = self.pressure[sys_idx]
                onset_value = self.pressure[onset_idx]
                amplitude = sys_value - onset_value

                # Target amplitude: drops by percent% from sys_value
                target_value = sys_value - (percent / 100) * amplitude

                end_idx = systolic_sorted[i + 1] if i + \
                    1 < n_systolic else len(self.pressure)
                segment_time = self.time[sys_idx:end_idx]
                segment_pressure = self.pressure[sys_idx:end_idx]

                crossing_idx = np.where(segment_pressure <= target_value)[0]
                if len(crossing_idx) > 0:
                    t_x = segment_time[crossing_idx[0]]
                    dw = t_x - sys_time
                    if dw > 0:
                        dw_times.append(dw)
                i += 1
                j += 1
            elif sys_time - onset_time > time_distance:
                j += 1
            else:
                i += 1

        return np.array(dw_times)

    def get_systolic_width(self, percent=50, time_distance=0.5):
        """
        Calculate the Systolic Width (SW##) for each cycle, defined as t(x = ##) - t(0).
        t(x = ##) is the time where the signal rises to ##% of the amplitude on the rising limb.

        Parameters:
        - percent: Percentage of amplitude rise (e.g., 50 for SW50, range 0-100)
        - time_distance: Maximum allowed time difference for pairing (default 0.5s)

        Returns:
        - Array of SW## times in seconds, where higher percent means longer time (SW75 > SW10)
        """
        if self.systolic_idx is None or self.onset_idx is None:
            raise RuntimeError("Run find_pressure_features() first")

        if not 0 <= percent <= 100:
            raise ValueError("Percent must be between 0 and 100")

        systolic_sorted = self.systolic_idx[np.argsort(
            self.time[self.systolic_idx])]
        onset_sorted = self.onset_idx[np.argsort(self.time[self.onset_idx])]

        sw_times = []
        i, j = 0, 0
        n_systolic = len(systolic_sorted)
        n_onset = len(onset_sorted)

        while i < n_systolic and j < n_onset:
            sys_idx = systolic_sorted[i]
            onset_idx = onset_sorted[j]
            sys_time = self.time[sys_idx]
            onset_time = self.time[onset_idx]

            if abs(sys_time - onset_time) < time_distance and sys_time > onset_time:
                sys_value = self.pressure[sys_idx]
                onset_value = self.pressure[onset_idx]
                amplitude = sys_value - onset_value

                # Target amplitude: rises to percent% of amplitude from onset_value
                target_value = onset_value + (percent / 100) * amplitude

                # Define the rising limb segment (onset to sys)
                segment_time = self.time[onset_idx:sys_idx + 1]
                segment_pressure = self.pressure[onset_idx:sys_idx + 1]

                # Find the first point where pressure reaches or exceeds target_value
                crossing_idx = np.where(segment_pressure >= target_value)[0]
                if len(crossing_idx) > 0:
                    t_x = segment_time[crossing_idx[0]]
                    sw = t_x - onset_time  # Time from onset to ##% rise
                    if sw > 0:
                        sw_times.append(sw)
                i += 1
                j += 1
            elif sys_time - onset_time > time_distance:
                j += 1
            else:
                i += 1

        return np.array(sw_times)

    def get_ipr(self):
        """
        Calculate the Instantaneous Pulse Rate (IPR) for each cycle, defined as 60 / T.
        T is the time period between consecutive pulse onsets (t(0) to next t(0)).

        Returns:
        - Array of IPR values in beats per minute (bpm)
        """
        if self.onset_idx is None or len(self.onset_idx) < 2:
            raise RuntimeError("Need at least 2 onset points to calculate IPR")

        # Sort onset indices by time
        onset_sorted = self.onset_idx[np.argsort(self.time[self.onset_idx])]
        onset_times = self.time[onset_sorted]

        # Calculate periods (T) between consecutive onsets
        T = np.diff(onset_times)  # Time differences in seconds

        # Compute IPR = 60 / T, handle division by zero
        with np.errstate(divide='warn', invalid='warn'):
            ipr = 60 / T
            # Replace infinities or NaNs with NaN (or could use a max reasonable bpm, e.g., 300)
            ipr = np.where(np.isfinite(ipr), ipr, np.nan)

        return ipr

    def get_pulse_amplitude(self, time_distance=0.5):
        """
        Calculate the Pulse Wave Amplitude (Amp) for each cycle, defined as x(sys) - x(0).

        Parameters:
        - time_distance: Maximum allowed time difference for pairing onset and systolic (default 0.5s)

        Returns:
        - Array of amplitude values in the same units as the pressure signal
        """
        if self.systolic_idx is None or self.onset_idx is None:
            raise RuntimeError("Run find_pressure_features() first")

        systolic_sorted = self.systolic_idx[np.argsort(
            self.time[self.systolic_idx])]
        onset_sorted = self.onset_idx[np.argsort(self.time[self.onset_idx])]

        amplitudes = []
        i, j = 0, 0
        n_systolic = len(systolic_sorted)
        n_onset = len(onset_sorted)

        while i < n_systolic and j < n_onset:
            sys_idx = systolic_sorted[i]
            onset_idx = onset_sorted[j]
            sys_time = self.time[sys_idx]
            onset_time = self.time[onset_idx]

            # Pair onset and systolic within time_distance
            if abs(sys_time - onset_time) < time_distance and sys_time > onset_time:
                sys_value = self.pressure[sys_idx]
                onset_value = self.pressure[onset_idx]
                amp = sys_value - onset_value
                if amp > 0:  # Ensure positive amplitude
                    amplitudes.append(amp)
                i += 1
                j += 1
            elif sys_time - onset_time > time_distance:
                j += 1
            else:
                i += 1

        return np.array(amplitudes)

    def get_reflection_index(self, time_distance=1):
        """
        Calculate the Reflection Index (RI) for each cycle, defined as (x(dia) - x(0)) / (x(sys) - x(0)).

        Parameters:
        - time_distance: Maximum allowed time difference for pairing (default 0.5s)

        Returns:
        - Array of RI values (dimensionless ratio)
        """
        if self.systolic_idx is None or self.onset_idx is None or self.diastolic_idx is None:
            raise RuntimeError("Run find_pressure_features() first")

        systolic_sorted = self.systolic_idx[np.argsort(
            self.time[self.systolic_idx])]
        onset_sorted = self.onset_idx[np.argsort(self.time[self.onset_idx])]
        diastolic_sorted = self.diastolic_idx[np.argsort(
            self.time[self.diastolic_idx])]

        ri_values = []
        i, j, k = 0, 0, 0  # Pointers for onset, systolic, diastolic
        n_onset = len(onset_sorted)
        n_systolic = len(systolic_sorted)
        n_diastolic = len(diastolic_sorted)

        while i < n_onset and j < n_systolic and k < n_diastolic:
            onset_time = self.time[onset_sorted[i]]
            sys_time = self.time[systolic_sorted[j]]
            dia_time = self.time[diastolic_sorted[k]]

            # Check if all points are in the same cycle and in correct order
            if (sys_time > onset_time and dia_time > sys_time and
                abs(sys_time - onset_time) < time_distance and
                    abs(dia_time - sys_time) < time_distance):
                onset_value = self.pressure[onset_sorted[i]]
                sys_value = self.pressure[systolic_sorted[j]]
                dia_value = self.pressure[diastolic_sorted[k]]

                numerator = dia_value - onset_value
                denominator = sys_value - onset_value

                # Ensure denominator is positive and non-zero
                if denominator > 0:
                    ri = numerator / denominator
                    # RI should typically be between 0 and 1, but allow negative for rare cases
                    ri_values.append(ri)
                i += 1
                j += 1
                k += 1
            else:
                # Move the pointer of the earliest time forward
                times = [onset_time, sys_time, dia_time]
                min_idx = np.argmin(times)
                if min_idx == 0:
                    i += 1
                elif min_idx == 1:
                    j += 1
                else:
                    k += 1

        return np.array(ri_values)

    def get_systolic_area(self, time_distance=1):
        """
        Calculate the Systolic Area (A1) for each cycle, defined as the area under the curve
        from pulse onset (t(0)) to dicrotic notch (t(dic)), above the onset baseline.
        """
        if self.onset_idx is None or self.dicrotic_idx is None:
            raise RuntimeError("Run find_pressure_features() first")

        onset_sorted = self.onset_idx[np.argsort(self.time[self.onset_idx])]
        dicrotic_sorted = self.dicrotic_idx[np.argsort(
            self.time[self.dicrotic_idx])]

        a1_values = []
        i, j = 0, 0
        n_onset = len(onset_sorted)
        n_dicrotic = len(dicrotic_sorted)

        while i < n_onset and j < n_dicrotic:
            onset_idx = onset_sorted[i]
            dicrotic_idx = dicrotic_sorted[j]
            onset_time = self.time[onset_idx]
            dicrotic_time = self.time[dicrotic_idx]

            if abs(dicrotic_time - onset_time) < time_distance and dicrotic_time > onset_time:
                segment_time = self.time[onset_idx:dicrotic_idx + 1]
                segment_pressure = self.pressure[onset_idx:dicrotic_idx + 1]
                onset_value = self.pressure[onset_idx]
                # Subtract onset baseline to get area above x(0)
                segment_pressure_baseline = segment_pressure - onset_value
                a1 = np.trapz(segment_pressure_baseline, segment_time)
                if a1 > 0:  # Ensure positive area
                    a1_values.append(a1)
                else:
                    print(
                        f"Warning: Negative A1 ({a1:.4f}) at onset {onset_idx} to dicrotic {dicrotic_idx}")
                i += 1
                j += 1
            elif dicrotic_time - onset_time > time_distance:
                i += 1
            else:
                j += 1

        return np.array(a1_values)

    def get_diastolic_area(self, time_distance=0.8):
        """
        Calculate the Diastolic Area (A2) for each cycle, defined as the area under the curve
        from dicrotic notch (t(dic)) to pulse end (T, next onset), above the onset baseline.
        """
        if self.dicrotic_idx is None or self.onset_idx is None:
            raise RuntimeError("Run find_pressure_features() first")

        dicrotic_sorted = self.dicrotic_idx[np.argsort(
            self.time[self.dicrotic_idx])]
        onset_sorted = self.onset_idx[np.argsort(self.time[self.onset_idx])]

        a2_values = []
        i, j = 0, 0
        n_dicrotic = len(dicrotic_sorted)
        n_onset = len(onset_sorted)

        while i < n_dicrotic and j < n_onset - 1:
            dicrotic_idx = dicrotic_sorted[i]
            onset_idx = onset_sorted[j]  # Current onset for baseline
            next_onset_idx = onset_sorted[j + 1]
            dicrotic_time = self.time[dicrotic_idx]
            next_onset_time = self.time[next_onset_idx]

            if abs(next_onset_time - dicrotic_time) < time_distance and next_onset_time > dicrotic_time:
                segment_time = self.time[dicrotic_idx:next_onset_idx + 1]
                segment_pressure = self.pressure[dicrotic_idx:next_onset_idx + 1]
                # Use current onset as baseline
                onset_value = self.pressure[onset_idx]
                # Subtract onset baseline to get area above x(0)
                segment_pressure_baseline = segment_pressure - onset_value
                a2 = np.trapz(segment_pressure_baseline, segment_time)
                if a2 > 0:  # Ensure positive area
                    a2_values.append(a2)
                else:
                    print(
                        f"Warning: Negative A2 ({a2:.4f}) at dicrotic {dicrotic_idx} to next onset {next_onset_idx}")
                i += 1
                j += 1
            elif next_onset_time - dicrotic_time > time_distance:
                i += 1
            else:
                j += 1

        return np.array(a2_values)

    def get_second_derivative(self):
        """
        Calculate the second derivative of the pressure signal across all time points.

        Returns:
        - Array of second derivative values (d²x/dt²) in units of pressure / s²
        """
        if self.second_derivative is None:
            self.calculate_second_derivative()
        return self.second_derivative

    def plot_second_derivative(self, fig=None, ax=None):
        """Plot the second derivative of the pressure signal."""
        if self.second_derivative is None:
            raise RuntimeError(
                "Second derivative not calculated. Run find_pressure_features() first.")
        if self.second_deriv_peaks_idx is None:
            self.find_second_derivative_peaks()
        if self.b_points_idx is None:
            self.find_b_points()


        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(self.time, self.second_derivative,
                'b-', label='Second Derivative')
        ax.plot(self.time[self.second_deriv_peaks_idx],
                self.second_derivative[self.second_deriv_peaks_idx], 'ro', label='Wave Peaks')
        ax.plot(self.time[self.b_points_idx],
                self.second_derivative[self.b_points_idx], 'go', label='b Points')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('d²P/dt²')
        ax.set_title('Second Derivative Analysis')
        ax.legend()
        ax.grid(True)

        return fig, ax


if __name__ == "__main__":
    try:
        # Assuming synthetic or example data for testing; replace with your actual data paths if needed
        # Example time data (0 to 3 seconds, 100 Hz)
        t = np.load(
            "storage/2025-02-20 12_51_25 top 7000 120 (320, 320)/ippg/time_ppg.npy")
        pressure = np.load(
            "storage/2025-02-20 12_51_25 top 7000 120 (320, 320)/ippg/filtered_ppg.npy")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    parser = argparse.ArgumentParser(description="Analyze pressure waveforms")
    parser.add_argument("--sample_rate", type=int,
                        default=100, help="Sampling rate in Hz")

    args = parser.parse_args()

    analyzer = PressureAnalyzer(
        time_data=t, pressure_data=pressure, sample_rate=args.sample_rate)

    analyzer.find_pressure_features(
        min_distance=0.5, height_percentile=75, diastolic_prominence=0.1, onset_search_range=0.1)

    results = analyzer.get_results()

    print(f"Systolic peaks found: {len(results['systolic']['values'])}")
    print(f"Onset peaks found: {len(results['onset']['values'])}")
    print(f"Diastolic peaks found: {len(results['diastolic']['values'])}")
    print(f"Dicrotic notches found: {len(results['dicrotic']['values'])}")
    print(f"Maximum slope (ms) points found: {(results['ms']['values'])}")
    print(
        f"Number of Second Derivative Peaks: {len(results['second_deriv_peaks']['times'])}")
    print(f"Number of b Points: {len(results['b_points']['times'])}")

    # Detect and count the pattern
    pattern_count = analyzer.detect_pattern()
    print(
        f"Number of complete patterns (Systolic Peak -> Dicrotic Notch -> Diastolic Peak -> Onset): {pattern_count}")

    # # Calculate and print time delays between systolic and diastolic peaks
    # time_delays = analyzer.get_time_delay_systolic_diastolic()
    # print(f"Time delays (t_dia - t_sys) in seconds: {time_delays}")
    # if time_delays:
    #     print(f"Average time delay: {np.mean(time_delays):.4f} seconds")
    #     print(
    #         f"Standard deviation of time delays: {np.std(time_delays):.4f} seconds")

    # # Calculate and print Crest Time (t_sys - t_0)
    # crest_times = analyzer.get_crest_time()
    # for i in crest_times:
    #     print(i)
    # if crest_times:
    #     print(f"Average Crest Time: {np.mean(crest_times):.4f} seconds")
    #     print(
    #         f"Standard deviation of Crest Times: {np.std(crest_times):.4f} seconds")

    # # Calculate and print time delays between dicrotic notch and diastolic peak
    # time_delays_dd = analyzer.get_time_delay_dicrotic_diastolic()
    # print(f"Time delays (t_dia - t_dicrotic) in seconds: {time_delays_dd}")
    # if time_delays_dd:
    #     print(
    #         f"Average time delay (dicrotic to diastolic): {np.mean(time_delays_dd):.4f} seconds")
    #     print(
    #         f"Standard deviation of time delays (dicrotic to diastolic): {np.std(time_delays_dd):.4f} seconds")

    # # Add timing features
    # t_sys = analyzer.get_t_sys()
    # print(f"t_sys: {len(t_sys)} {t_sys}")

    # # Add timing features
    # t_dia = analyzer.get_t_dia()
    # print(f"t_dia: {len(t_dia)} {t_dia}")

    # dw_75 = analyzer.get_diastolic_width(percent=75)
    # print(f"dw 75: {len(dw_75)} {np.mean(dw_75)}")
    # dw_66 = analyzer.get_diastolic_width(percent=66)
    # print(f"dw 66: {len(dw_66)} {np.mean(dw_66)}")
    # dw_50 = analyzer.get_diastolic_width(percent=50)
    # print(f"dw 50: {len(dw_50)} {np.mean(dw_50)}")
    # dw_33 = analyzer.get_diastolic_width(percent=33)
    # print(f"dw 33: {len(dw_33)} {np.mean(dw_33)}")
    # dw_25 = analyzer.get_diastolic_width(percent=25)
    # print(f"dw 25: {len(dw_25)} {np.mean(dw_25)}")
    # dw_10 = analyzer.get_diastolic_width(percent=10)
    # print(f"dw 10: {len(dw_10)} {np.mean(dw_10)}")

    # sw_75 = analyzer.get_systolic_width(percent=75)
    # print(f"sw 75: {len(sw_75)} {np.mean(sw_75)} {np.max(sw_75)}")
    # sw_66 = analyzer.get_systolic_width(percent=66)
    # print(f"sw 66: {len(sw_66)} {np.mean(sw_66)} {np.max(sw_66)}")
    # sw_50 = analyzer.get_systolic_width(percent=50)
    # print(f"sw 50: {len(sw_50)} {np.mean(sw_50)} {np.max(sw_50)}")
    # sw_33 = analyzer.get_systolic_width(percent=33)
    # print(f"sw 33: {len(sw_33)} {np.mean(sw_33)} {np.max(sw_33)}")
    # sw_25 = analyzer.get_systolic_width(percent=25)
    # print(f"sw 25: {len(sw_25)} {np.mean(sw_25)} {np.max(sw_25)}")
    # sw_10 = analyzer.get_systolic_width(percent=10)
    # print(f"sw 10: {len(sw_10)} {np.mean(sw_10)} {np.max(sw_10)}")

    # ipr = analyzer.get_ipr()
    # print(f"ipr: {len(ipr)} {ipr}")

    # pulse_amplitude = analyzer.get_pulse_amplitude()
    # print(f"pulse_amplitude: {len(pulse_amplitude)} {pulse_amplitude}")

    # reflection_index = analyzer.get_reflection_index()
    # print(f"reflection_index: {len(reflection_index)} {reflection_index}")

    # systolic_area = analyzer.get_systolic_area()
    # print(f"systolic_area: {len(systolic_area)} {systolic_area}")

    # diastolic_area = analyzer.get_diastolic_area()
    # print(f"diastolic_area: {len(diastolic_area)} {diastolic_area}")

    # normalized_max_slope = analyzer.get_normalized_max_slope()
    # print(f"Normalized maximum slope values: {normalized_max_slope}")

    # analyzer.plot_results()
    # analyzer.plot_first_derivative()
    # analyzer.plot_second_derivative()
    analyzer.plot_results_all()
    plt.show()
