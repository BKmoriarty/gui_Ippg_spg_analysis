import numpy as np
import matplotlib.pyplot as plt


class HarmonicAnalyzer:
    def __init__(self, signal, fs=1000, f_range_min=0.5, f_range_max=3.0, offset=0.2, verbose=True):
        """
        Initialize the HarmonicAnalyzer with a signal and sampling parameters.

        Args:
            signal (np.ndarray): Input signal array.
            fs (float): Sampling frequency in Hz (default: 1000).
            f_range_min (float): Min frequency for fundamental (default: 0.5 Hz).
            f_range_max (float): Max frequency for fundamental (default: 3.0 Hz).
            offset (float): Frequency offset for harmonic ranges (default: 0.2 Hz).
            verbose (bool): If True, print analysis results; if False, suppress printing (default: True).
        """
        self.signal = signal
        self.fs = fs
        self.f_range_min = f_range_min
        self.f_range_max = f_range_max
        self.offset = offset
        self.verbose = verbose

        # Process the signal
        self._setup_signal()
        self._compute_fft()
        self._find_fundamental()
        self._define_harmonics()
        self._analyze_harmonics()

        # Expose harmonic frequencies and magnitudes as attributes
        self.fundamental_freq = self.peaks['Fundamental'][0]
        self.fundamental_mag = self.peaks['Fundamental'][1]
        self.second_freq = self.peaks['Second'][0]
        self.second_mag = self.peaks['Second'][1]
        self.third_freq = self.peaks['Third'][0]
        self.third_mag = self.peaks['Third'][1]
        self.fourth_freq = self.peaks['Fourth'][0]
        self.fourth_mag = self.peaks['Fourth'][1]

    def _setup_signal(self):
        """Set up the time array based on the signal."""
        self.N = len(self.signal)
        self.t = np.linspace(0, self.N / self.fs, self.N, endpoint=False)

    def _compute_fft(self):
        """Compute the FFT and extract positive frequencies."""
        self.fft = np.fft.fft(self.signal)
        self.frequencies = np.fft.fftfreq(len(self.fft), 1 / self.fs)
        self.magnitude = np.abs(self.fft)

        # Limit to positive frequencies
        self.positive_mask = self.frequencies > 0
        self.frequencies = self.frequencies[self.positive_mask]
        self.magnitude = self.magnitude[self.positive_mask]

    def _find_fundamental(self):
        """Detect the fundamental frequency in the specified range."""
        fund_mask = (self.frequencies >= self.f_range_min) & (
            self.frequencies <= self.f_range_max)
        fund_freqs = self.frequencies[fund_mask]
        fund_mags = self.magnitude[fund_mask]

        if len(fund_mags) > 0:
            self.f = fund_freqs[np.argmax(fund_mags)]
            if self.verbose:
                print(f"Detected fundamental frequency: {self.f:.2f} Hz")
        else:
            raise ValueError(
                f"No fundamental frequency found in range {self.f_range_min}–{self.f_range_max} Hz.")

    def _define_harmonics(self):
        """Define the harmonic frequencies."""
        self.harmonics = {
            'Fundamental': self.f,
            'Second': 2 * self.f,
            'Third': 3 * self.f,
            'Fourth': 4 * self.f
        }

    def _analyze_harmonics(self):
        """Analyze each harmonic and find peak frequencies and magnitudes using the offset."""
        self.peaks = {}
        for name, harmonic_freq in self.harmonics.items():
            f_min = harmonic_freq - self.offset
            f_max = harmonic_freq + self.offset
            range_mask = (self.frequencies >= f_min) & (
                self.frequencies <= f_max)
            freqs_in_range = self.frequencies[range_mask]
            mags_in_range = self.magnitude[range_mask]

            if len(mags_in_range) > 0:
                peak_magnitude = np.max(mags_in_range)
                peak_frequency = freqs_in_range[np.argmax(mags_in_range)]
                self.peaks[name] = (peak_frequency, peak_magnitude)
                if self.verbose:
                    print(f"{name} harmonic peak in range {f_min:.2f} Hz to {f_max:.2f} Hz: "
                          f"Frequency = {peak_frequency:.2f} Hz, Magnitude = {peak_magnitude:.2f}")
            else:
                if self.verbose:
                    print(f"No peaks found in {name.lower()} harmonic range.")
                self.peaks[name] = (None, None)

    def plot_spectrum(self, fig=None, ax=None, color='r'):
        """
        Plot the spectrum with harmonic ranges and peaks.

        Args:
            fig (matplotlib.figure.Figure, optional): Existing figure to use. If None, creates a new one.
            ax (matplotlib.axes.Axes, optional): Existing axes to plot on. If None, creates a new one.

        Returns:
            fig, ax: The figure and axes objects used for plotting.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(self.frequencies, self.magnitude,
                label='Spectrum', color=color)

        # Colors and markers for harmonics
        colors = {'Fundamental': 'green', 'Second': 'yellow',
                  'Third': 'cyan', 'Fourth': 'magenta'}
        markers = {'Fundamental': 'go', 'Second': 'ro',
                   'Third': 'bo', 'Fourth': 'mo'}

        # Plot each harmonic
        for name, (peak_freq, peak_mag) in self.peaks.items():
            if peak_freq is not None:
                f_min = self.harmonics[name] - self.offset
                f_max = self.harmonics[name] + self.offset
                ax.axvspan(
                    f_min, f_max, color=colors[name], alpha=0.3, label=f'{name} range')
                ax.plot(peak_freq, peak_mag,
                        markers[name], label=f'{name} at {peak_freq:.2f} Hz')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title(
            f'Signal Spectrum with Harmonics (Fundamental = {self.f:.2f} Hz, Offset = {self.offset} Hz)')
        ax.set_xlim(0, self.harmonics['Fourth'] + 1)
        ax.legend()
        ax.grid()

        return fig, ax


# Example usage
if __name__ == "__main__":
    # Load the signal outside the class
    file_path = "storage/2025-02-20 12_51_25 top 7000 120 (320, 320)/ippg/filtered_ppg.npy"
    signal = np.load(file_path)

    # Example 1: With printing (verbose=True, default)
    print("With printing:")
    analyzer_with_print = HarmonicAnalyzer(
        signal, fs=1000, offset=0.1, verbose=True)
    print("\nHarmonic Results:")
    print(
        f"Fundamental: Freq = {analyzer_with_print.fundamental_freq:.2f} Hz, Mag = {analyzer_with_print.fundamental_mag:.2f}")
    print(
        f"Second: Freq = {analyzer_with_print.second_freq:.2f} Hz, Mag = {analyzer_with_print.second_mag:.2f}")
    print(
        f"Third: Freq = {analyzer_with_print.third_freq:.2f} Hz, Mag = {analyzer_with_print.third_mag:.2f}")
    print(
        f"Fourth: Freq = {analyzer_with_print.fourth_freq:.2f} Hz, Mag = {analyzer_with_print.fourth_mag:.2f}")
    analyzer_with_print.plot_spectrum()
    plt.show()
