from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

# จำลองข้อมูลสัญญาณหัวใจ
t = np.load(
    "storage/2025-02-20 12_51_25 top 7000 120 (320, 320)/ippg/time_ppg.npy")
signal = np.load(
    "storage/2025-02-20 12_51_25 top 7000 120 (320, 320)/ippg/filtered_ppg.npy")

# คำนวณช่วงห่างระหว่างจุดข้อมูล
h = t[1] - t[0]

# คำนวณอนุพันธ์อันดับ 1, 2, 3 ด้วย np.gradient
first_derivative = np.gradient(signal, h)              # อนุพันธ์อันดับ 1
second_derivative = np.gradient(first_derivative, h)   # อนุพันธ์อันดับ 2
third_derivative = np.gradient(second_derivative, h)   # อนุพันธ์อันดับ 3


def Derivative(xlist, ylist):
    yprime = np.diff(ylist)/np.diff(xlist)
    xprime = []
    for i in range(len(yprime)):
        xtemp = (xlist[i+1]+xlist[i])/2
        xprime = np.append(xprime, xtemp)
    return xprime, yprime


xprime_1, yprime_1 = Derivative(t, signal)
xprime_2, yprime_2 = Derivative(xprime_1, yprime_1)
xprime_3, yprime_3 = Derivative(xprime_2, yprime_2)

# หาพีก

# หาพีกของอนุพันธ์อันดับ 2
peaks_second_derivative, _ = find_peaks(yprime_2, prominence=0.01)

# แสดงกราฟ


# พล็อตกราฟทั้งหมด
plt.figure(figsize=(4, 4))

limit = [0, 20]
# limit = [10.5, 11.67]

# สัญญาณดั้งเดิม
plt.subplot(2, 1, 1)
plt.plot(t, signal, label='Heart Signal (Simulated)', color='blue')

plt.title('Original Heart Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(limit)
plt.grid(True)


# อนุพันธ์อันดับ 1
# plt.subplot(2, 1, 2)
# # plt.plot(t, first_derivative, label="First Derivative (f')", color='red')
# plt.plot(xprime_1, yprime_1, label="First Derivative (f')", color='green')
# plt.title('First Derivative')
# plt.xlabel('Time (s)')
# plt.ylabel('Rate of Change')
# plt.xlim(limit)
# plt.grid(True)


# อนุพันธ์อันดับ 2
plt.subplot(2, 1, 2)
# plt.plot(t, second_derivative, label="Second Derivative (f'')", color='green')
plt.plot(xprime_2, yprime_2, label="Second Derivative (f'')", color='red')
plt.plot(xprime_2[peaks_second_derivative],
         yprime_2[peaks_second_derivative], 'x', color='red')
plt.title('Second Derivative')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration')
plt.xlim(limit)
plt.grid(True)


# อนุพันธ์อันดับ 3
# plt.subplot(2, 1, 2)
# # plt.plot(t, third_derivative, label="Third Derivative (f''')", color='purple')
# plt.plot(xprime_3, yprime_3, label="Third Derivative (f''')", color='green')
# plt.title('Third Derivative')
# plt.xlabel('Time (s)')
# plt.ylabel('Jerk')
# plt.xlim(limit)
# plt.grid(True)


# ปรับระยะห่างกราฟ
plt.tight_layout()

# แสดงกราฟ
plt.show()
