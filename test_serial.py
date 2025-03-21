import PySpin
import numpy as np
import serial
import threading
import queue
import time

SERIAL_PORT = 'COM9'  # Adjust as needed
BAUD_RATE = 9600
video_queue = queue.Queue()
pulse_queue = queue.Queue()
stop_event = threading.Event()


def record_video():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        print("No cameras found")
        return
    cam = cam_list.GetByIndex(0)
    cam.Init()
    # Get the camera's actual sensor dimensions
    nodemap = cam.GetNodeMap()
    sensor_width = PySpin.CIntegerPtr(nodemap.GetNode('SensorWidth'))
    sensor_height = PySpin.CIntegerPtr(nodemap.GetNode('SensorHeight'))

    if not PySpin.IsReadable(sensor_width) or not PySpin.IsReadable(sensor_height):
        print(
            'Unable to read sensor dimensions. Using Width/Height max values instead.')
        max_width = cam.Width.GetMax()
        max_height = cam.Height.GetMax()
    else:
        max_width = sensor_width.GetValue()
        max_height = sensor_height.GetValue()
        print('Sensor dimensions: %dx%d' % (max_width, max_height))
# Set width to width configured
    if cam.Width.GetAccessMode() == PySpin.RW and cam.Width.GetInc() != 0:
        # Calculate center offset for X
        center_offset_x = (max_width - 296) // 2
        # Set width first
        cam.Width.SetValue(296)
        print('Width set to %i...' % cam.Width.GetValue())
        # Then set X offset to center
        if cam.OffsetX.GetAccessMode() == PySpin.RW:
            cam.OffsetX.SetValue(center_offset_x)
            print('Offset X set to %d...' % cam.OffsetX.GetValue())
    else:
        print('Width not available...')
        result = False

    # Set height to height configured
    if cam.Height.GetAccessMode() == PySpin.RW and cam.Height.GetInc() != 0:
        # Calculate center offset for Y
        center_offset_y = (max_height - 300) // 2
        # Set height first
        cam.Height.SetValue(300)
        print('Height set to %i...' % cam.Height.GetValue())
        # Then set Y offset to center
        if cam.OffsetY.GetAccessMode() == PySpin.RW:
            cam.OffsetY.SetValue(center_offset_y)
            print('Offset Y set to %d...' % cam.OffsetY.GetValue())
    else:
        print('Height not available...')
        result = False
    # Configure frame rate
    # Configure Frame Rate using QuickSpin API
    if cam.AcquisitionFrameRateEnable is not None:
        if cam.AcquisitionFrameRateEnable.GetAccessMode() == PySpin.RW:
            # Enable frame rate control
            cam.AcquisitionFrameRateEnable.SetValue(True)

            # Set frame rate directly if available
            if hasattr(cam, 'AcquisitionFrameRate') and cam.AcquisitionFrameRate.GetAccessMode() == PySpin.RW:
                # Get limits
                min_frame_rate = cam.AcquisitionFrameRate.GetMin()
                max_frame_rate = cam.AcquisitionFrameRate.GetMax()
                print(
                    f'Frame rate range: {min_frame_rate} to {max_frame_rate} fps')

                # Set frame rate (make sure it's within range)
                frame_rate = min(max_frame_rate, float(120))
                cam.AcquisitionFrameRate.SetValue(frame_rate)
                print(f'Frame rate set to: {frame_rate} fps')

                # Verify the frame rate was set
                actual_frame_rate = cam.AcquisitionFrameRate.GetValue()
                print(f'Actual frame rate: {actual_frame_rate} fps')
    else:
        print('Frame rate control not available for this camera model')
    print(f"Frame rate set to: {cam.AcquisitionFrameRate.GetValue()} FPS")

    cam.LineSelector.SetValue(PySpin.LineSelector_Line2)
    cam.LineMode.SetValue(PySpin.LineMode_Input)

    cam.BeginAcquisition()
    # Configure GPIO (Line 1 as output, strobe-like pulse)
    cam.LineMode.SetValue(PySpin.LineMode_Output)
    cam.LineSource.SetValue(PySpin.LineSource_ExposureActive)  # Start pulses

    print("Video recording started...")

    while not stop_event.is_set():
        try:
            image = cam.GetNextImage(1000)
            if image.IsIncomplete():
                print("Image incomplete")
                continue
            timestamp = image.GetTimeStamp() / 1000
            video_queue.put((timestamp, image))
            print(f"Video frame at {timestamp} µs")
            image.Release()
        except PySpin.SpinnakerException as e:
            print(f"Video error: {e}")

    cam.EndAcquisition()
    cam.LineMode.SetValue(PySpin.LineMode_Input)  # Stop pulses
    cam.DeInit()
    del cam
    cam_list.Clear()
    system.ReleaseInstance()


def collect_pulse_data(ser):
    print("Waiting for Arduino start signal...")
    # while True:
    #     line = ser.readline().decode('utf-8').strip()
    #     print(f"Serial read: {line}")
    #     if line == "START":
    #         break
    # print("Pulse data collection started...")
    start = False
    while not stop_event.is_set() and start == False:
        start = True
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            print(f"Pulse line: {line}")
            if line.startswith("PULSE,"):
                parts = line.split(",")
                if len(parts) == 5:
                    timestamp = int(parts[1])
                    value = int(parts[2])
                    status = int(parts[3])
                    checksum = int(parts[4])
                    pulse_queue.put((timestamp, value, status, checksum))

        start = False
    ser.close()


# Main execution
print("Starting data collection...")
ser = serial.Serial(port=SERIAL_PORT, baudrate=BAUD_RATE, parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS, timeout=1)


print("Starting data collection...")
video_thread = threading.Thread(target=record_video)
pulse_thread = threading.Thread(target=collect_pulse_data, args=(ser,))
video_thread.start()
pulse_thread.start()

time.sleep(3)  # Run for 5 seconds
stop_event.set()
video_thread.join()
pulse_thread.join()

ser.close()

# Collect data
video_data = [video_queue.get() for _ in range(video_queue.qsize())]
pulse_data = [pulse_queue.get() for _ in range(pulse_queue.qsize())]

# Adjust video timestamps relative to time_start
video_data = [(ts - video_data[0][0], img)
              for ts, img in video_data]  # Relative to first frame
pulse_data = [(ts - pulse_data[0][0], val, stat, chk)
              for ts, val, stat, chk in pulse_data]  # Already relative

print("\nVerification:")
print(f"Video frames: {len(video_data)}")
print(f"Pulse samples: {len(pulse_data)}")
