import PySpin
import os
import time


def configure_camera(cam):
    try:
        # Set acquisition mode to continuous
        cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

        # Enable trigger mode
        cam.TriggerMode.SetValue(PySpin.TriggerMode_On)

        # Set trigger source to Line 0 (GPIO input from Arduino)
        cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)

        # Set trigger activation to falling edge (matches Arduino pulse)
        cam.TriggerActivation.SetValue(PySpin.TriggerActivation_FallingEdge)

        # Optional: Set exposure time (adjust for lighting/frame rate)
        # 5 ms (shorter than 8.33 ms trigger interval)
        # cam.ExposureTime.SetValue(5000)

        # Optional: Reduce resolution to increase frame rate (uncomment if needed)
        cam.Width.SetValue(640)
        cam.Height.SetValue(480)

        print("Camera configured for external trigger on Line 0.")
    except PySpin.SpinnakerException as e:
        print(f"Error configuring camera: {e}")
        raise


def save_frames(cam, output_dir="captured_frames"):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 0
    print("Starting frame capture. Press Ctrl+C to stop...")

    try:
        while True:
            # Get the next image (timeout in 1000 ms)
            image = cam.GetNextImage(1000)
            if image.IsIncomplete():
                print(f"Frame {frame_count}: Image incomplete, skipping.")
            else:
                # Generate a filename with frame count and timestamp
                timestamp = int(time.time() * 1000)  # Milliseconds
                filename = os.path.join(
                    output_dir, f"frame_{frame_count:04d}_{timestamp}.jpg")

                # Save the image
                image.Save(filename, PySpin.JPEG)
                print(f"Saved {filename}")

                frame_count += 1

            # Release the image
            image.Release()

    except PySpin.SpinnakerException as e:
        print(f"Error capturing frame: {e}")
    except KeyboardInterrupt:
        print(f"Captured {frame_count} frames.")


def main():
    # Initialize the system
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        print("No camera found!")
        system.ReleaseInstance()
        return

    cam = cam_list[0]
    cam.Init()

    # Configure the camera
    configure_camera(cam)

    # Start acquisition
    cam.BeginAcquisition()
    print("Acquisition started.")

    # Save frames to files
    save_frames(cam)

    # Stop acquisition
    cam.EndAcquisition()
    cam.DeInit()

    # Release system
    cam_list.Clear()
    system.ReleaseInstance()
    print("Acquisition stopped.")


if __name__ == "__main__":
    main()
