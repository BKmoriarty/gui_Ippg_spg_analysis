import serial
import time

# Configure serial port (adjust port name as needed)
# Replace with your Arduino's port (e.g., '/dev/ttyUSB0' on Linux/Mac)
port = 'COM9'
baud_rate = 9600
ser = serial.Serial(port, baud_rate, timeout=1)

# Function to parse a line of Arduino data


def parse_line(line):
    try:
        # Remove leading/trailing whitespace and split by comma
        parts = line.strip().split(", ")

        # Dictionary to store parsed data
        data = {}
        for part in parts:
            # Split "Key: Value" into key and value
            key, value = part.split(": ", 1)
            data[key] = value

        # Convert numeric values
        data["Sample"] = int(data["Sample"])
        data["FPS"] = float(data["FPS"])
        if "Value" in data:  # Success case
            data["Value"] = int(data["Value"])
            data["Status"] = int(data["Status"])
            data["Checksum"] = int(data["Checksum"])
        data["Time (ms)"] = int(data["Time (ms)"])

        return data
    except (ValueError, KeyError) as e:
        print(f"Error parsing line: {line} - {e}")
        return None


# Main loop to read and process serial data
print(f"Listening on {port} at {baud_rate} baud...")
try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if line:
                print(f"Raw: {line}")

                # Parse the line
                parsed_data = parse_line(line)
                if parsed_data:
                    # Handle stats lines separately
                    if "Success Count" in line:
                        success = int(parsed_data["Success Count"])
                        fail = int(parsed_data["Fail Count"])
                        print(f"Stats: Success = {success}, Fail = {fail}")

except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    ser.close()
    print("Serial port closed")
