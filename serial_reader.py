import serial
import time

def read_sensor_data(port="COM3", baudrate=115200):
    try:
        ser = serial.Serial(port, baudrate, timeout=2)
        time.sleep(2)  # allow ESP32 to reset

        for _ in range(10):  # try multiple times
            line = ser.readline().decode("utf-8").strip()

            if not line:
                continue

            # ✅ only accept correct format
            if "PH:" not in line:
                continue

            print("VALID RAW:", line)

            data = {}
            parts = line.split(",")

            for part in parts:
                if ":" not in part:
                    continue

                key, value = part.split(":", 1)

                try:
                    data[key.strip()] = float(value.strip())
                except:
                    continue

            return {
                "pH": data.get("PH", 0),
                "Turbidity": data.get("TURB", 0),
                "TDS": data.get("TDS", 0),
                "Temperature": data.get("TEMP", 0),
            }

        return None  # if no valid data found

    except Exception as e:
        print("Serial Error:", e)
        return None