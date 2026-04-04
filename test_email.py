import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from email_alert import send_email_alert

print("=" * 70)
print("EMAIL ALERT TEST - UNSAFE WATER WITH STATION CODE & PARAMETERS")
print("=" * 70)
print()

# Simulate unsafe water parameters
unsafe_parameters = [
    {
        'parameter': 'Temperature',
        'value': 38.5,
        'threshold_type': 'above max',
        'threshold_value': 'Max: 35°C'
    },
    {
        'parameter': 'Dissolved Oxygen',
        'value': 3.2,
        'threshold_type': 'below min',
        'threshold_value': 'Min: 5 ppm'
    },
    {
        'parameter': 'pH Level',
        'value': 9.1,
        'threshold_type': 'above max',
        'threshold_value': 'Max: 8.5'
    }
]

station_code = "WQ-STATION-01"

print(f"Testing email alert for station: {station_code}")
print(f"Unsafe parameters: {len(unsafe_parameters)}")
print()

# Send the alert
success = send_email_alert(unsafe_parameters, station_code)

print()
print("=" * 70)
if success:
    print("✅ SUCCESS! Unsafe water alert email sent successfully.")
    print()
    print("Email Details:")
    print(f"  - Station Code: {station_code}")
    print(f"  - Status: UNSAFE")
    print(f"  - Parameters Sent: {len(unsafe_parameters)}")
    for param in unsafe_parameters:
        print(f"    • {param['parameter']}: {param['value']:.2f} ({param['threshold_type']})")
else:
    print("❌ FAILED! Email could not be sent.")
print("=" * 70)