from twilio.rest import Client
import streamlit as st


import os

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")

def send_whatsapp_alert(message: str) -> bool:
    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        client.messages.create(
            body=message,
            from_=FROM_PHONE,
            to=TO_PHONE
        )
        st.write("✅ Message SID:", msg.sid)
        st.write("✅ Status:", msg.status)
        return True
    except Exception as e:
        print("Alert failed:", e)
        return False



