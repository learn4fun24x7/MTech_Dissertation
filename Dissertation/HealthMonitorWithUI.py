import paho.mqtt.client as mqtt
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import json
from collections import deque
import threading
import gradio as gr
import plotly.graph_objects as go
from twilio.rest import Client
import os
from dotenv import load_dotenv

# Twilio Credentials
load_dotenv()
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
client = Client(TWILIO_SID, TWILIO_TOKEN)

# MQTT Settings
BROKER = "broker.emqx.io"
TOPIC = "smartwatch/healthdata"

# Global State
latest_data = {}
anomaly_status = "Waiting for data..."
history = deque(maxlen=100)

# Generate baseline data to train the model
def simulate_data():
    import numpy as np
    return {
        'heart_rate': np.clip(np.random.normal(72, 10), 60,100),     # normal HR 60-100 bpm
        'spo2': np.clip(np.random.normal(97, 2), 95,100),            # normal SpO2 ~98%
        'temperature_f': np.clip(np.random.normal(98, 1), 97,99),    # Fahrenheit
        'stress': np.clip(np.random.normal(3, 2), 1,6)               # scale 1–10
    }

# Generate baseline dataset to train the model
def generate_baseline_data(n=50000):
    return pd.DataFrame([simulate_data() for _ in range(n)])

# Model training using baseline dataset
baseline_df = generate_baseline_data()
model = IsolationForest(contamination=0.01)
model.fit(baseline_df)

# Risk detection & alerting
def detect_anomalies(data):
    df = pd.DataFrame([data])
    pred = model.predict(df)
    scores = -model.decision_function(df)

    if pred[0] == -1:
        print("Warning: Health Risk Detected! Anomaly in readings:", data)

        client.messages.create(
                        from_='whatsapp:+14155238886',
                        body=f"Health Risk Detected! Reading: {data} and Score: {scores}. Please discuss with Sanjeevani Virtual Care Assistant.",
                        to='whatsapp:+919163040468')

        print("Warning: Alert sent to user.\n")
    else:
        print("INFO: No Health Risk Detected. Readings are normal:", data)
        print("INFO: Continuing to monitor...\n")

    return "⚠️ Health Risk Detected" if pred[0] == -1 else "✅ Normal"

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    print("Connection: Starting Sanjeevani monitoring service")
    print("Connection: Connected to MQTT broker")
    client.subscribe(TOPIC)
    print("Connection: Subscribing to health data stream")

def on_message(client, userdata, msg):
    global latest_data, anomaly_status
    try:
        data = json.loads(msg.payload.decode())
        latest_data = data
        anomaly_status = detect_anomalies(data)
        history.append(data)
    except Exception as e:
        print("Error processing message:", e)

def start_mqtt():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, 1883, 60)
    client.loop_forever()

mqtt_thread = threading.Thread(target=start_mqtt)
mqtt_thread.daemon = True
mqtt_thread.start()

# Web Interface - Gradio Dashboard
def refresh_dashboard():
    if not latest_data:
        return "Waiting for data...", "Waiting...", go.Figure()

    display = ""
    for key in latest_data:
        value = round(latest_data[key],2)
        display += key + " : " + str(value) + "\n"

    df = pd.DataFrame(history)
    fig = go.Figure()
    for metric, color in zip(["heart_rate", "spo2", "body_temp", 'stress_level'], ["red", "green", "blue", "black"]):
        if metric in df:
            fig.add_trace(go.Scatter(
                y=df[metric],
                mode="lines+markers",
                name=metric,
                line=dict(color=color)
            ))

    fig.update_layout(title="📊 Real-time Health Metrics", height=400, margin=dict(t=30), 
                      plot_bgcolor="white", font=dict(family="Source Sans Pro", size=10.5))
    return display, anomaly_status, fig

# Live Gradio App
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Sanjeevani - Health Monitoring Dashboard")

    with gr.Row():
        data_text = gr.Textbox(label="📋 Latest Health Data", lines=6)
        status_text = gr.Textbox(label="🚨 Risk Status")

    chart = gr.Plot(label="📈 Vitals (Live Graph)")

    # Auto-refresh every 3 seconds
    timer = gr.Timer(value=3.0, active=True)
    timer.tick(refresh_dashboard, [], [data_text, status_text, chart])

demo.launch()