import paho.mqtt.client as mqtt
import json
import time
import numpy as np

BROKER = "broker.emqx.io"
TOPIC = "smartwatch/healthdata"

client = mqtt.Client()
client.connect(BROKER, 1883, 60)
print("Connection: Connected to MQTT broker")

def generate_data():
    return {
        'heart_rate': np.clip(np.random.normal(72, 20), 60,120),
        'spo2': np.clip(np.random.normal(97, 3), 90,100),
        'temperature_f': np.clip(np.random.normal(98, 3), 97,102),
        'stress': np.clip(np.random.normal(3, 2), 1,10)
    }

print("Connection: Publishing health data to MQTT----")
while True:
    data = generate_data()
    client.publish(TOPIC, json.dumps(data))
    print("Sent:", data)
    time.sleep(3)