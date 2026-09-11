"""
Generates a small synthetic sample CSV shaped like the real Edge-IIoTset
dataset (same kind of column names), purely so we can test that
preprocess.py works correctly before downloading the real, much larger
dataset.

This is NOT real network traffic data — it's random values for testing
pipeline mechanics only.

Run once:
    python generate_sample_data.py
"""
import random

import pandas as pd

random.seed(42)  # fixed seed -> same "random" data every time we run this

N_ROWS = 300

protocols = ["TCP", "UDP", "ICMP"]
http_methods = ["GET", "POST", None]
mqtt_topics = ["sensor/temperature", "sensor/humidity", None]
attack_types = ["Normal", "DDoS_UDP", "Port_Scanning", "SQL_injection", "XSS", "Password", "MITM"]

rows = []
for i in range(N_ROWS):
    attack_type = random.choice(attack_types)
    # A few rows with a missing/blank label, on purpose -- real datasets
    # have this, and it's what our "drop_missing_label_rows" step handles.
    if random.random() < 0.03:
        attack_type = None

    rows.append({
        "frame.time": f"2023-01-01 00:{i // 60:02d}:{i % 60:02d}",
        "ip.src_host": f"192.168.1.{random.randint(1, 254)}",
        "ip.dst_host": f"10.0.0.{random.randint(1, 254)}",
        "tcp.srcport": random.randint(1024, 65535) if random.random() > 0.05 else None,
        "tcp.dstport": random.choice([80, 443, 22, 502]),
        "protocol": random.choice(protocols),
        "http.request.method": random.choice(http_methods),
        "mqtt.topic": random.choice(mqtt_topics),
        "icmp.checksum": None,          # entirely empty column, on purpose -- should get dropped
        "frame.len": round(random.uniform(60, 1500), 1),
        "constant_col": "same_value",   # same value every row, on purpose -- should get dropped
        "random_session_id": f"sess_{i}_{random.randint(100000000, 999999999)}",  # unique per row -- should get dropped
        "Attack_type": attack_type,
    })

df = pd.DataFrame(rows)
df.to_csv("data/raw/sample_synthetic_flows.csv", index=False)
print(f"Wrote {len(df)} rows to data/raw/sample_synthetic_flows.csv")
print(df["Attack_type"].value_counts(dropna=False))