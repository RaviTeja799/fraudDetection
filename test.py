"""
measure_latency.py
===================
Sends repeated requests to the running FastAPI /predict endpoint
and reports latency statistics (mean, median, p95, p99).

Usage:
    1. Make sure the FastAPI server is running (uvicorn main:app --port 8000)
    2. python measure_latency.py
"""

import time
import statistics
import requests

URL = "http://127.0.0.1:8000/predict"
N_REQUESTS = 200

# A couple of representative payloads to vary across requests
payloads = [
    {
        "kyc_verified": "No",
        "account_age_days": 120,
        "transaction_amount": 256369,
        "channel": "Mobile",
        "timestamp": "2025-08-12T02:10:24",
    },
    {
        "kyc_verified": "Yes",
        "account_age_days": 1800,
        "transaction_amount": 4200,
        "channel": "ATM",
        "timestamp": "2025-08-12T14:10:24",
    },
]

latencies_ms = []

# Warm-up requests (exclude from stats — first few calls are often slower
# due to lazy imports / connection setup, not representative of steady-state)
for _ in range(5):
    requests.post(URL, json=payloads[0])

print(f"Running {N_REQUESTS} requests...")
for i in range(N_REQUESTS):
    payload = payloads[i % len(payloads)]
    start = time.perf_counter()
    response = requests.post(URL, json=payload)
    end = time.perf_counter()

    if response.status_code != 200:
        print("Request failed:", response.status_code, response.text)
        continue

    latencies_ms.append((end - start) * 1000)

latencies_ms.sort()
n = len(latencies_ms)

print(f"\nCompleted {n} successful requests")
print(f"Mean latency:   {statistics.mean(latencies_ms):.2f} ms")
print(f"Median latency: {statistics.median(latencies_ms):.2f} ms")
print(f"Min latency:    {min(latencies_ms):.2f} ms")
print(f"Max latency:    {max(latencies_ms):.2f} ms")
print(f"P95 latency:    {latencies_ms[int(n * 0.95)]:.2f} ms")
print(f"P99 latency:    {latencies_ms[int(n * 0.99)]:.2f} ms")