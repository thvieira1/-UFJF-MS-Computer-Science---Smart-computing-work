# Fuzzy Anomaly Detection in Multivariate Time Series

Final project for the *Fuzzy Systems* module (Sistemas Nebulosos) – Mestrado em Ciência da Computação / UFJF.

This repository implements a **Fuzzy Inference System (FIS)** for **anomaly detection in multivariate time series**, inspired by papers on fuzzy logic applied to anomaly detection and condition monitoring.

## Problem / Context

We consider a multivariate time series from an industrial equipment (e.g., motor, pump), where sensors provide:
- temperature
- vibration
- electric current
- pressure (etc.)

From a sliding time window, we extract indicators that summarize the recent behavior of the system. The goal of the FIS is to aggregate these indicators into a **linguistic anomaly level**.

## Fuzzy Inference System

**Inputs (linguistic variables):**

All inputs are normalized in [0, 1]:

- `forecast_error` (EP): normalized prediction error  
- `variance_change` (MV): normalized change in variance  
- `correlation_change` (MC): normalized change in correlation structure between variables  

Each input has three fuzzy sets: `low`, `medium`, `high`.

**Output (linguistic variable):**

- `anomaly_level` ∈ [0, 10]

With four fuzzy sets:

- `normal`
- `slightly_anomalous`
- `moderately_anomalous`
- `strongly_anomalous`

**Inference:**

- Mamdani-type rules (IF–THEN)
- Several rules combining input terms (for example: high forecast error + high variance change → strongly anomalous)
- Defuzzification method: **centroid (center of gravity)**

## Implementation

Main files:

- `fuzzy_anomaly_system.py` – defines the `FuzzyAnomalyDetector` class and the fuzzy rule base
- `main.py` – generates synthetic multivariate time series, computes indicators, and evaluates the FIS

## How to run

```bash
pip install -r requirements.txt
python main.py
