Readme.md

---

# Time Series Forecasting using Sequence Models

## Overview

This project implements a time-series forecasting pipeline from scratch to predict electricity production values using different models.

The focus is on understanding:

* how sequence data is handled
* how models use past information
* where models fail and why

---

## Personalized Parameters

Based on roll number 102303769, the following parameters were computed:

```
window_size = 9
prediction_horizon = 1
hidden_size = 14
```

These values are used consistently across all models to ensure fair comparison.

---

## Dataset

The primary dataset used is the Electricity Production dataset.

* Type: Univariate time-series
* Description: Monthly electricity production values

### Preprocessing

* Extracted the numerical column from dataset
* Applied normalization:

```
x' = (x - mean) / std
```

---

## Sequence Creation (Windowing)

Time-series data is converted into supervised learning format using a sliding window approach.

* Input: past window_size values
* Output: next prediction_horizon value

Example:

```
Input  → [x1, x2, ..., x9]
Target → [x10]
```

---

## Models Implemented

### 1. MLP (Baseline)

* Fully connected neural network
* No sequence awareness
* Treats input as independent features

---

### 2. Custom GRU (From Scratch)

* Implemented without using nn.GRU
* Uses update and reset gates
* Maintains temporal memory

---

### 3. LSTM (Prebuilt)

* Uses input, forget, and output gates
* Handles long-term dependencies

---

### 4. Transformer (Prebuilt)

* Uses self-attention mechanism
* Processes sequences in parallel
* No explicit positional encoding

---

## Training Details

* Loss Function: Mean Squared Error (MSE)
* Optimizer: Adam
* Epochs: 50
* Train/Test Split: 80% / 20% (chronological split)

---

## Evaluation Metrics

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

---

## Results and Observations

### Model Behavior

MLP:

* Fails to capture temporal dependencies
* Produces overly smooth predictions

GRU:

* Captures general trends
* Struggles with sharp peaks

LSTM:

* Slight improvement over GRU
* Better handling of longer dependencies

Transformer:

* Performance depends on dataset size
* May underperform on smaller datasets

---

## Failure Analysis

The GRU model produces smooth predictions close to the mean and fails to capture sharp peaks.

Reasons:

* Limited hidden size
* Short window size
* MSE loss encourages averaging
* Model underfitting

---

## Ablation Study

Effect of changing window size:

* Window size 4: insufficient context, higher error
* Window size 9: balanced performance
* Window size 18: increased complexity, unstable learning

Conclusion:
There is a trade-off between context and learnability.

---

## Model Comparison

| Model       | Sequence Awareness | Performance    |
| ----------- | ------------------ | -------------- |
| MLP         | No                 | Poor           |
| GRU         | Yes                | Good           |
| LSTM        | Yes                | Better         |
| Transformer | Yes                | Data dependent |

---

## Key Learnings

* Sequence models outperform non-sequential models on time-series data
* Window size significantly affects performance
* Model complexity should match data complexity
* Simpler models may generalize better on small datasets

---

## Limitations

* No hyperparameter tuning
* No positional encoding in Transformer
* Limited dataset size
* Single-step forecasting only

---

## How to Run

Install dependencies:

```
pip install torch numpy pandas matplotlib scikit-learn
```

Run the notebook:

```
Sequence_Modeling.ipynb
```

---

## Author

Khushi Mehta - 102303769
Computer Engineering Student

---

