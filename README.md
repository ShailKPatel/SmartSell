# 📱 SmartSell: Smartphone Sales Prediction with Neural Networks

SmartSell is a machine learning project designed to predict the number of smartphones sold in a given target country based on their specifications, market position, and launch details. This project uses a neural network model trained on engineered and raw features to forecast unit sales.

---

## 🧠 Core Objective

To train a neural network that accurately predicts `num_units_sold` using both raw and derived features from smartphone specifications and launch metadata.

---

## 📊 Dataset Features

| Feature | Use in Model | Type | Notes |
|--------|--------------|------|-------|
| `model_name` | ✅ | Raw | Encoded via text embeddings or clustering |
| `brand_name` | ✅ | Raw | Strong categorical indicator |
| `category` | ✅ | Raw | Budget / Midrange / Flagship |
| `price_usd` | ✅ | Raw | Key numerical feature |
| `launch_month` | ✅ | Derived | Captures seasonal trends |
| `launch_day` | ✅ | Derived | Captures daily trends |
| `target_country` | ✅ | Raw | Regional trend modeling |
| `dust_resistance_level` | ✅ | Derived | From IP rating (0–6 scale) |
| `water_resistance_level` | ✅ | Derived | From IP rating (0–9 scale) |
| `ram_gb` | ✅ | Raw | Physical RAM in GB |
| `extendable_ram_gb` | ✅ | Raw | Virtual RAM |
| `storage_gb` | ✅ | Raw | Internal storage size |
| `battery_mah` | ✅ | Raw | Battery size |
| `camera_mp` | ✅ | Raw | Rear camera resolution |
| `processor_name` | ✅ | Raw | Categorical or clustered |
| `screen_size_inches` | ✅ | Raw | Display size |
| `os_version` | ✅ | Raw | Version or grouped category |
| `num_units_sold` | 🔚 Target | Raw | Regression target |

---

## 🏗️ Model Architecture

- Input: Preprocessed numerical and encoded categorical features
- Layers: Dense layers with ReLU activation
- Regularization: Dropout & BatchNorm
- Output: Single neuron for regression (units sold)
- Loss: MSE or MAE

---

## 🧰 Tools & Technologies

- Python
- Pandas / NumPy
- Scikit-learn
- TensorFlow / Keras or PyTorch
- Matplotlib / Seaborn for EDA

---



## 📄 License

MIT License

---



