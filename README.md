

# 🚀 DemandSense AI

**AI-Powered Sales Forecasting & Analytics Platform**

> Turn raw sales data into actionable insights for inventory planning, demand forecasting, and promotional strategy.

---

## 📌 Overview

**DemandSense AI** is an end-to-end machine learning system that predicts future sales demand and visualizes business insights through an interactive dashboard. It helps businesses reduce guesswork and make **data-driven decisions** using time-series modeling and analytics.

---

## 🎯 Problem

Businesses often rely on manual or static forecasting methods, which leads to:

* ❌ Overstocking or stockouts
* ❌ Inefficient promotional planning
* ❌ Poor visibility into demand trends

---

## ✅ Solution

DemandSense AI provides:

* 📈 Automated demand forecasting
* 📊 Interactive KPI dashboard
* 🔍 Insight generation (trends, volatility, promotions)
* ⚡ Real-time data analysis

---

## 🧠 Key Features

* **📊 Forecasting Engine**
  Predicts sales using XGBoost with advanced time-series feature engineering.

* **📉 Business Metrics**
  Uses MAE, RMSE, R², SMAPE, and WAPE for realistic evaluation.

* **📊 Dashboard UI (Streamlit)**
  Clean interface with:

  * KPIs (Total Sales, Avg Sales, Peak Sales)
  * Forecast graphs
  * Filters (store, product, date)

* **🔧 Feature Engineering**

  * Lag features: 1, 7, 14, 28
  * Rolling statistics
  * Trend indicators
  * Promotion impact

* **📂 Flexible Input**

  * Upload CSV
  * Upload Excel
  * Google Sheets support

---

## 🏗️ Tech Stack

| Layer               | Tools                 |
| ------------------- | --------------------- |
| **Language**        | Python                |
| **ML Model**        | XGBoost, Scikit-learn |
| **Data Processing** | Pandas, NumPy         |
| **Visualization**   | Plotly                |
| **Frontend**        | Streamlit             |

---

## ⚙️ Project Structure

```bash
sales-forecast/
│
├── app/
│   └── app.py                # Streamlit dashboard
│
├── src/
│   ├── train_model.py        # Model training
│   └── feature_engineering.py
│
├── model/
│   ├── model.pkl
│   └── model_meta.json
│
├── data/
│   └── processed/demo_sales.csv
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Clone the repo

```bash
git clone https://github.com/your-username/demandsense-ai.git
cd demandsense-ai
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run app/app.py
```

---

## 📊 How It Works

1. Upload your dataset (CSV / Excel / Sheets)
2. Data is processed and transformed using feature engineering
3. Pre-trained model predicts future sales
4. Results are displayed in an interactive dashboard

---

## 📈 Model Performance

| Metric | Value   |
| ------ | ------- |
| MAE    | ~1.4    |
| RMSE   | ~2.5    |
| R²     | ~0.45   |
| SMAPE  | ~85–90% |
| WAPE   | ~45–50% |

> Note: Performance depends on dataset quality and size.

---

## 💡 Example Insights

* 📉 Detect declining or increasing sales trends
* 📦 Identify high-demand periods
* 🎯 Evaluate promotion impact
* ⚠️ Analyze sales volatility

---



