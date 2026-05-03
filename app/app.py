import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import io
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import subprocess
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_engineering import engineer_features

# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DemandSense AI — Demand Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS (Enhanced for better readability)
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

:root {
    --bg:         #F0F4FF;
    --surface:    #E8EEFF;
    --card:       #FFFFFF;
    --border:     #C8D6F8;
    --accent:     #3B6FF0;
    --accent2:    #00C9A7;
    --accent3:    #F59E0B;
    --warn:       #F59E0B;
    --danger:     #EF4444;
    --text:       #0F172A;
    --text-light: #1E293B;
    --muted:      #475569;
    --mono:       'JetBrains Mono', monospace;
    --shadow-sm:  0 2px 8px rgba(59,111,240,0.08);
    --shadow-md:  0 6px 24px rgba(59,111,240,0.13);
    --shadow-lg:  0 12px 40px rgba(59,111,240,0.18);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #E8EEFF 0%, #EEF2FF 100%) !important;
    border-right: 1.5px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

.stApp {
    background: linear-gradient(135deg, #F0F4FF 0%, #EDF5FF 50%, #F0F7F4 100%);
    color: var(--text);
}

div[data-testid="metric-container"] {
    background: var(--card);
    border: 1.5px solid var(--border);
    border-radius: 16px;
    padding: 1.3rem 1.6rem;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s;
}
div[data-testid="metric-container"]:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}
div[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-size: 11px !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 700 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-size: 32px !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.8);
    border-radius: 14px;
    padding: 6px 8px;
    gap: 4px;
    border: 1.5px solid var(--border);
    backdrop-filter: blur(8px);
    box-shadow: var(--shadow-sm);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 10px;
    color: var(--muted) !important;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 24px;
    letter-spacing: 0.01em;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent) 0%, #5B8AF5 100%) !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(59,111,240,0.35);
}

[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, rgba(59,111,240,0.05) 0%, rgba(0,201,167,0.05) 100%);
    border: 2.5px dashed var(--accent);
    border-radius: 18px;
    padding: 2rem 1.8rem;
    transition: all 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent);
    box-shadow: 0 0 0 6px rgba(59,111,240,0.12);
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #5B8AF5 100%);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    font-size: 14px;
    padding: 0.75rem 2rem;
    transition: all 0.2s;
    box-shadow: 0 4px 14px rgba(59,111,240,0.35);
    letter-spacing: 0.01em;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(59,111,240,0.45);
}
.stButton > button:active {
    transform: translateY(0px);
}

.upload-card {
    background: rgba(255,255,255,0.92);
    border: 1.5px solid var(--border);
    border-radius: 24px;
    padding: 32px 40px 28px;
    box-shadow: var(--shadow-md);
    backdrop-filter: blur(12px);
}

.map-card {
    background: rgba(255,255,255,0.95);
    border: 1.5px solid var(--border);
    border-radius: 20px;
    padding: 26px 32px;
    box-shadow: var(--shadow-md);
    margin-bottom: 1.4rem;
}

.map-required-badge {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    color: #B91C1C;
    font-size: 10px;
    font-family: var(--mono);
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 99px;
    letter-spacing: 0.1em;
    margin-left: 8px;
    border: 1.2px solid rgba(239,68,68,0.35);
}
.map-optional-badge {
    display: inline-block;
    background: rgba(148,163,184,0.15);
    color: #64748B;
    font-size: 10px;
    font-family: var(--mono);
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 99px;
    letter-spacing: 0.1em;
    margin-left: 8px;
    border: 1.2px solid rgba(148,163,184,0.35);
}

.alert-warn {
    background: rgba(245,158,11,0.12);
    border: 1.5px solid rgba(245,158,11,0.4);
    border-left: 5px solid #F59E0B;
    border-radius: 12px;
    padding: 14px 20px;
    margin: 12px 0;
    font-size: 13.5px;
    color: #92400E;
    font-weight: 500;
    line-height: 1.6;
}
.alert-success {
    background: rgba(0,201,167,0.12);
    border: 1.5px solid rgba(0,201,167,0.35);
    border-left: 5px solid #00C9A7;
    border-radius: 12px;
    padding: 14px 20px;
    margin: 12px 0;
    font-size: 13.5px;
    color: #065F46;
    font-weight: 500;
    line-height: 1.6;
}
.alert-danger {
    background: rgba(239,68,68,0.12);
    border: 1.5px solid rgba(239,68,68,0.35);
    border-left: 5px solid #EF4444;
    border-radius: 12px;
    padding: 14px 20px;
    margin: 12px 0;
    font-size: 13.5px;
    color: #7F1D1D;
    font-weight: 500;
    line-height: 1.6;
}
.alert-info {
    background: rgba(59,111,240,0.1);
    border: 1.5px solid rgba(59,111,240,0.3);
    border-left: 5px solid var(--accent);
    border-radius: 12px;
    padding: 14px 20px;
    margin: 12px 0;
    font-size: 13.5px;
    color: #1E3A8A;
    font-weight: 500;
    line-height: 1.6;
}

.ai-card {
    background: linear-gradient(135deg, rgba(59,111,240,0.07) 0%, rgba(0,201,167,0.06) 100%);
    border: 1.5px solid rgba(59,111,240,0.25);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 12px 0;
    transition: all 0.2s;
    cursor: pointer;
}
.ai-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
    border-color: rgba(59,111,240,0.4);
}
.ai-card .ai-label {
    font-size: 11px;
    font-family: var(--mono);
    color: var(--accent);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 8px;
    font-weight: 700;
}
.ai-card .ai-text {
    font-size: 14.5px;
    color: var(--text-light);
    line-height: 1.7;
    font-weight: 500;
}

.sec-header {
    font-size: 11px;
    font-family: var(--mono);
    color: var(--accent);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 2px solid var(--border);
    padding-bottom: 12px;
    margin: 2rem 0 1.4rem;
    font-weight: 700;
}

.logo-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.6rem 0 2rem;
}
.logo-mark {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #3B6FF0 0%, #5B8AF5 100%);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; font-weight: 800; color: white;
    font-family: var(--mono);
    box-shadow: 0 6px 16px rgba(59,111,240,0.4);
}
.logo-text {
    font-size: 17px; font-weight: 800; color: var(--text);
    letter-spacing: -0.03em;
}
.logo-sub {
    font-size: 10px; color: var(--muted);
    font-family: var(--mono);
    letter-spacing: 0.12em;
    font-weight: 700;
}

.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: #FFFFFF !important;
    border-color: var(--border) !important;
    border-width: 1.5px !important;
    color: var(--text) !important;
    border-radius: 11px !important;
    font-weight: 500 !important;
    box-shadow: var(--shadow-sm) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 4px rgba(59,111,240,0.15) !important;
}

.stDataFrame {
    border-radius: 14px;
    overflow: hidden;
    border: 1.5px solid var(--border) !important;
    box-shadow: var(--shadow-sm);
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 100%) !important;
    border-radius: 99px;
}

.hero-eyebrow {
    font-family: var(--mono);
    font-size: 11.5px;
    color: var(--accent);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 20px;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    gap: 9px;
    background: rgba(59,111,240,0.1);
    border: 1.2px solid rgba(59,111,240,0.25);
    padding: 7px 18px;
    border-radius: 99px;
}

.insight-icon {
    font-size: 28px;
    margin-right: 8px;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

.pulse {
    animation: pulse 1.5s infinite;
}
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ═════════════════════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,251,255,0.9)",
    font=dict(family="Sora", color="#0F172A", size=12),
    xaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", tickcolor="#94A3B8"),
    yaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", tickcolor="#94A3B8"),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#C8D6F8",
        borderwidth=1.5,
        font=dict(size=11, family="Sora", color="#0F172A")
    ),
    margin=dict(l=12, r=12, t=45, b=12),
    colorway=["#3B6FF0", "#00C9A7", "#F59E0B", "#EF4444", "#8B5CF6", "#10B981"]
)

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS & CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
META_PATH = os.path.join(BASE_DIR, "model", "model_meta.json")
DEMO_PATH = os.path.join(BASE_DIR, "data", "processed", "demo_sales.csv")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "src", "train_model.py")

REQUIRED_SCHEMA = {
    "date": {
        "label": "Date",
        "required": True,
        "description": "Transaction or order date",
        "synonyms": ["date", "order_date", "order date", "transaction_date", "transaction date",
                     "sale_date", "sale date", "day", "datetime", "timestamp", "invoice_date",
                     "invoice date", "billing_date", "dt", "trans_date"]
    },
    "store_nbr": {
        "label": "Store / Outlet",
        "required": True,
        "description": "Store, outlet, or branch identifier",
        "synonyms": ["store_nbr", "store", "store_id", "store id", "store_no", "store number",
                     "outlet", "outlet_id", "branch", "branch_id", "shop", "shop_id",
                     "location", "location_id", "store_code", "storeid"]
    },
    "family": {
        "label": "Product Family / Category",
        "required": True,
        "description": "Product family, category, or SKU group",
        "synonyms": ["family", "category", "product_family", "product family", "product_category",
                     "product category", "sku_family", "department", "product_type", "product type",
                     "segment", "group", "class", "item_category", "prod_family"]
    },
    "sales": {
        "label": "Sales / Units Sold",
        "required": True,
        "description": "Sales amount or units sold",
        "synonyms": ["sales", "units_sold", "units sold", "quantity", "qty", "units",
                     "sold_qty", "sales_qty", "sale_qty", "volume", "amount", "revenue",
                     "total_sales", "total sales", "sold", "demand", "unit_sales"]
    },
    "onpromotion": {
        "label": "Promotion Flag / Count",
        "required": False,
        "description": "Number of items on promotion (0 if none). Optional — defaults to 0.",
        "synonyms": ["onpromotion", "on_promotion", "on promotion", "promo", "promotion",
                     "promo_flag", "promotion_flag", "is_promo", "discount", "offer",
                     "promo_count", "promotion_count", "on_promo"]
    },
}


def load_model():
    """Load trained model and metadata."""
    if not os.path.exists(MODEL_PATH):
        return None, {}
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
    return model, meta


def load_gsheet(url: str) -> pd.DataFrame:
    """Load data from public Google Sheet."""
    if "spreadsheets/d/" in url:
        gid = url.split("spreadsheets/d/")[1].split("/")[0]
        csv_url = f"https://docs.google.com/spreadsheets/d/{gid}/export?format=csv"
        resp = requests.get(csv_url, timeout=15)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    raise ValueError("Invalid Google Sheets URL")


def auto_detect_mapping(df_columns: list) -> dict:
    """Auto-detect column mappings using synonyms."""
    mapping = {}
    lowered = {c.lower().strip(): c for c in df_columns}
    used = set()
    
    for target_col, info in REQUIRED_SCHEMA.items():
        match = None
        
        # Exact match
        if target_col.lower() in lowered and lowered[target_col.lower()] not in used:
            match = lowered[target_col.lower()]
        else:
            # Synonym match
            for syn in info["synonyms"]:
                if syn.lower() in lowered and lowered[syn.lower()] not in used:
                    match = lowered[syn.lower()]
                    break
            
            # Substring match
            if match is None:
                for low, orig in lowered.items():
                    if orig in used:
                        continue
                    for syn in info["synonyms"]:
                        if syn.lower() in low or low in syn.lower():
                            match = orig
                            break
                    if match:
                        break
        
        mapping[target_col] = match
        if match:
            used.add(match)
    
    return mapping


def schema_already_matches(df_columns: list) -> bool:
    """Check if all required columns are already present."""
    cols_lower = [c.lower().strip() for c in df_columns]
    for target_col, info in REQUIRED_SCHEMA.items():
        if info["required"] and target_col.lower() not in cols_lower:
            return False
    return True


def apply_mapping(df_raw: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Apply column mapping and prepare data."""
    df_new = df_raw.copy()
    
    rename_dict = {}
    for target_col, source_col in mapping.items():
        if source_col is not None and source_col != target_col:
            rename_dict[source_col] = target_col
    
    df_new = df_new.rename(columns=rename_dict)
    
    # Handle optional columns
    if "onpromotion" not in df_new.columns:
        df_new["onpromotion"] = 0
    
    # Coerce types
    if "date" in df_new.columns:
        df_new["date"] = pd.to_datetime(df_new["date"], errors="coerce")
        df_new = df_new.dropna(subset=["date"])
    
    for col in ["sales", "onpromotion"]:
        if col in df_new.columns:
            df_new[col] = pd.to_numeric(df_new[col], errors="coerce").fillna(0)
    
    return df_new


def generate_ai_insights(df: pd.DataFrame) -> list[dict]:
    """Generate actionable business insights from data."""
    insights = []
    total = df["sales"].sum()
    avg = df["sales"].mean()
    std = df["sales"].std()
    
    # Peak store
    if len(df) > 0:
        top_store = df.groupby("store_nbr")["sales"].sum().idxmax()
        top_store_val = df.groupby("store_nbr")["sales"].sum().max()
        insights.append({
            "icon": "🏆",
            "title": "Top Performing Store",
            "text": f"<strong>Store {top_store}</strong> leads with {top_store_val:,.0f} units sold — "
                    f"<strong>{top_store_val/total*100:.1f}%</strong> of total volume. "
                    f"Increase inventory allocation here to capture additional demand."
        })
    
    # Promo impact
    promo_data = df[df["onpromotion"] > 0]
    no_promo_data = df[df["onpromotion"] == 0]
    if len(promo_data) > 0 and len(no_promo_data) > 0:
        promo_avg = promo_data["sales"].mean()
        no_promo_avg = no_promo_data["sales"].mean()
        if pd.notna(promo_avg) and pd.notna(no_promo_avg) and no_promo_avg > 0:
            lift = (promo_avg - no_promo_avg) / no_promo_avg * 100
            if lift > 5:
                insights.append({
                    "icon": "📣",
                    "title": "Promotion Impact Detected",
                    "text": f"Promotional days drive <strong>{lift:.1f}% higher</strong> sales on average. "
                            f"ROI positive — increase promo frequency for low-velocity SKUs by 15-20%."
                })
            elif lift < -5:
                insights.append({
                    "icon": "⚠️",
                    "title": "Promotion Underperforming",
                    "text": f"Promos showing <strong>{abs(lift):.1f}% lower</strong> sales. "
                            f"Review pricing strategy and customer targeting."
                })
    
    # Weekend effect
    if "is_weekend" in df.columns:
        wk = df[df["is_weekend"] == 1]["sales"].mean()
        wkd = df[df["is_weekend"] == 0]["sales"].mean()
        if pd.notna(wk) and pd.notna(wkd) and wkd > 0:
            diff = (wk - wkd) / wkd * 100
            if diff > 10:
                insights.append({
                    "icon": "📅",
                    "title": "Strong Weekend Demand",
                    "text": f"Weekend sales are <strong>{abs(diff):.1f}% higher</strong>. "
                            f"Staff up on Fridays and increase stock buffers before weekends."
                })
            elif diff < -10:
                insights.append({
                    "icon": "🏢",
                    "title": "Weekday Preference",
                    "text": f"Weekday sales are <strong>{abs(diff):.1f}% higher</strong>. "
                            f"Weekday demand is the primary driver — optimize delivery routes accordingly."
                })
    
    # Volatility alert
    if std > 0 and avg > 0:
        cv = std / avg
        if cv > 0.8:
            insights.append({
                "icon": "⚠️",
                "title": "High Sales Volatility",
                "text": f"Coefficient of variation is <strong>{cv:.2f}</strong> — demand is highly erratic. "
                        f"Recommend 25-35% safety stock and more frequent reforecasting (weekly)."
            })
    
    # Zero sales alert
    zero_pct = (df["sales"] == 0).mean() * 100
    if zero_pct > 15:
        insights.append({
            "icon": "🚨",
            "title": "Stockout / Zero-Sales Risk",
            "text": f"<strong>{zero_pct:.1f}%</strong> of records show zero sales. "
                    f"May indicate stockouts, listing gaps, or seasonal inactivity. Investigate urgently."
        })
    
    # Top family
    if df["family"].nunique() > 0:
        top_fam = df.groupby("family")["sales"].sum().idxmax()
        top_fam_pct = df.groupby("family")["sales"].sum().max() / total * 100
        insights.append({
            "icon": "📦",
            "title": "Revenue Leader",
            "text": f"<strong>'{top_fam}'</strong> is the highest-revenue family ({top_fam_pct:.1f}%). "
                    f"Prioritize inventory and marketing budget allocation here."
        })
    
    # Demand stability
    if std > 0:
        skew = (df["sales"].skew()) if len(df) > 3 else 0
        if skew > 1:
            insights.append({
                "icon": "📈",
                "title": "High-Demand Tail Events",
                "text": f"Distribution is right-skewed ({skew:.2f}). "
                        f"Occasional high-demand spikes detected. Plan surge capacity."
            })
    
    return insights


def forecast_next_weeks(df: pd.DataFrame, model, feature_cols: list, weeks: int = 4):
    """Generate multi-step forecasts with dynamic feature updates."""
    try:
        df_eng, X, _, _ = engineer_features(df)
        
        if not feature_cols:
            feature_cols = X.columns.tolist()
        
        X = X[feature_cols].copy()
        last_row = X.iloc[[-1]].copy()
        
        preds = []
        dates = []
        
        last_date = pd.to_datetime(df["date"].max())
        
        for step in range(weeks):
            pred = float(model.predict(last_row.values)[0])
            pred = max(0, pred)
            preds.append(pred)
            
            forecast_date = last_date + timedelta(weeks=step+1)
            dates.append(forecast_date)
            
            # Update lag features dynamically
            if "lag_1" in last_row.columns:
                last_row.loc[:, "lag_1"] = pred
            if "lag_7" in last_row.columns and step >= 6:
                last_row.loc[:, "lag_7"] = pred
            
            # Update rolling features
            if "rolling_mean_7" in last_row.columns:
                last_row.loc[:, "rolling_mean_7"] = pred * 0.9 + last_row["rolling_mean_7"].iloc[0] * 0.1
            if "rolling_mean_14" in last_row.columns:
                last_row.loc[:, "rolling_mean_14"] = pred * 0.85 + last_row["rolling_mean_14"].iloc[0] * 0.15
            
            # Update trend
            if "trend" in last_row.columns and "rolling_mean_7" in last_row.columns and "rolling_mean_14" in last_row.columns:
                last_row.loc[:, "trend"] = last_row["rolling_mean_7"].iloc[0] - last_row["rolling_mean_14"].iloc[0]
            
            # Update time features for next week
            if "day_of_week" in last_row.columns:
                last_row.loc[:, "day_of_week"] = (last_row["day_of_week"].iloc[0] + 1) % 7
            if "day_of_month" in last_row.columns:
                last_row.loc[:, "day_of_month"] = min(last_row["day_of_month"].iloc[0] + 7, 28)
        
        return pd.DataFrame({"date": dates, "forecast": preds})
    
    except Exception as e:
        print(f"Forecast error: {e}")
        return pd.DataFrame()


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════
if "df" not in st.session_state:
    st.session_state.df = None
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "source_label" not in st.session_state:
    st.session_state.source_label = ""
if "needs_mapping" not in st.session_state:
    st.session_state.needs_mapping = False
if "show_retrain_modal" not in st.session_state:
    st.session_state.show_retrain_modal = False


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="logo-bar">
      <div class="logo-mark">DS</div>
      <div>
        <div class="logo-text">DemandSense</div>
        <div class="logo-sub">AI FORECAST</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.df is not None and not st.session_state.needs_mapping:
        df_side = st.session_state.get("df_filtered", st.session_state.df).copy()
        _cols = df_side.columns.tolist()
        
        required_filter_cols = ["store_nbr", "family"]
        if all(c in _cols for c in required_filter_cols):
            st.markdown('<div class="sec-header">Filters</div>', unsafe_allow_html=True)
            
            stores = sorted(df_side["store_nbr"].unique().tolist())
            sel_stores = st.multiselect("Stores", stores, default=stores[:min(3, len(stores))])
            
            families = sorted(df_side["family"].unique().tolist())
            sel_fam = st.multiselect("Product Family", families, default=families[:min(2, len(families))])
            
            if sel_stores:
                df_side = df_side[df_side["store_nbr"].isin(sel_stores)]
            if sel_fam:
                df_side = df_side[df_side["family"].isin(sel_fam)]
            
            st.session_state.df_filtered = df_side
        
        st.markdown('<div class="sec-header">Model Status</div>', unsafe_allow_html=True)
        model_exists = os.path.exists(MODEL_PATH)
        if model_exists:
            st.markdown('<div class="alert-success">✅ Trained model available</div>', unsafe_allow_html=True)
            if st.button("🔄 Retrain Model", use_container_width=True):
                st.session_state.show_retrain_modal = True
                st.rerun()
        else:
            st.markdown('<div class="alert-warn">⚠️ No model yet — run <code>python src/train_model.py</code></div>',
                       unsafe_allow_html=True)
        
        st.markdown('<div class="sec-header">Data Summary</div>', unsafe_allow_html=True)
        st.caption(f"📊 {st.session_state.source_label}")
        st.caption(f"📈 Rows: {len(df_side):,}")
        st.caption(f"📅 {df_side['date'].min().strftime('%d %b %Y')} → {df_side['date'].max().strftime('%d %b %Y')}")
        
        st.markdown('<div class="sec-header">Actions</div>', unsafe_allow_html=True)
        if st.button("📂 Upload New File", use_container_width=True):
            st.session_state.df = None
            st.session_state.df_raw = None
            st.session_state.needs_mapping = False
            st.session_state.source_label = ""
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# RETRAIN MODAL (if activated)
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.show_retrain_modal:
    st.markdown("""
    <div style="position:fixed; top:0; left:0; right:0; bottom:0; background:rgba(0,0,0,0.4);
                display:flex; align-items:center; justify-content:center; z-index:9999;">
      <div style="background:white; border-radius:20px; padding:2rem; box-shadow:0 20px 60px rgba(0,0,0,0.3);
                  max-width:500px; animation:slideUp 0.3s ease-out;">
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("✅ Confirm Retrain", use_container_width=True, type="primary"):
            with st.spinner("⏳ Retraining model..."):
                try:
                    result = subprocess.run(
                        [sys.executable, TRAIN_SCRIPT],
                        capture_output=True,
                        text=True,
                        timeout=600
                    )
                    if result.returncode == 0:
                        st.success("✅ Model retrained successfully!")
                        st.session_state.show_retrain_modal = False
                        st.rerun()
                    else:
                        st.error(f"❌ Training failed:\n{result.stderr}")
                        st.session_state.show_retrain_modal = False
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.session_state.show_retrain_modal = False
    
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.show_retrain_modal = False
            st.rerun()
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# COLUMN MAPPING SCREEN
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.df_raw is not None and st.session_state.needs_mapping:
    df_raw = st.session_state.df_raw
    
    st.markdown("""
    <div style="text-align:center; padding: 2rem 1rem 1.5rem;">
      <div class="hero-eyebrow">
        <span class="pulse" style="width:7px;height:7px;background:#3B6FF0;border-radius:50%;
                                   display:inline-block;"></span>
        Step 2 of 2 — Column Mapping
      </div>
      <h1 style="font-size:2.6rem; font-weight:800; color:#0F172A; margin:0 0 14px;
                 letter-spacing:-0.04em; line-height:1.1;">
        Map Your Columns
      </h1>
      <p style="font-size:15px; color:#475569; max-width:680px; margin:0 auto 1.8rem;
                line-height:1.7; font-weight:400;">
        Your file's columns don't match the model's expected schema. We've auto-detected
        the best matches below — review and adjust them, then continue.
      </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("👀 Preview your data (first 10 rows)", expanded=False):
        st.dataframe(df_raw.head(10), use_container_width=True)
        st.caption(f"Detected {df_raw.shape[1]} columns, {len(df_raw):,} rows")
    
    auto_map = auto_detect_mapping(df_raw.columns.tolist())
    
    st.markdown('<div class="map-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-header" style="margin-top:0">⚙️ Map your columns</div>',
        unsafe_allow_html=True
    )
    
    user_mapping = {}
    available_cols = ["— Not in my file —"] + sorted(df_raw.columns.tolist())
    
    for target_col, info in REQUIRED_SCHEMA.items():
        col_a, col_b = st.columns([1.2, 1])
        
        with col_a:
            badge = '<span class="map-required-badge">REQUIRED</span>' if info["required"] \
                    else '<span class="map-optional-badge">OPTIONAL</span>'
            st.markdown(
                f"""<div style="padding-top:20px;">
                       <div style="font-weight:700; font-size:14.5px; color:#0F172A;">
                           {info['label']} {badge}
                       </div>
                       <div style="font-size:12.5px; color:#64748B; margin-top:3px; line-height:1.5;">
                           {info['description']}
                       </div>
                       <div style="font-size:10px; color:#94A3B8; margin-top:5px;
                                   font-family:var(--mono); font-weight:600; letter-spacing:0.05em;">
                           → {target_col}
                       </div>
                    </div>""",
                unsafe_allow_html=True
            )
        
        with col_b:
            detected = auto_map.get(target_col)
            default_idx = 0
            if detected and detected in df_raw.columns:
                default_idx = sorted(df_raw.columns.tolist()).index(detected) + 1
            
            picked = st.selectbox(
                f"Pick column for {target_col}",
                available_cols,
                index=default_idx,
                key=f"map_{target_col}",
                label_visibility="collapsed"
            )
            user_mapping[target_col] = None if picked == "— Not in my file —" else picked
            
            if detected and picked == detected:
                st.markdown(
                    f'<div style="font-size:11px; color:#00A388; margin-top:-10px; '
                    f'font-family:var(--mono); font-weight:700;">✓ auto-detected</div>',
                    unsafe_allow_html=True
                )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Validation
    missing_required = [
        REQUIRED_SCHEMA[k]["label"]
        for k, v in user_mapping.items()
        if REQUIRED_SCHEMA[k]["required"] and v is None
    ]
    
    picked_cols = [v for v in user_mapping.values() if v is not None]
    duplicates = [c for c in picked_cols if picked_cols.count(c) > 1]
    
    if missing_required:
        st.markdown(
            f'<div class="alert-danger">❌ Please map: <strong>{", ".join(missing_required)}</strong></div>',
            unsafe_allow_html=True
        )
    if duplicates:
        st.markdown(
            f'<div class="alert-danger">❌ Duplicate mappings: <strong>{", ".join(set(duplicates))}</strong></div>',
            unsafe_allow_html=True
        )
    
    btn_a, _, btn_c = st.columns([1, 0.2, 1])
    with btn_a:
        if st.button("← Back", use_container_width=True):
            st.session_state.df_raw = None
            st.session_state.needs_mapping = False
            st.rerun()
    with btn_c:
        if st.button("✓ Apply & Continue", use_container_width=True,
                    disabled=bool(missing_required or duplicates), type="primary"):
            try:
                df_mapped = apply_mapping(df_raw, user_mapping)
                still_missing = [c for c in ["date", "store_nbr", "family", "sales"] if c not in df_mapped.columns]
                if still_missing:
                    st.error(f"After mapping, still missing: {still_missing}")
                elif len(df_mapped) == 0:
                    st.error("No valid rows after parsing dates.")
                else:
                    st.session_state.df = df_mapped
                    st.session_state.needs_mapping = False
                    st.session_state.df_raw = None
                    st.rerun()
            except Exception as e:
                st.error(f"Mapping failed: {e}")
    
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# LANDING / UPLOAD SCREEN
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.df is None:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 2rem 2rem;">
      <div class="hero-eyebrow">
        <span style="width:7px;height:7px;background:#3B6FF0;border-radius:50%;
                     display:inline-block;box-shadow:0 0 12px #3B6FF0;"></span>
        AI-Powered Sales Intelligence
      </div>
      <h1 style="font-size:3.8rem; font-weight:800; color:#0F172A; margin:0 0 18px;
                 letter-spacing:-0.04em; line-height:1.1; font-family:'Sora',sans-serif;">
        DemandSense <span style="background:linear-gradient(135deg,#3B6FF0,#00C9A7);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;">AI</span>
      </h1>
      <p style="font-size:17px; color:#475569; max-width:700px; margin:0 auto 2.8rem;
                line-height:1.8; font-weight:400;">
        Upload your sales data and unlock AI-driven forecasts, promotion analysis,
        and actionable insights. Any column names work — we'll map them for you.
      </p>
      <div style="display:flex; justify-content:center; gap:24px; flex-wrap:wrap; margin-top:2.5rem;">
        <div style="text-align:left; background:#FFFFFF; border:1.5px solid #C8D6F8;
                    border-radius:18px; padding:24px 28px; min-width:180px;
                    box-shadow:0 6px 24px rgba(59,111,240,0.1);
                    transition:all 0.2s; cursor:pointer;">
          <div style="font-size:28px; margin-bottom:10px;">📄</div>
          <div style="font-size:17px; font-weight:800; color:#3B6FF0; letter-spacing:-0.02em;">CSV / Excel</div>
          <div style="font-size:13px; color:#475569; margin-top:6px; line-height:1.6;">Flexible schema — any file format</div>
        </div>
        <div style="text-align:left; background:#FFFFFF; border:1.5px solid #BBF0E6;
                    border-radius:18px; padding:24px 28px; min-width:180px;
                    box-shadow:0 6px 24px rgba(0,201,167,0.1);
                    transition:all 0.2s; cursor:pointer;">
          <div style="font-size:28px; margin-bottom:10px;">🔗</div>
          <div style="font-size:17px; font-weight:800; color:#00A388; letter-spacing:-0.02em;">Google Sheets</div>
          <div style="font-size:13px; color:#475569; margin-top:6px; line-height:1.6;">Live data — real-time updates</div>
        </div>
        <div style="text-align:left; background:#FFFFFF; border:1.5px solid #FDE68A;
                    border-radius:18px; padding:24px 28px; min-width:180px;
                    box-shadow:0 6px 24px rgba(245,158,11,0.1);
                    transition:all 0.2s; cursor:pointer;">
          <div style="font-size:28px; margin-bottom:10px;">⚡</div>
          <div style="font-size:17px; font-weight:800; color:#D97706; letter-spacing:-0.02em;">Demo Dataset</div>
          <div style="font-size:13px; color:#475569; margin-top:6px; line-height:1.6;">Instant exploration — no setup</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    
    source_tab = st.radio(
        "Choose input method",
        ["📂 Upload File", "🔗 Google Sheet URL", "🎬 Load Demo Data"],
        horizontal=True
    )
    
    def _ingest_dataframe(df_raw: pd.DataFrame, source_label: str):
        st.session_state.source_label = source_label
        
        if schema_already_matches(df_raw.columns.tolist()):
            df_clean = df_raw.copy()
            if "onpromotion" not in df_clean.columns:
                df_clean["onpromotion"] = 0
            df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
            df_clean = df_clean.dropna(subset=["date"])
            for c in ["sales", "onpromotion"]:
                if c in df_clean.columns:
                    df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce").fillna(0)
            st.session_state.df = df_clean
            st.session_state.needs_mapping = False
            st.session_state.df_raw = None
        else:
            st.session_state.df_raw = df_raw
            st.session_state.needs_mapping = True
            st.session_state.df = None
    
    if source_tab == "📂 Upload File":
        uploaded = st.file_uploader(
            "Drop CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Any column names work — you'll map them next."
        )
        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    df_raw = pd.read_csv(uploaded)
                else:
                    df_raw = pd.read_excel(uploaded)
                _ingest_dataframe(df_raw, f"📂 {uploaded.name}")
                st.markdown('<div class="alert-success">✓ File loaded</div>', unsafe_allow_html=True)
                st.rerun()
            except Exception as e:
                st.markdown(f'<div class="alert-danger">✗ Error: {e}</div>', unsafe_allow_html=True)
    
    elif source_tab == "🔗 Google Sheet URL":
        gsheet_url = st.text_input(
            "Paste Google Sheet URL",
            placeholder="https://docs.google.com/spreadsheets/d/..."
        )
        if st.button("Load from Google Sheets"):
            if gsheet_url:
                with st.spinner("Fetching sheet..."):
                    try:
                        df_raw = load_gsheet(gsheet_url)
                        _ingest_dataframe(df_raw, "🔗 Google Sheets")
                        st.rerun()
                    except Exception as e:
                        st.markdown(f'<div class="alert-danger">✗ {e}</div>', unsafe_allow_html=True)
            else:
                st.warning("Paste a URL first.")
    
    else:
        st.markdown('<div class="alert-info">📊 Uses demo_sales.csv from data/processed/</div>',
                   unsafe_allow_html=True)
        if st.button("Load Demo Dataset"):
            if os.path.exists(DEMO_PATH):
                df_raw = pd.read_csv(DEMO_PATH)
                _ingest_dataframe(df_raw, "🎬 Demo Dataset")
                st.rerun()
            else:
                st.markdown('<div class="alert-danger">✗ demo_sales.csv not found</div>',
                           unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
df = st.session_state.get("df_filtered", st.session_state.df).copy()
required_cols = ["date", "store_nbr", "family", "sales", "onpromotion"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

df["date"] = pd.to_datetime(df["date"])
df_sorted = df.sort_values("date")

model, meta = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:space-between;
            margin-bottom:1.8rem; border-bottom:2px solid #C8D6F8; padding-bottom:1.3rem;">
  <div>
    <h2 style="margin:0; font-size:26px; font-weight:800; color:#0F172A; letter-spacing:-0.04em;">
      📊 Demand Intelligence
    </h2>
    <div style="font-size:13px; color:#475569; margin-top:6px; font-weight:500;">
      {st.session_state.source_label} &nbsp;·&nbsp;
      {len(df):,} records &nbsp;·&nbsp;
      {df['date'].min().strftime('%d %b %Y')} → {df['date'].max().strftime('%d %b %Y')}
    </div>
  </div>
  <div style="font-family:var(--mono); font-size:11px; color:#3B6FF0;
              background:rgba(59,111,240,0.12); border:1.5px solid rgba(59,111,240,0.3);
              padding:8px 18px; border-radius:99px; font-weight:700; letter-spacing:0.12em;
              display:flex; align-items:center; gap:8px;">
    <span style="width:7px;height:7px;background:#3B6FF0;border-radius:50%;
                 box-shadow:0 0 8px #3B6FF0;display:inline-block;"></span>
    LIVE
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("💰 Total Sales", f"{df['sales'].sum():,.0f}")
k2.metric("📈 Avg Daily", f"{df['sales'].mean():.1f}")
k3.metric("🔝 Peak Sales", f"{df['sales'].max():,.0f}")
k4.metric("🏪 Stores", f"{df['store_nbr'].nunique()}")
k5.metric("📦 SKU Families", f"{df['family'].nunique()}")

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "🔮 Forecast",
    "📣 Promo Analysis",
    "🤖 AI Insights",
    "📋 Model Details",
    "🗂️ Data"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="sec-header">Time Series Analysis</div>', unsafe_allow_html=True)
    
    col_l, col_r = st.columns([2, 1])
    
    with col_l:
        daily = df_sorted.groupby("date")["sales"].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["sales"],
            mode="lines", name="Daily Sales",
            line=dict(color="#3B6FF0", width=2.8),
            fill="tozeroy", fillcolor="rgba(59,111,240,0.08)",
            hovertemplate="<b>%{x|%d %b}</b><br>Sales: %{y:,.0f}<extra></extra>"
        ))
        fig.update_layout(title="Sales Over Time", **PLOTLY_LAYOUT, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_r:
        by_store = df.groupby("store_nbr")["sales"].sum().reset_index().sort_values("sales", ascending=False).head(10)
        fig2 = px.bar(
            by_store, y="store_nbr", x="sales", orientation="h",
            title="Top 10 Stores",
            color="sales", color_continuous_scale=["#C7D8FF", "#3B6FF0"]
        )
        fig2.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        fig2.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig2, use_container_width=True)
    
    col_ll, col_rr = st.columns(2)
    
    with col_ll:
        by_fam = df.groupby("family")["sales"].sum().sort_values(ascending=False).reset_index()
        fig3 = px.bar(
            by_fam, y="family", x="sales", orientation="h",
            title="Sales by Product Family",
            color="sales", color_continuous_scale=["#B2F0E8", "#00C9A7"]
        )
        fig3.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        fig3.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig3, use_container_width=True)
    
    with col_rr:
        df["month_label"] = df["date"].dt.to_period("M").astype(str)
        monthly = df.groupby("month_label")["sales"].sum().reset_index()
        fig4 = px.line(
            monthly, x="month_label", y="sales",
            title="Monthly Trend",
            markers=True
        )
        fig4.update_traces(
            line=dict(color="#F59E0B", width=3),
            marker=dict(color="#F59E0B", size=10, line=dict(width=2, color="white"))
        )
        fig4.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig4, use_container_width=True)
    
    if df["store_nbr"].nunique() > 1:
        st.markdown('<div class="sec-header">Store × Month Heatmap</div>', unsafe_allow_html=True)
        pivot = df.pivot_table(
            index="store_nbr", columns="month_label", values="sales", aggfunc="sum"
        ).fillna(0)
        fig5 = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(),
            y=[f"Store {s}" for s in pivot.index],
            colorscale=[[0, "#EEF2FF"], [0.5, "#3B6FF0"], [1, "#00C9A7"]],
            hovertemplate="Store: %{y}<br>Month: %{x}<br>Sales: %{z:,.0f}<extra></extra>"
        ))
        fig5.update_layout(title="Sales Heatmap", **PLOTLY_LAYOUT)
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    if model is None:
        st.markdown('<div class="alert-warn">⚠️ No trained model found. Run <code>python src/train_model.py</code> first.</div>',
                   unsafe_allow_html=True)
    else:
        if meta and "metrics" in meta:
            m = meta["metrics"]
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("MAE", f"{m.get('MAE', '-')}")
            mc2.metric("RMSE", f"{m.get('RMSE', '-')}")
            mc3.metric("R²", f"{m.get('R2', '-')}")
            mc4.metric("SMAPE", f"{m.get('SMAPE', '-')}%")
            mc5.metric("WAPE", f"{m.get('WAPE', '-')}%")
        
        st.markdown('<div class="sec-header">Next-Week Forecast</div>', unsafe_allow_html=True)
        
        weeks_ahead = st.slider("Weeks to forecast", 1, 12, 4, help="How many weeks ahead to predict")
        fc_df = forecast_next_weeks(df, model, meta.get("feature_cols", []), weeks=weeks_ahead)
        
        if not fc_df.empty:
            hist_tail = df_sorted.groupby("date")["sales"].sum().reset_index().tail(30)
            hist_tail["date"] = pd.to_datetime(hist_tail["date"], errors="coerce")
            fc_df["date"] = pd.to_datetime(fc_df["date"], errors="coerce")
            hist_tail = hist_tail.dropna(subset=["date"])
            fc_df = fc_df.dropna(subset=["date"]).sort_values("date")
            
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=hist_tail["date"], y=hist_tail["sales"],
                mode="lines+markers", name="Historical",
                line=dict(color="#3B6FF0", width=2.8),
                marker=dict(size=6)
            ))
            fig_fc.add_trace(go.Scatter(
                x=fc_df["date"], y=fc_df["forecast"],
                mode="lines+markers", name="Forecast",
                line=dict(color="#00C9A7", width=2.8, dash="dot"),
                marker=dict(size=10, symbol="diamond"),
                fill="tozeroy", fillcolor="rgba(0,201,167,0.08)"
            ))
            
            upper_band = fc_df["forecast"] * 1.2
            lower_band = fc_df["forecast"] * 0.8
            fig_fc.add_trace(go.Scatter(
                x=list(fc_df["date"]) + list(fc_df["date"])[::-1],
                y=list(upper_band) + list(lower_band)[::-1],
                fill="toself", fillcolor="rgba(0,201,167,0.1)",
                line=dict(color="rgba(0,0,0,0)"), name="±20% Confidence Band", showlegend=True
            ))
            
            if not hist_tail.empty:
                today_marker_x = pd.to_datetime(hist_tail["date"].iloc[-1])
                fig_fc.add_shape(
                    type="line", x0=today_marker_x, x1=today_marker_x, y0=0, y1=1,
                    xref="x", yref="paper", line=dict(dash="dash", color="#94A3B8", width=1.5),
                )
                fig_fc.add_annotation(
                    x=today_marker_x, y=1, xref="x", yref="paper",
                    text="Today", showarrow=False, yshift=10,
                    font=dict(color="#94A3B8", size=12),
                )
            
            fig_fc.update_layout(title="Sales Forecast (Next 4-12 Weeks)", **PLOTLY_LAYOUT, hovermode="x unified")
            st.plotly_chart(fig_fc, use_container_width=True)
            
            fc_display = fc_df.copy()
            fc_display["date"] = fc_display["date"].dt.strftime("%d %b %Y (%A)")
            fc_display["forecast"] = fc_display["forecast"].round(1)
            fc_display.columns = ["Week", "Forecasted Sales"]
            st.dataframe(fc_display, use_container_width=True, hide_index=True)
        
        if meta and "feature_importance" in meta:
            st.markdown('<div class="sec-header">Feature Importance</div>', unsafe_allow_html=True)
            st.markdown("**Model relies heavily on:** past sales trends, seasonal patterns, and promotional activity.")
            
            fi = pd.Series(meta["feature_importance"]).sort_values()
            fig_fi = px.bar(
                x=fi.values, y=fi.index, orientation="h",
                title="Top 15 Features Driving Forecasts",
                color=fi.values,
                color_continuous_scale=["#C7D8FF", "#3B6FF0", "#00C9A7"]
            )
            fig_fi.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: PROMO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="sec-header">Promotion Impact Analysis</div>', unsafe_allow_html=True)
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        promo_grp = df.groupby("onpromotion")["sales"].agg(["mean", "count"]).reset_index()
        promo_grp["label"] = promo_grp["onpromotion"].apply(
            lambda x: "No Promotion" if x == 0 else f"{int(x)} items"
        )
        fig_p1 = px.bar(
            promo_grp, x="label", y="mean",
            title="Avg Sales by Promotion Level",
            color="mean", color_continuous_scale=["#FEF3C7", "#F59E0B"],
            labels={"mean": "Average Sales"}
        )
        fig_p1.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        fig_p1.update_yaxes(title_text="Average Sales")
        st.plotly_chart(fig_p1, use_container_width=True)
    
    with col_p2:
        promo_fam = df.groupby(["family", "onpromotion"])["sales"].mean().reset_index()
        promo_fam["promo_label"] = promo_fam["onpromotion"].apply(
            lambda x: "On Promotion" if x > 0 else "No Promo"
        )
        fig_p2 = px.bar(
            promo_fam, x="family", y="sales", color="promo_label",
            title="Promo Effect by Product Family",
            barmode="group",
            color_discrete_map={"On Promotion": "#F59E0B", "No Promo": "#3B6FF0"}
        )
        fig_p2.update_layout(**PLOTLY_LAYOUT)
        fig_p2.update_yaxes(title_text="Average Sales")
        st.plotly_chart(fig_p2, use_container_width=True)
    
    # Weekend effect
    if "is_weekend" in df.columns or "day_of_week" in df.columns:
        st.markdown('<div class="sec-header">Temporal Patterns</div>', unsafe_allow_html=True)
        
        if "is_weekend" in df.columns:
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                wk_grp = df.groupby("is_weekend")["sales"].mean().reset_index()
                wk_grp["label"] = wk_grp["is_weekend"].map({0: "Weekday", 1: "Weekend"})
                fig_p3 = px.pie(
                    wk_grp, values="sales", names="label",
                    title="Weekday vs Weekend Sales",
                    color_discrete_sequence=["#3B6FF0", "#00C9A7"]
                )
                fig_p3.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_p3, use_container_width=True)
            
            with col_t2:
                if "day_of_week" in df.columns:
                    daily_pattern = df.groupby("day_of_week")["sales"].mean().reset_index()
                    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    daily_pattern["day_name"] = daily_pattern["day_of_week"].apply(
                        lambda x: day_names[x] if x < 7 else str(x)
                    )
                    fig_p4 = px.bar(
                        daily_pattern, x="day_name", y="sales",
                        title="Avg Sales by Day of Week",
                        color="sales", color_continuous_scale=["#EDE9FE", "#8B5CF6"]
                    )
                    fig_p4.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
                    fig_p4.update_yaxes(title_text="Average Sales")
                    st.plotly_chart(fig_p4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: AI INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:1.8rem;">
      <div class="pulse" style="width:9px; height:9px; background:#3B6FF0; border-radius:50%;
                                box-shadow:0 0 12px #3B6FF0;"></div>
      <span style="font-size:12px; font-family:var(--mono); color:#3B6FF0;
                   letter-spacing:0.15em; font-weight:800;">AI ENGINE ACTIVE</span>
    </div>
    """, unsafe_allow_html=True)
    
    insights = generate_ai_insights(df)
    
    cols_per_row = 2
    for i in range(0, len(insights), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col_idx in enumerate(range(i, min(i + cols_per_row, len(insights)))):
            with cols[j]:
                ins = insights[col_idx]
                st.markdown(f"""
                <div class="ai-card">
                  <div class="ai-label">💡 AI Insight</div>
                  <div class="ai-text"><strong>{ins['title']}</strong><br><br>{ins['text']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('<div class="sec-header">Anomaly Detection</div>', unsafe_allow_html=True)
    daily_anom = df_sorted.groupby("date")["sales"].sum().reset_index()
    rolling_mu = daily_anom["sales"].rolling(7, min_periods=1).mean()
    rolling_sd = daily_anom["sales"].rolling(7, min_periods=1).std().fillna(1)
    daily_anom["z_score"] = (daily_anom["sales"] - rolling_mu) / (rolling_sd + 1e-9)
    anomalies = daily_anom[daily_anom["z_score"].abs() > 2.5]
    
    fig_an = go.Figure()
    fig_an.add_trace(go.Scatter(
        x=daily_anom["date"], y=daily_anom["sales"],
        mode="lines", name="Sales", line=dict(color="#3B6FF0", width=2.5),
        fill="tozeroy", fillcolor="rgba(59,111,240,0.07)"
    ))
    if not anomalies.empty:
        fig_an.add_trace(go.Scatter(
            x=anomalies["date"], y=anomalies["sales"],
            mode="markers", name="Anomaly",
            marker=dict(color="#EF4444", size=14, symbol="x", line=dict(color="#EF4444", width=3))
        ))
    fig_an.update_layout(title="Anomaly Detection (Z-score > 2.5σ)", **PLOTLY_LAYOUT)
    st.plotly_chart(fig_an, use_container_width=True)
    
    if not anomalies.empty:
        st.markdown(f"""
        <div class="alert-danger">
          🚨 <strong>{len(anomalies)} anomalies detected</strong> — Days where sales deviated >2.5σ from the rolling average.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-success">✅ No significant anomalies detected.</div>', unsafe_allow_html=True)
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        fig_dist = px.histogram(
            df, x="sales", nbins=50,
            title="Sales Distribution",
            color_discrete_sequence=["#3B6FF0"]
        )
        fig_dist.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col_d2:
        if "store_nbr" in df.columns:
            store_stats = df.groupby("store_nbr")["sales"].std().reset_index()
            store_stats.columns = ["Store", "Volatility (Std Dev)"]
            store_stats = store_stats.sort_values("Volatility (Std Dev)", ascending=False).head(10)
            fig_vol = px.bar(
                store_stats, x="Volatility (Std Dev)", y="Store", orientation="h",
                title="Top 10 Stores by Sales Volatility",
                color="Volatility (Std Dev)",
                color_continuous_scale=["#E0E7FF", "#EF4444"]
            )
            fig_vol.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig_vol, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: MODEL DETAILS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="sec-header">Model Information</div>', unsafe_allow_html=True)
    
    if model is None:
        st.markdown('<div class="alert-warn">⚠️ No trained model. Run <code>python src/train_model.py</code></div>',
                   unsafe_allow_html=True)
    else:
        if meta:
            st.markdown(f"""
            <div class="alert-info">
              <strong>Model Description:</strong><br>
              {meta.get('model_description', 'XGBoost regression model trained on time-series sales data.')}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="sec-header">Performance Metrics</div>', unsafe_allow_html=True)
            metrics_df = pd.DataFrame([meta.get("metrics", {})])
            st.dataframe(metrics_df.T, use_container_width=True)
            
            st.markdown('<div class="sec-header">Training Data Summary</div>', unsafe_allow_html=True)
            info_cols = st.columns(3)
            info_cols[0].metric("Training Rows", f"{meta.get('train_rows', '-'):,}")
            info_cols[1].metric("Test Rows", f"{meta.get('test_rows', '-'):,}")
            info_cols[2].metric("Total Rows Used", f"{meta.get('train_rows', 0) + meta.get('test_rows', 0):,}")
            
            st.markdown('<div class="sec-header">Features Used</div>', unsafe_allow_html=True)
            feature_list = meta.get("feature_cols", [])
            
            col_feat1, col_feat2 = st.columns(2)
            with col_feat1:
                st.markdown("**Temporal Features:**")
                temporal = [f for f in feature_list if any(x in f for x in ["month", "day", "quarter", "weekend", "start", "end"])]
                for f in temporal:
                    st.caption(f"• {f}")
            
            with col_feat2:
                st.markdown("**Lag & Trend Features:**")
                lag_trend = [f for f in feature_list if any(x in f for x in ["lag", "rolling", "trend"])]
                for f in lag_trend:
                    st.caption(f"• {f}")
            
            st.markdown("**Categorical & Business Features:**")
            other = [f for f in feature_list if f not in temporal and f not in lag_trend]
            for f in other:
                st.caption(f"• {f}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: DATA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="sec-header">Data Summary</div>', unsafe_allow_html=True)
    
    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.metric("Total Rows", f"{len(df):,}")
    col_d2.metric("Total Columns", f"{df.shape[1]}")
    col_d3.metric("Date Range", f"{(df['date'].max() - df['date'].min()).days} days")
    
    st.markdown('<div class="sec-header">Preview (First 50 Rows)</div>', unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True)
    
    st.markdown('<div class="sec-header">Column Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe(include="all").T, use_container_width=True)
    
    st.markdown('<div class="sec-header">Export Data</div>', unsafe_allow_html=True)
    csv_out = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download as CSV",
        data=csv_out,
        file_name=f"demandsense_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
