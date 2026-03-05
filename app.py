# ==========================================#
# Time Series Forecasting — Streamlit App  #
# ==========================================#

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error

### Page Config ###
st.set_page_config(
    page_title="Time Series Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Time Series Forecaster")
st.markdown(
    "Upload **any CSV** with a datetime column and a numeric target column. "
    "The app will explore your data, train an XGBoost model, and analyse forecast accuracy."
)

# --------------------------------------------------
# Sidebar — File Upload
# --------------------------------------------------
st.sidebar.header("Data Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file",
    type=["csv"],
    help="Any CSV with at least one datetime column and one numeric column.",
)

if uploaded_file is None:
    st.info("Upload a CSV file in the sidebar to get started.")
    st.stop()

# --------------------------------------------------
# Load Raw CSV (columns unknown until file arrives)
# --------------------------------------------------
@st.cache_data
def read_raw(file) -> pd.DataFrame:
    return pd.read_csv(file)

raw_df = read_raw(uploaded_file)

# --------------------------------------------------
# Sidebar — Column Selection
# --------------------------------------------------
st.sidebar.header("Column Mapping")

all_cols = raw_df.columns.tolist()

# Auto-detect a datetime column
default_dt_idx = next(
    (i for i, c in enumerate(all_cols) if "date" in c.lower() or "time" in c.lower()), 0
)
datetime_col = st.sidebar.selectbox(
    "Datetime column",
    options=all_cols,
    index=default_dt_idx,
    help="Column containing timestamps or dates.",
)

# Numeric columns only for the target
numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found. Please upload a CSV with at least one numeric column.")
    st.stop()

target_col = st.sidebar.selectbox(
    "Target column (what to forecast)",
    options=numeric_cols,
    index=0,
    help="The numeric column you want the model to predict.",
)

# --------------------------------------------------
# Parse & index by datetime
# --------------------------------------------------
@st.cache_data
def load_data(file, dt_col: str) -> pd.DataFrame:
    df = pd.read_csv(file)
    df[dt_col] = pd.to_datetime(df[dt_col], infer_datetime_format=True)
    df = df.set_index(dt_col).sort_index()
    return df

try:
    df = load_data(uploaded_file, datetime_col)
except Exception as e:
    st.error(f"Could not parse datetime column '{datetime_col}': {e}")
    st.stop()

st.success(
    f"✅ Loaded **{len(df):,}** rows  |  "
    f"Date range: **{df.index.min().date()}** → **{df.index.max().date()}**  |  "
    f"Target: **{target_col}**"
)

# --------------------------------------------------
# Sidebar — Split & Model Settings
# --------------------------------------------------
st.sidebar.header("⚙️ Model Settings")

# Default split at 80% of the timeline
default_split = df.index.min() + (df.index.max() - df.index.min()) * 0.8
split_date = st.sidebar.date_input(
    "Train / Test Split Date",
    value=default_split.date(),
    min_value=df.index.min().date(),
    max_value=df.index.max().date(),
    help="Data up to this date → training; after → test.",
)

n_estimators = st.sidebar.slider(
    "XGBoost — n_estimators (trees)", 100, 2000, 1000, step=100
)
early_stopping = st.sidebar.slider(
    "XGBoost — early_stopping_rounds", 10, 100, 50, step=10
)

run_model = st.sidebar.button("Train Model", use_container_width=True)

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def create_features(df: pd.DataFrame, label: str = None):
    """Calendar-based time features — works for any datetime-indexed DataFrame."""
    d = df.copy()
    d["hour"]        = d.index.hour
    d["dayofweek"]   = d.index.dayofweek
    d["quarter"]     = d.index.quarter
    d["month"]       = d.index.month
    d["year"]        = d.index.year
    d["dayofyear"]   = d.index.dayofyear
    d["dayofmonth"]  = d.index.day
    d["weekofyear"]  = d.index.isocalendar().week.astype(int).values

    FEATURES = ["hour", "dayofweek", "quarter", "month", "year",
                "dayofyear", "dayofmonth", "weekofyear"]
    X = d[FEATURES]
    if label:
        y = d[label]
        return X, y
    return X


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# Tab Layout
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Data Explorer", "Train / Test Split", "Model & Predictions", "Error Analysis"]
)

# ==========================================
# TAB 1 — Data Explorer
# ==========================================
with tab1:
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader(f"Full Time Series — {target_col}")
    fig, ax = plt.subplots(figsize=(14, 4))
    df[target_col].plot(ax=ax, linewidth=0.7, color="#1f77b4")
    ax.set_title(f"{target_col} over time")
    ax.set_ylabel(target_col)
    ax.set_xlabel("")
    plt.tight_layout()
    show_fig(fig)

    st.subheader(f"Target Distribution — {target_col}")
    fig, ax = plt.subplots(figsize=(8, 3))
    df[target_col].hist(bins=60, ax=ax, color="#5C6BC0", edgecolor="none")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {target_col}")
    plt.tight_layout()
    show_fig(fig)

# ==========================================
# TAB 2 — Train / Test Split
# ==========================================
with tab2:
    split_ts = pd.Timestamp(split_date)
    df_train = df.loc[df.index <= split_ts].copy()
    df_test  = df.loc[df.index >  split_ts].copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total rows",    f"{len(df):,}")
    col2.metric("Training rows", f"{len(df_train):,}")
    col3.metric("Test rows",     f"{len(df_test):,}")

    if len(df_train) == 0 or len(df_test) == 0:
        st.warning("Split date leaves one set empty — adjust the split date in the sidebar.")
    else:
        st.subheader("Training vs Test Split")
        combined = df_test[[target_col]].rename(columns={target_col: "Test Set"}).join(
            df_train[[target_col]].rename(columns={target_col: "Training Set"}),
            how="outer"
        )
        fig, ax = plt.subplots(figsize=(14, 4))
        combined["Training Set"].plot(ax=ax, label="Training Set", linewidth=0.5, color="#2196F3")
        combined["Test Set"].plot(ax=ax, label="Test Set", linewidth=0.5, color="#FF5722")
        ax.set_title(f"{target_col} — Train / Test Split")
        ax.set_ylabel(target_col)
        ax.legend()
        plt.tight_layout()
        show_fig(fig)

# ==========================================
# TAB 3 — Model & Predictions
# ==========================================
with tab3:
    if not run_model:
        st.info("Configure settings in the sidebar, then click **🚀 Train Model** to start.")
    else:
        split_ts = pd.Timestamp(split_date)
        df_train = df.loc[df.index <= split_ts].copy()
        df_test  = df.loc[df.index >  split_ts].copy()

        if len(df_train) == 0 or len(df_test) == 0:
            st.error("Split date leaves one set empty — adjust the split date in the sidebar.")
            st.stop()

        X_train, y_train = create_features(df_train, label=target_col)
        X_test,  y_test  = create_features(df_test,  label=target_col)

        with st.spinner("Training XGBoost model..."):
            reg = xgb.XGBRegressor(
                n_estimators=n_estimators,
                early_stopping_rounds=early_stopping,
                eval_metric="rmse",
                verbosity=0,
            )
            reg.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False,
            )

        st.success(f"✅ Model trained using **{target_col}** as the target!")

        # — Metrics —
        preds  = reg.predict(X_test)
        rmse   = np.sqrt(mean_squared_error(y_test, preds))
        mae    = mean_absolute_error(y_test, preds)
        nonzero = y_test != 0
        mape   = np.mean(np.abs((y_test[nonzero] - preds[nonzero]) / y_test[nonzero])) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{rmse:,.2f}")
        c2.metric("MAE",  f"{mae:,.2f}")
        c3.metric("MAPE", f"{mape:.2f} %")

        # — Feature Importance —
        st.subheader("Feature Importance")
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_importance(reg, ax=ax, height=0.7, importance_type="gain")
        ax.set_title("XGBoost Feature Importance (Gain)")
        plt.tight_layout()
        show_fig(fig)

        # — Actual vs Predicted — Full Period —
        st.subheader("Actual vs Predicted — Full Period")
        X_all   = create_features(df)
        all_pred = reg.predict(X_all)

        fig, ax = plt.subplots(figsize=(14, 4))
        df[target_col].plot(ax=ax, label="Actual", linewidth=0.7, color="#AED6F1")
        ax.scatter(df_test.index, preds, label="Predicted",
                   color="#E74C3C", s=8, zorder=5)
        ax.set_title(f"Full Period: Actual vs Predicted — {target_col}")
        ax.set_ylabel(target_col)
        ax.legend()
        plt.tight_layout()
        show_fig(fig)
        
        # — Test Period Zoom —
        st.subheader("Actual vs Predicted — Test Period (Jan 2014 onwards)")
        
        fig3, ax3 = plt.subplots(figsize=(14, 4))
        ax3.plot(y_test.index, y_test.values,
                label="Actual", linewidth=1.2, color="#54A24B")
        ax3.plot(y_test.index, preds,
                label="XGBoost Forecast", linewidth=1.5,
                linestyle="--", color="#F58518")
        ax3.set_title(f"Test Period: Actual vs Predicted — {target_col}")
        ax3.set_ylabel(target_col)
        ax3.legend()
        plt.tight_layout()
        show_fig(fig3)

# ==========================================
# TAB 4 — Error Analysis
# ==========================================
with tab4:
    if not run_model:
        st.info("Train the model first (Tab 3) to see error analysis.")
    else:
        st.subheader("Residuals over Time")
        residuals = y_test.values - preds

        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(df_test.index, residuals, linewidth=0.7, color="#8E44AD")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title("Residuals (Actual − Predicted)")
        ax.set_ylabel("Residual")
        plt.tight_layout()
        show_fig(fig)

        st.subheader("Residual Distribution")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(residuals, bins=50, color="#5C6BC0", edgecolor="none")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Residuals")
        plt.tight_layout()
        show_fig(fig)

        st.subheader("Actual vs Predicted Scatter")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, preds, alpha=0.4, s=10, color="#2196F3")
        lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        plt.tight_layout()
        show_fig(fig)
