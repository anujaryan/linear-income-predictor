
import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Per-Capita Income Predictor", page_icon="📈")

st.title("📈 Linear Regression: Per-Capita Income vs Year")
st.write(
    "This Streamlit app mirrors your notebook: it fits a simple linear regression "
    "to predict per-capita income from **Year**. Upload the same CSV used in your notebook "
    "(`canada_per_capita_income.csv`) or use the synthetic sample to try it out."
)

# Sidebar controls
st.sidebar.header("Settings")
st.sidebar.markdown("**Data source**")
use_sample = st.sidebar.checkbox("Use synthetic sample data (if no file uploaded)", value=True)

uploaded = st.file_uploader("Upload CSV (expects columns: `year`, `per capita income (US$)` or `income`)",
                            type=["csv"])

def load_data(file):
    df = pd.read_csv(file)
    # Normalize expected column names
    df = df.rename(columns={"per capita income (US$)": "income"})
    # Only keep relevant columns
    if "year" not in df.columns or "income" not in df.columns:
        raise ValueError("CSV must contain columns: 'year' and 'per capita income (US$)' (or 'income').")
    # Coerce types
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["income"] = pd.to_numeric(df["income"], errors="coerce")
    df = df.dropna(subset=["year", "income"]).sort_values("year").reset_index(drop=True)
    return df

def sample_data():
    # Synthetic, gently increasing income over time (for demo only)
    years = np.arange(1970, 2026)
    rng = np.random.default_rng(42)
    base = 5000 + (years - 1970) * 400  # linear growth
    noise = rng.normal(0, 500, size=years.size)
    income = base + noise
    return pd.DataFrame({"year": years, "income": np.round(income, 2)})

# Load data
df = None
if uploaded is not None:
    try:
        df = load_data(uploaded)
        st.success("Loaded uploaded CSV.")
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
if df is None and use_sample:
    df = sample_data()
    st.info("Using synthetic sample data. Upload your CSV to use real data.")

if df is not None:
    st.subheader("Preview")
    st.dataframe(df.head(20))

    # Train the linear regression model
    X = df[["year"]].values
    y = df["income"].values
    model = LinearRegression()
    model.fit(X, y)

    # Compute metrics
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    st.subheader("Model Summary")
    st.markdown(f"- **Intercept**: `{model.intercept_:.4f}`")
    st.markdown(f"- **Coefficient (year)**: `{model.coef_[0]:.6f}`")
    st.markdown(f"- **R² on training data**: `{r2:.4f}`")

    # Visualization: scatter + regression line
    st.subheader("Fit Visualization")
    fig, ax = plt.subplots()
    ax.scatter(df["year"], df["income"], label="Data", alpha=0.8)
    years_line = np.linspace(df["year"].min(), df["year"].max(), 200).reshape(-1, 1)
    ax.plot(years_line, model.predict(years_line), label="Linear fit")
    ax.set_xlabel("Year")
    ax.set_ylabel("Per capita income (US$)")
    ax.legend()
    st.pyplot(fig)

    # Prediction UI
    st.subheader("Make a Prediction")
    min_year = int(df["year"].min())
    max_year = int(max(df["year"].max(), 2050))
    year_input = st.number_input("Enter a year to predict", min_value=min_year, max_value=max_year, value=min(2025, max_year), step=1)
    pred = float(model.predict(np.array([[year_input]]))[0])
    st.success(f"**Predicted per capita income for {year_input}: ${pred:,.2f}**")

    # Download artifacts
    st.subheader("Export")
    # Model artifact
    bytes_model = io.BytesIO()
    joblib.dump(model, bytes_model)
    st.download_button("Download trained model (.joblib)", data=bytes_model.getvalue(), file_name="linear_income_model.joblib")

    # Cleaned data
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button("Download cleaned dataset (CSV)", data=csv_buf.getvalue(), file_name="cleaned_income_data.csv")

    st.caption("Tip: To use the trained model elsewhere, load with `joblib.load('linear_income_model.joblib')` and call `.predict([[YEAR]])`.")
else:
    st.warning("Please upload a CSV or enable the synthetic sample in the sidebar.")
