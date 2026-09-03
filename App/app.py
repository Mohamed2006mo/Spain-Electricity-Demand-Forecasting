import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Load model and feature list
# -----------------------------
model = joblib.load('NoteBooks/random_forest_model.pkl')
feature_columns = joblib.load('NoteBooks/feature_columns.pkl')

st.set_page_config(page_title="Spain Electricity Load Predictor", layout="wide", page_icon="⚡")

# -----------------------------
# Global styling
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #101c26 0%, #0b1218 60%);
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricValue"] {
        color: #7fd8ff;
    }
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        height: 3em;
    }
    .hero {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 30px 34px; border-radius: 16px; margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <h1 style="color:white; margin-bottom:4px;">⚡ Spain Electricity Demand Predictor</h1>
        <p style="color:#cfe8ff; font-size:16px; margin-bottom:0;">
            A machine learning pipeline forecasting Spain's national hourly electricity load (MW) —
            trained on 4 years (2015–2018) of energy market and weather data across 5 cities.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

kcol1, kcol2, kcol3, kcol4 = st.columns(4)
kcol1.metric("Best Model", "Random Forest")
kcol2.metric("Test R²", "0.61")
kcol3.metric("Test MAPE", "7.63%")
kcol4.metric("Years of Data", "2015–2018")

st.write("")

# -----------------------------
# Sidebar navigation
# -----------------------------
with st.sidebar:
    st.markdown("### ⚡ Navigation")
    page = st.radio(
        "Go to",
        ["🔮 Predict", "📈 Model Insights", "ℹ️ About"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("Graduation Project — Optical Soft")
    st.caption("Spain Electricity Demand Forecasting")

# =========================================================
# PAGE — PREDICT
# =========================================================
if page == "🔮 Predict":
    st.subheader("🕐 When")
    tcol1, tcol2, tcol3 = st.columns(3)
    with tcol1:
        hour = st.slider("Hour of day", 0, 23, 12)
    with tcol2:
        month = st.select_slider(
            "Month",
            options=list(range(1, 13)),
            value=6,
            format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                    "Jul","Aug","Sep","Oct","Nov","Dec"][m-1]
        )
    with tcol3:
        day_of_week = st.selectbox(
            "Day of week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )

    is_weekend_flag = day_of_week in ["Saturday", "Sunday"]
    daypart = (
        "🌙 Night" if hour < 6 else
        "🌅 Morning" if hour < 12 else
        "☀️ Afternoon" if hour < 18 else
        "🌆 Evening"
    )
    st.caption(f"{daypart} · {'🏖️ Weekend' if is_weekend_flag else '💼 Weekday'}")

    st.divider()
    st.subheader("🌦️ Weather Conditions")

    col1, col2 = st.columns(2)
    with col1:
        temp_mean = st.slider("Average temperature (°C)", -15.0, 45.0, 20.0)
        temp_std = st.slider("Temperature spread across cities (°C)", 0.0, 15.0, 3.0)
        pressure_mean = st.slider("Average pressure (hPa)", 900.0, 1100.0, 1015.0)
        humidity_mean = st.slider("Average humidity (%)", 0.0, 100.0, 60.0)

    with col2:
        wind_speed_mean = st.slider("Average wind speed (m/s)", 0.0, 30.0, 3.0)
        wind_deg_mean = st.slider("Average wind direction (°)", 0.0, 360.0, 180.0)
        clouds_all_mean = st.slider("Average cloud cover (%)", 0.0, 100.0, 30.0)
        weather_main = st.selectbox(
            "Dominant weather condition",
            ["clear", "clouds", "rain", "drizzle", "fog", "haze", "mist", "thunderstorm"]
        )

    with st.expander("🌧️ Precipitation details (optional)"):
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            rain_1h_mean = st.number_input("Rain last hour (mm)", 0.0, 20.0, 0.0)
        with pcol2:
            rain_3h_mean = st.number_input("Rain last 3 hours (mm)", 0.0, 20.0, 0.0)
        with pcol3:
            snow_3h_mean = st.number_input("Snow last 3 hours (mm)", 0.0, 20.0, 0.0)

    with st.expander("📉 Advanced: Market & Generation Forecasts"):
        st.caption(
            "These are day-ahead figures a grid operator/analyst would already have on hand "
            "(published by Spain's market operators). Defaults are typical averages."
        )
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            forecast_solar = st.slider("Forecast solar generation (MW)", 0.0, 6000.0, 1000.0)
        with fcol2:
            forecast_wind = st.slider("Forecast wind onshore generation (MW)", 0.0, 17500.0, 5000.0)
        with fcol3:
            price_day_ahead = st.slider("Day-ahead price (€/MWh)", 0.0, 110.0, 55.0)

    # -----------------------------
    # Build the input row
    # -----------------------------
    def build_input_row():
        row = {col: 0 for col in feature_columns}

        row['forecast solar day ahead'] = forecast_solar
        row['forecast wind onshore day ahead'] = forecast_wind
        row['price day ahead'] = price_day_ahead
        row['month'] = month
        row['temp_mean'] = temp_mean
        row['temp_std'] = temp_std
        row['pressure_mean'] = pressure_mean
        row['humidity_mean'] = humidity_mean
        row['wind_speed_mean'] = wind_speed_mean
        row['wind_deg_mean'] = wind_deg_mean
        row['rain_1h_mean'] = rain_1h_mean
        row['rain_3h_mean'] = rain_3h_mean
        row['snow_3h_mean'] = snow_3h_mean
        row['clouds_all_mean'] = clouds_all_mean
        row['hour'] = hour
        row['is_weekend'] = 1 if is_weekend_flag else 0

        day_col = f'day_of_week_{day_of_week}'
        if day_col in row:
            row[day_col] = 1

        weather_col = f'weather_main_mode_{weather_main}'
        if weather_col in row:
            row[weather_col] = 1

        return pd.DataFrame([row])[feature_columns]

    st.divider()
    predict_clicked = st.button("🔮 Predict Demand", use_container_width=True, type="primary")

    if predict_clicked:
        input_df = build_input_row()
        prediction = model.predict(input_df)[0]

        st.session_state.setdefault("history", [])
        st.session_state["history"].append(prediction)

        rcol1, rcol2 = st.columns([1, 2])
        with rcol1:
            st.metric("Predicted Total Load", f"{prediction:,.0f} MW")
            if prediction < 24000:
                st.info("🌙 Low demand period — typical of weekend nights")
            elif prediction > 32000:
                st.warning("🔥 High demand period — typical of weekday peaks")
            else:
                st.success("✅ Moderate, typical demand level")

        with rcol2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': "Predicted Load (MW)"},
                number={'font': {'color': '#7fd8ff'}},
                gauge={
                    'axis': {'range': [15000, 42000]},
                    'bar': {'color': "#3fa9f5"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'steps': [
                        {'range': [15000, 24000], 'color': "#12293a"},
                        {'range': [24000, 32000], 'color': "#1c4a63"},
                        {'range': [32000, 42000], 'color': "#2874a6"},
                    ],
                }
            ))
            fig.update_layout(
                height=280, margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}
            )
            st.plotly_chart(fig, use_container_width=True)

        if len(st.session_state["history"]) > 1:
            st.subheader("📊 Your Predictions This Session")
            hist_df = pd.DataFrame({
                "Prediction #": range(1, len(st.session_state["history"]) + 1),
                "Predicted Load (MW)": st.session_state["history"]
            })
            st.line_chart(hist_df.set_index("Prediction #"))

# =========================================================
# PAGE — MODEL INSIGHTS
# =========================================================
elif page == "📈 Model Insights":
    st.header("Model Insights")

    st.subheader("Feature Importance")
    st.caption("Which inputs matter most to the model's predictions, based on Random Forest's built-in importance scores.")

    importances = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(15)

    fig_imp = px.bar(
        importances.sort_values("Importance"),
        x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues"
    )
    fig_imp.update_layout(
        height=500, showlegend=False, coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'color': 'white'}
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()

    st.subheader("Model Performance Summary")
    perf_df = pd.DataFrame({
        "Model": ["Random Forest", "Gradient Boosting", "XGBoost", "KNN"],
        "RMSE": [2320.06, 2338.75, 2392.11, 3330.26],
        "MAE": [1716.80, 1794.48, 1789.76, 2570.65],
        "R²": [0.7600, 0.7530, 0.7416, 0.4993],
        "MAPE (%)": [6.07, 6.36, 6.30, 8.94],
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    st.caption("Validation set results. Random Forest was selected as the final deployed model.")

    st.subheader("Final Test Set Performance (Random Forest)")
    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
    tcol1.metric("RMSE", "2861.61 MW")
    tcol2.metric("MAE", "2128.42 MW")
    tcol3.metric("R²", "0.6115")
    tcol4.metric("MAPE", "7.63%")

# =========================================================
# PAGE — ABOUT
# =========================================================
elif page == "ℹ️ About":
    st.header("About This Project")
    st.markdown("""
    This app is part of a graduation project predicting **Spain's national hourly electricity demand**
    using a full machine learning pipeline: data cleaning, feature engineering, model training, and evaluation.

    **Data sources:**
    - Energy data: hourly generation, load, and price data for Spain (2015-2018)
    - Weather data: hourly weather across 5 Spanish cities (Madrid, Barcelona, Valencia, Seville, Bilbao), aggregated nationally

    **Modeling approach:**
    - Leakage-free feature set: only information realistically available *in advance* (weather, day-ahead forecasts, time features) is used — no same-hour actual generation values or the incumbent day-ahead load forecast, to ensure a genuine forecasting task rather than one that mirrors an existing prediction.
    - Four ML algorithms compared: Random Forest, XGBoost, Gradient Boosting, and K-Nearest Neighbors.
    - Chronological train/validation/test split (70/15/15), respecting the time-series nature of the data.

    **Deployment:** This Streamlit app loads the final trained Random Forest model to generate live predictions from user-specified conditions.
    """)
