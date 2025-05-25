import streamlit as st
from prophet.serialize import model_from_json
import pandas as pd
import json
import matplotlib.pyplot as plt

st.title("📊 Prophet Modellvergleich – First Exceed Check")

uploaded_models = st.file_uploader("📥 Mehrere Prophet-Modelle auswählen", type="json", accept_multiple_files=True)
threshold = st.number_input("⚠️ Schwellenwert (Mbit/s)", value=5000)
days = st.slider("🔮 Vorhersagezeitraum (in Tagen)", 1, 90, 30)
periods = days * 24

results = []

if uploaded_models:
    st.write(f"Analysiere {len(uploaded_models)} Modelle…")

    for file in uploaded_models:
        model = model_from_json(json.load(file))
        future = model.make_future_dataframe(periods=periods, freq='H')
        forecast = model.predict(future)

        # First exceed
        exceeds = forecast[forecast['yhat'] > threshold]
        first_exceed = exceeds['ds'].iloc[0] if not exceeds.empty else None

        results.append({
            "Modellname": file.name,
            "First Exceed": first_exceed,
        })

        # Optional: Plot anzeigen
        with st.expander(f"📈 Plot für {file.name}"):
            fig = model.plot(forecast)
            st.pyplot(fig)

    # Ergebnisse anzeigen
    df_results = pd.DataFrame(results).sort_values(by="First Exceed", na_position="last")
    st.subheader("🕓 Übersicht: Erste Schwellenüberschreitung pro Modell")
    st.dataframe(df_results)
