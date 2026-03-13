import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.wireless_channel import simulate_with_wireless
from src.metrics import compute_metrics, compute_cost

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Wireless PID Tuner",
    page_icon="📡",
    layout="wide"
)

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        kp = joblib.load("models/kp_model.pkl")
        ki = joblib.load("models/ki_model.pkl")
        kd = joblib.load("models/kd_model.pkl")
        return kp, ki, kd, True
    except Exception as e:
        return None, None, None, False


def predict_gains(model_kp, model_ki, model_kd, delay_ms, packet_loss, noise_std):
    """Use trained models to predict best PID gains."""
    X = np.array([[delay_ms, packet_loss, noise_std]])
    Kp = float(np.clip(model_kp.predict(X)[0], 0.1, 5.0))
    Ki = float(np.clip(model_ki.predict(X)[0], 0.05, 3.0))
    Kd = float(np.clip(model_kd.predict(X)[0], 0.0,  0.2))
    return round(Kp, 3), round(Ki, 3), round(Kd, 3)


def run_simulation(Kp, Ki, Kd, delay_ms, packet_loss, noise_std):
    """Run one full simulation and return results."""
    t, omega, u, y_rx = simulate_with_wireless(
        Kp, Ki, Kd,
        delay_ms    = delay_ms,
        packet_loss = packet_loss,
        noise_std   = noise_std,
        reference   = 1.0,
        t_span      = (0, 5),
        dt          = 0.01,
    )
    metrics = compute_metrics(t, omega, reference=1.0, u_history=u)
    cost    = compute_cost(metrics)
    return t, omega, u, y_rx, metrics, cost


def make_comparison_plot(t, omega_manual, omega_ai, u_manual, u_ai, reference=1.0):
    """Plot manual PID vs AI PID side by side."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Speed response
    ax1.plot(t, omega_manual, color="#D85A30", linewidth=2,
             label="Manual PID", linestyle="--")
    ax1.plot(t, omega_ai,     color="#1D9E75", linewidth=2,
             label="AI-tuned PID")
    ax1.axhline(reference, color="gray", linewidth=1.2,
                linestyle=":", label="Reference")
    ax1.set_ylabel("Motor speed ω (rad/s)", fontsize=11)
    ax1.set_title("Motor Speed Response — Manual vs AI-Tuned PID", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.7)

    # Control effort
    ax2.plot(t, u_manual, color="#D85A30", linewidth=1.5,
             label="Manual PID effort", linestyle="--", alpha=0.8)
    ax2.plot(t, u_ai,     color="#1D9E75", linewidth=1.5,
             label="AI-tuned PID effort")
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("Control signal u (V)", fontsize=11)
    ax2.set_title("Control Effort", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.title("📡 AI-Driven PID Tuner for Wireless Control Systems")
st.markdown(
    "Adjust the **wireless channel conditions** below. "
    "The AI will predict optimal PID gains and compare them against fixed manual gains."
)
st.divider()

# ── Load models ────────────────────────────────────────────────────────────────
model_kp, model_ki, model_kd, models_loaded = load_models()

if not models_loaded:
    st.error("⚠️ Models not found. Run `python src/train_model.py` first.")
    st.stop()

# ── Sidebar — wireless conditions ─────────────────────────────────────────────
with st.sidebar:
    st.header("📶 Wireless Channel Settings")

    delay_ms = st.slider(
        "Delay (ms)", min_value=0, max_value=200, value=50, step=10,
        help="How many milliseconds the sensor measurement is delayed"
    )
    packet_loss = st.slider(
        "Packet loss (%)", min_value=0, max_value=25, value=10, step=1,
        help="Percentage of sensor packets randomly dropped"
    ) / 100.0

    noise_std = st.slider(
        "Measurement noise (σ)", min_value=0.0, max_value=0.05,
        value=0.02, step=0.005,
        help="Standard deviation of Gaussian noise on sensor reading"
    )

    st.divider()
    st.header("🔧 Manual PID Gains")
    manual_kp = st.number_input("Manual Kp", value=2.5, min_value=0.1, max_value=5.0, step=0.1)
    manual_ki = st.number_input("Manual Ki", value=1.0, min_value=0.0, max_value=3.0, step=0.1)
    manual_kd = st.number_input("Manual Kd", value=0.05, min_value=0.0, max_value=0.2, step=0.01)

    st.divider()
    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)


# ── Main area ──────────────────────────────────────────────────────────────────

# Always show AI predicted gains
ai_kp, ai_ki, ai_kd = predict_gains(
    model_kp, model_ki, model_kd,
    delay_ms, packet_loss, noise_std
)

# Channel condition summary
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Delay",       f"{delay_ms} ms")
with col2:
    st.metric("Packet loss", f"{packet_loss*100:.0f}%")
with col3:
    st.metric("Noise σ",     f"{noise_std:.3f}")

st.divider()

# Gains comparison
st.subheader("🤖 AI-Predicted vs Manual Gains")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Manual Kp", manual_kp)
with col2:
    st.metric("Manual Ki", manual_ki)
with col3:
    st.metric("Manual Kd", manual_kd)
with col4:
    delta_kp = round(ai_kp - manual_kp, 3)
    st.metric("AI Kp", ai_kp, delta=delta_kp)
with col5:
    delta_ki = round(ai_ki - manual_ki, 3)
    st.metric("AI Ki", ai_ki, delta=delta_ki)
with col6:
    delta_kd = round(ai_kd - manual_kd, 3)
    st.metric("AI Kd", ai_kd, delta=delta_kd)

st.divider()

# ── Run simulation on button click ─────────────────────────────────────────────
if run_btn:
    with st.spinner("Running simulations..."):

        # Manual PID simulation
        t, omega_m, u_m, _, metrics_m, cost_m = run_simulation(
            manual_kp, manual_ki, manual_kd,
            delay_ms, packet_loss, noise_std
        )

        # AI PID simulation
        _, omega_ai, u_ai, _, metrics_ai, cost_ai = run_simulation(
            ai_kp, ai_ki, ai_kd,
            delay_ms, packet_loss, noise_std
        )

    # ── Response plot ──────────────────────────────────────────────────────────
    st.subheader("📊 Simulation Results")
    fig = make_comparison_plot(t, omega_m, omega_ai, u_m, u_ai)
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── Metrics comparison table ───────────────────────────────────────────────
    st.subheader("📋 Performance Metrics")

    improvement = round((cost_m - cost_ai) / cost_m * 100, 1) if cost_m > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Manual PID**")
        st.dataframe(pd.DataFrame({
            "Metric": ["IAE", "Overshoot (%)", "Settling time (s)", "SS Error", "Total cost"],
            "Value":  [
                metrics_m["iae"],
                f"{metrics_m['overshoot_pct']:.2f}%",
                f"{metrics_m['settling_time']:.3f}s",
                f"{metrics_m['ss_error']:.4f}",
                cost_m
            ]
        }), hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**AI-Tuned PID**")
        st.dataframe(pd.DataFrame({
            "Metric": ["IAE", "Overshoot (%)", "Settling time (s)", "SS Error", "Total cost"],
            "Value":  [
                metrics_ai["iae"],
                f"{metrics_ai['overshoot_pct']:.2f}%",
                f"{metrics_ai['settling_time']:.3f}s",
                f"{metrics_ai['ss_error']:.4f}",
                cost_ai
            ]
        }), hide_index=True, use_container_width=True)

    with col3:
        st.markdown("**Summary**")
        if improvement > 0:
            st.success(f"✅ AI improved cost by **{improvement}%**")
        elif improvement < 0:
            st.warning(f"⚠️ Manual gains performed better by **{abs(improvement)}%**")
        else:
            st.info("Both performed equally")

        st.metric("Manual cost", cost_m)
        st.metric("AI cost",     cost_ai,
                  delta=f"{improvement}% better" if improvement > 0 else f"{abs(improvement)}% worse",
                  delta_color="normal" if improvement > 0 else "inverse")

    # ── Instability warning ────────────────────────────────────────────────────
    if metrics_m["unstable"]:
        st.error("🚨 Manual PID went unstable under these wireless conditions!")
    if metrics_ai["unstable"]:
        st.warning("⚠️ AI-tuned PID also went unstable — try regenerating the dataset.")

else:
    st.info("👈 Adjust the wireless settings in the sidebar and click **Run Simulation**")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("AI-Driven PID Tuning for Wireless Networked Control Systems — "
           "Plant: DC Motor | Controller: Discrete PID | AI: Random Forest Regressor")