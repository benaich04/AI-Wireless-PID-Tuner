import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# DC Motor parameters (first-order model: τ·dω/dt + ω = K·u)
TAU = 0.5   # time constant (seconds) — how fast the motor responds
K   = 2.0   # motor gain — how much speed you get per unit voltage


def motor_ode(t, omega, u, tau=TAU, k=K):
    """
    ODE for DC motor speed:
        dω/dt = (K·u - ω) / τ
    """
    return (k * u - omega) / tau


def simulate_plant(u_func, t_span=(0, 5), dt=0.01, omega0=0.0):
    """
    Simulate the DC motor over time.

    Args:
        u_func  : callable u(t) — control input as a function of time
        t_span  : (start, end) in seconds
        dt      : time step in seconds
        omega0  : initial motor speed

    Returns:
        t       : time array
        omega   : motor speed array
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)
    omega  = np.zeros(len(t_eval))
    omega[0] = omega0

    for i in range(1, len(t_eval)):
        t_now = t_eval[i - 1]
        u_now = u_func(t_now)

        # Simple Euler integration
        domega = motor_ode(t_now, omega[i - 1], u_now)
        omega[i] = omega[i - 1] + domega * dt

    return t_eval, omega


def plot_response(t, omega, title="DC Motor Step Response", reference=1.0):
    plt.figure(figsize=(8, 4))
    plt.plot(t, omega, label="Motor speed ω(t)", color="#1D9E75", linewidth=2)
    plt.axhline(reference, color="#D85A30", linestyle="--", linewidth=1.5, label=f"Reference = {reference}")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed ω (rad/s)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/plots/plant_step_response.png", dpi=150)
    plt.show()
    print("Plot saved to results/plots/plant_step_response.png")


# Quick test — run this file directly to verify the plant works
if __name__ == "__main__":
    REFERENCE = 1.0

    # Step input: constant voltage u = 1.0 for all time
    def step_input(t):
        return REFERENCE

    t, omega = simulate_plant(step_input, t_span=(0, 5), dt=0.01)
    plot_response(t, omega, title="DC Motor Open-Loop Step Response", reference=REFERENCE)

    print(f"Final speed : {omega[-1]:.4f} rad/s")
    print(f"Expected    : {K * REFERENCE:.4f} rad/s  (K × u = {K} × {REFERENCE})")