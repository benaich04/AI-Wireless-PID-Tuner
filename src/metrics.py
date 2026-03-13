import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.wireless_channel import simulate_with_wireless


def compute_metrics(t, omega, reference=1.0, u_history=None):
    """
    Compute all performance metrics from a simulation response.

    Args:
        t         : time array
        omega     : motor speed array
        reference : target setpoint
        u_history : control signal array (optional, for effort metric)

    Returns:
        dict of all metrics
    """
    dt    = t[1] - t[0]
    error = reference - omega

    # --- IAE: Integral of Absolute Error ---
    # Total accumulated error over the whole run
    # Lower = better tracking
    iae = np.sum(np.abs(error)) * dt

    # --- Overshoot ---
    # How far above the reference the motor went (%)
    # 0% is ideal — negative means it never reached reference
    max_speed = np.max(omega)
    overshoot = max(0.0, (max_speed - reference) / reference * 100.0)

    # --- Settling time ---
    # First time the speed enters and STAYS within 2% of reference
    band    = 0.02 * reference
    settled = np.abs(omega - reference) < band
    settling_time = t[-1]   # default: never settled
    for i in range(len(settled) - 1, -1, -1):
        if not settled[i]:
            if i + 1 < len(t):
                settling_time = t[i + 1]
            break
    else:
        settling_time = t[0]

    # --- Steady-state error ---
    # Average error in the last 10% of simulation time
    tail      = int(0.1 * len(omega))
    ss_error  = np.abs(np.mean(reference - omega[-tail:]))

    # --- Control effort ---
    # Total energy spent by the controller
    # High effort = aggressive, expensive, stressful on actuator
    if u_history is not None:
        control_effort = np.sum(np.abs(u_history)) * dt
    else:
        control_effort = None

    # --- Instability flag ---
    # If speed diverges beyond 3x reference, mark as unstable
    unstable = bool(np.any(np.abs(omega) > 3.0 * reference))

    return {
        "iae"            : round(iae, 4),
        "overshoot_pct"  : round(overshoot, 4),
        "settling_time"  : round(settling_time, 4),
        "ss_error"       : round(ss_error, 6),
        "control_effort" : round(control_effort, 4) if control_effort is not None else None,
        "unstable"       : unstable,
    }


def compute_cost(metrics,
                 w_iae=1.0,
                 w_overshoot=0.5,
                 w_settling=0.3,
                 w_effort=0.1):
    """
    Combine metrics into a single cost score.
    Lower cost = better PID performance.

    If the system is unstable, return a large penalty instead.

    Args:
        metrics     : dict from compute_metrics()
        w_*         : weights for each term

    Returns:
        cost        : scalar score
    """
    if metrics["unstable"]:
        return 99999.0

    cost = (
        w_iae       * metrics["iae"] +
        w_overshoot * metrics["overshoot_pct"] +
        w_settling  * metrics["settling_time"] +
        w_effort    * (metrics["control_effort"] or 0.0)
    )
    return round(cost, 4)


def print_metrics(metrics, cost=None, label=""):
    """Pretty-print metrics to terminal."""
    tag = f"[{label}]" if label else ""
    print(f"\n{tag} Performance Metrics")
    print(f"  IAE (tracking error)  : {metrics['iae']}")
    print(f"  Overshoot             : {metrics['overshoot_pct']:.2f}%")
    print(f"  Settling time         : {metrics['settling_time']:.3f} s")
    print(f"  Steady-state error    : {metrics['ss_error']:.6f}")
    print(f"  Control effort        : {metrics['control_effort']}")
    print(f"  Unstable              : {metrics['unstable']}")
    if cost is not None:
        print(f"  ► TOTAL COST          : {cost}")


def plot_metrics_comparison():
    """
    Show metrics side by side for ideal vs degraded wireless conditions.
    """
    Kp, Ki, Kd = 2.5, 1.0, 0.05

    scenarios = [
        {"label": "Ideal",        "delay_ms": 0,   "packet_loss": 0.0,  "noise_std": 0.0 },
        {"label": "Delay 100ms",  "delay_ms": 100, "packet_loss": 0.0,  "noise_std": 0.0 },
        {"label": "Loss 20%",     "delay_ms": 0,   "packet_loss": 0.20, "noise_std": 0.0 },
        {"label": "All combined", "delay_ms": 100, "packet_loss": 0.15, "noise_std": 0.03},
    ]

    results = []
    for s in scenarios:
        t, omega, u, _ = simulate_with_wireless(
            Kp, Ki, Kd,
            delay_ms    = s["delay_ms"],
            packet_loss = s["packet_loss"],
            noise_std   = s["noise_std"],
        )
        m    = compute_metrics(t, omega, u_history=u)
        cost = compute_cost(m)
        results.append({"label": s["label"], "metrics": m, "cost": cost})
        print_metrics(m, cost=cost, label=s["label"])

    # Bar chart comparison
    labels   = [r["label"]              for r in results]
    iae      = [r["metrics"]["iae"]     for r in results]
    overshoot= [r["metrics"]["overshoot_pct"] for r in results]
    settling = [r["metrics"]["settling_time"] for r in results]
    costs    = [r["cost"]               for r in results]

    x    = np.arange(len(labels))
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    colors = ["#1D9E75", "#7F77DD", "#D85A30", "#BA7517"]

    for ax, values, title, unit in zip(
        axes,
        [iae, overshoot, settling, costs],
        ["IAE", "Overshoot (%)", "Settling time (s)", "Total cost"],
        ["", "%", "s", ""]
    ):
        bars = ax.bar(labels, values, color=colors, edgecolor="none", width=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(unit)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01 * max(values),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Performance Metrics Across Wireless Conditions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/plots/metrics_comparison.png", dpi=150)
    plt.show()
    print("\nPlot saved to results/plots/metrics_comparison.png")


if __name__ == "__main__":
    plot_metrics_comparison()