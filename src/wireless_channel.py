import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pid_controller import PIDController


class WirelessChannel:
    """
    Simulates a wireless communication channel with three impairments:

        1. Delay      — measurement arrives N steps late
        2. Packet loss — measurement randomly dropped with probability p
        3. Noise      — Gaussian noise added to measurement
    """

    def __init__(self, delay_ms=0, packet_loss=0.0, noise_std=0.0, dt=0.01):
        """
        Args:
            delay_ms    : one-way delay in milliseconds
            packet_loss : probability of dropping a packet (0.0 to 1.0)
            noise_std   : standard deviation of Gaussian noise
            dt          : simulation time step (seconds)
        """
        self.delay_steps  = int((delay_ms / 1000.0) / dt)   # convert ms → steps
        self.packet_loss  = packet_loss
        self.noise_std    = noise_std
        self.dt           = dt

        # Buffer stores past measurements to simulate delay
        self._buffer      = deque(maxlen=max(self.delay_steps + 1, 1))
        self._last_value  = 0.0   # last successfully received value

        # Fill buffer with zeros at startup
        for _ in range(max(self.delay_steps, 1)):
            self._buffer.append(0.0)

    def reset(self, init_value=0.0):
        """Reset channel state before each simulation run."""
        self._buffer    = deque(maxlen=max(self.delay_steps + 1, 1))
        self._last_value = init_value
        for _ in range(max(self.delay_steps, 1)):
            self._buffer.append(init_value)

    def transmit(self, true_measurement):
        """
        Pass one measurement through the wireless channel.

        Steps:
            1. Push true value into delay buffer
            2. Pull delayed value out
            3. Apply packet loss (randomly drop)
            4. Add Gaussian noise

        Args:
            true_measurement : clean sensor reading from plant

        Returns:
            received : degraded measurement seen by controller
        """
        # Step 1 — push into buffer
        self._buffer.append(true_measurement)

        # Step 2 — pull delayed value (oldest in buffer)
        delayed = self._buffer[0]

        # Step 3 — packet loss: drop with probability p, use last value
        if np.random.rand() < self.packet_loss:
            received = self._last_value   # hold last received sample
        else:
            received = delayed
            self._last_value = delayed

        # Step 4 — add Gaussian noise
        if self.noise_std > 0:
            received += np.random.normal(0, self.noise_std)

        return received


def simulate_with_wireless(Kp, Ki, Kd,
                            delay_ms=0,
                            packet_loss=0.0,
                            noise_std=0.0,
                            reference=1.0,
                            t_span=(0, 5),
                            dt=0.01,
                            omega0=0.0):
    """
    Full closed-loop simulation with wireless channel on measurement path.

    Returns:
        t         : time array
        omega     : motor speed array
        u_history : control signal array
        y_received: degraded measurement seen by controller
    """
    channel = WirelessChannel(delay_ms, packet_loss, noise_std, dt)
    pid     = PIDController(Kp, Ki, Kd, dt=dt)

    channel.reset(init_value=omega0)
    pid.reset()

    t_eval     = np.arange(t_span[0], t_span[1], dt)
    omega      = np.zeros(len(t_eval))
    u_history  = np.zeros(len(t_eval))
    y_received = np.zeros(len(t_eval))
    omega[0]   = omega0

    for i in range(1, len(t_eval)):
        # True plant output goes through wireless channel
        y_noisy = channel.transmit(omega[i - 1])
        y_received[i] = y_noisy

        # PID sees degraded measurement — NOT the true value
        u = pid.compute(reference, y_noisy)
        u_history[i] = u

        # Plant responds to control signal
        domega   = (2.0 * u - omega[i - 1]) / 0.5
        omega[i] = omega[i - 1] + domega * dt

        # Clip to prevent simulation explosion
        omega[i] = np.clip(omega[i], -10, 10)

    return t_eval, omega, u_history, y_received


def plot_wireless_comparison(Kp=2.5, Ki=1.0, Kd=0.05, reference=1.0):
    """
    Plot four scenarios side by side:
        1. Ideal (no wireless effects)
        2. Delay only
        3. Packet loss only
        4. All effects combined
    """
    scenarios = [
        {"label": "Ideal",           "delay_ms": 0,   "packet_loss": 0.0,  "noise_std": 0.0,  "color": "#1D9E75"},
        {"label": "Delay 100ms",     "delay_ms": 100, "packet_loss": 0.0,  "noise_std": 0.0,  "color": "#7F77DD"},
        {"label": "Packet loss 20%", "delay_ms": 0,   "packet_loss": 0.20, "noise_std": 0.0,  "color": "#D85A30"},
        {"label": "All combined",    "delay_ms": 100, "packet_loss": 0.15, "noise_std": 0.03, "color": "#BA7517"},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, s in zip(axes, scenarios):
        t, omega, u, y_rx = simulate_with_wireless(
            Kp, Ki, Kd,
            delay_ms    = s["delay_ms"],
            packet_loss = s["packet_loss"],
            noise_std   = s["noise_std"],
            reference   = reference
        )
        ax.plot(t, omega,  color=s["color"],  linewidth=2,   label="Motor speed")
        ax.plot(t, y_rx,   color="gray",       linewidth=0.8, alpha=0.5, label="Received signal")
        ax.axhline(reference, color="#D85A30", linewidth=1.2, linestyle="--", label="Reference")
        ax.set_title(s["label"], fontsize=12)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed ω (rad/s)")
        ax.set_ylim(-0.2, 1.6)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Wireless Channel Effects on PID Control   Kp={Kp}  Ki={Ki}  Kd={Kd}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/plots/wireless_comparison.png", dpi=150)
    plt.show()
    print("Plot saved to results/plots/wireless_comparison.png")


# Quick test
if __name__ == "__main__":
    print("Testing wireless channel — generating 4-panel comparison plot...\n")

    # Test single channel transmit
    ch = WirelessChannel(delay_ms=50, packet_loss=0.1, noise_std=0.02)
    ch.reset()
    test_val = ch.transmit(1.0)
    print(f"Single transmit test  →  input: 1.0,  received: {test_val:.4f}")
    print(f"Delay steps           →  {ch.delay_steps} steps  (50ms at dt=0.01s)\n")

    # Full comparison plot
    plot_wireless_comparison(Kp=2.5, Ki=1.0, Kd=0.05)