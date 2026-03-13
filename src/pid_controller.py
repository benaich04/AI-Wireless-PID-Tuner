import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plant import simulate_plant


class PIDController:
    """
    Discrete PID Controller.

    Control law:
        e[k]  = r - y[k]                      (error)
        I[k]  = I[k-1] + e[k] * dt            (integral)
        D[k]  = (e[k] - e[k-1]) / dt          (derivative)
        u[k]  = Kp*e[k] + Ki*I[k] + Kd*D[k]  (control signal)
    """

    def __init__(self, Kp, Ki, Kd, dt=0.01,
                 u_min=-10.0, u_max=10.0,
                 integral_limit=50.0):
        """
        Args:
            Kp             : proportional gain
            Ki             : integral gain
            Kd             : derivative gain
            dt             : time step (must match simulation dt)
            u_min / u_max  : actuator saturation limits (clips control signal)
            integral_limit : anti-windup clamp on the integral term
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        self.integral_limit = integral_limit

        # Internal state — reset between simulations
        self._integral  = 0.0
        self._prev_error = 0.0

    def reset(self):
        """Reset internal state before each new simulation run."""
        self._integral   = 0.0
        self._prev_error = 0.0

    def compute(self, reference, measurement):
        """
        Compute one control step.

        Args:
            reference   : desired setpoint r
            measurement : current plant output y

        Returns:
            u           : control signal (clipped to actuator limits)
        """
        error = reference - measurement

        # Proportional
        P = self.Kp * error

        # Integral with anti-windup clamp
        self._integral += error * self.dt
        self._integral  = np.clip(self._integral,
                                  -self.integral_limit,
                                   self.integral_limit)
        I = self.Ki * self._integral

        # Derivative
        D = self.Kd * (error - self._prev_error) / self.dt
        self._prev_error = error

        # Total control signal — clipped to actuator limits
        u = np.clip(P + I + D, self.u_min, self.u_max)
        return u


def simulate_pid(Kp, Ki, Kd,
                 reference=1.0,
                 t_span=(0, 5),
                 dt=0.01,
                 omega0=0.0):
    """
    Run a full closed-loop simulation: PID controlling the DC motor.

    Args:
        Kp, Ki, Kd  : PID gains
        reference   : target motor speed
        t_span      : (start, end) in seconds
        dt          : time step
        omega0      : initial motor speed

    Returns:
        t           : time array
        omega       : motor speed array
        u_history   : control signal array
    """
    pid = PIDController(Kp, Ki, Kd, dt=dt)
    pid.reset()

    t_eval   = np.arange(t_span[0], t_span[1], dt)
    omega    = np.zeros(len(t_eval))
    u_history = np.zeros(len(t_eval))
    omega[0] = omega0

    for i in range(1, len(t_eval)):
        # PID computes control signal from current error
        u = pid.compute(reference, omega[i - 1])
        u_history[i] = u

        # Plant responds to control signal
        domega = (2.0 * u - omega[i - 1]) / 0.5   # K=2.0, TAU=0.5
        omega[i] = omega[i - 1] + domega * dt

    return t_eval, omega, u_history


def plot_pid_response(t, omega, u_history, Kp, Ki, Kd, reference=1.0):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # Speed response
    ax1.plot(t, omega, color="#1D9E75", linewidth=2, label="Motor speed ω(t)")
    ax1.axhline(reference, color="#D85A30", linestyle="--",
                linewidth=1.5, label=f"Reference = {reference}")
    ax1.set_ylabel("Speed ω (rad/s)")
    ax1.set_title(f"PID Closed-Loop Response   Kp={Kp}  Ki={Ki}  Kd={Kd}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Control effort
    ax2.plot(t, u_history, color="#7F77DD", linewidth=2, label="Control signal u(t)")
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Voltage u (V)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/plots/pid_response.png", dpi=150)
    plt.show()
    print("Plot saved to results/plots/pid_response.png")


def print_metrics(t, omega, reference=1.0):
    """Print basic performance numbers."""
    error      = np.abs(reference - omega)
    overshoot  = max(0, (np.max(omega) - reference) / reference * 100)
    ss_error   = abs(reference - omega[-1])

    # Settling time: first time speed stays within 2% of reference
    band  = 0.02 * reference
    settled = np.where(np.all(
        np.abs(omega[i:] - reference) < band
        for i in range(len(omega))
        if True
    ))[0]
    within_band = np.abs(omega - reference) < band
    settling_idx = None
    for i in range(len(within_band) - 1, -1, -1):
        if not within_band[i]:
            settling_idx = i + 1
            break
    settling_time = t[settling_idx] if settling_idx and settling_idx < len(t) else t[-1]

    print(f"  Overshoot     : {overshoot:.2f}%")
    print(f"  Settling time : {settling_time:.3f} s")
    print(f"  Steady-state error : {ss_error:.4f}")
    print(f"  IAE (total error)  : {np.sum(error) * (t[1]-t[0]):.4f}")


# Quick test — run this file directly
if __name__ == "__main__":
    # Try these gains — feel free to change them and see what happens
    Kp, Ki, Kd = 2.5, 1.0, 0.05
    REFERENCE  = 1.0

    print(f"Running PID simulation with Kp={Kp}, Ki={Ki}, Kd={Kd}")
    t, omega, u_history = simulate_pid(Kp, Ki, Kd, reference=REFERENCE)

    print_metrics(t, omega, reference=REFERENCE)
    plot_pid_response(t, omega, u_history, Kp, Ki, Kd, reference=REFERENCE)