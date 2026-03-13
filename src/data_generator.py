import numpy as np
import pandas as pd
import itertools
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.wireless_channel import simulate_with_wireless
from src.metrics import compute_metrics, compute_cost


# ── Wireless condition search space ───────────────────────────────────────────
DELAY_MS_VALUES    = [0, 20, 50, 80, 100, 150, 200]        # ms
PACKET_LOSS_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]  # probability
NOISE_STD_VALUES   = [0.0, 0.01, 0.02, 0.03, 0.05]        # std deviation

# ── PID gain search space ──────────────────────────────────────────────────────
KP_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
KI_VALUES = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
KD_VALUES = [0.0, 0.01, 0.02, 0.05, 0.08, 0.1]

# ── Simulation settings ────────────────────────────────────────────────────────
REFERENCE  = 1.0
T_SPAN     = (0, 5)
DT         = 0.01
N_PID_RANDOM_SAMPLES = 80   # how many random PID combos to try per scenario


def evaluate_pid(Kp, Ki, Kd, delay_ms, packet_loss, noise_std):
    """
    Run one simulation and return the cost score.
    Returns large penalty if simulation explodes.
    """
    try:
        t, omega, u, _ = simulate_with_wireless(
            Kp, Ki, Kd,
            delay_ms    = delay_ms,
            packet_loss = packet_loss,
            noise_std   = noise_std,
            reference   = REFERENCE,
            t_span      = T_SPAN,
            dt          = DT,
        )
        metrics = compute_metrics(t, omega, reference=REFERENCE, u_history=u)
        cost    = compute_cost(metrics)
        return cost, metrics

    except Exception:
        return 99999.0, None


def find_best_pid(delay_ms, packet_loss, noise_std, n_samples=N_PID_RANDOM_SAMPLES):
    """
    For a given wireless condition, try many PID combinations
    and return the one with the lowest cost.

    Strategy: random search over the PID gain space.
    """
    best_cost    = np.inf
    best_gains   = (1.0, 0.5, 0.01)
    best_metrics = None

    for _ in range(n_samples):
        Kp = np.random.choice(KP_VALUES)
        Ki = np.random.choice(KI_VALUES)
        Kd = np.random.choice(KD_VALUES)

        cost, metrics = evaluate_pid(Kp, Ki, Kd, delay_ms, packet_loss, noise_std)

        if cost < best_cost:
            best_cost    = cost
            best_gains   = (Kp, Ki, Kd)
            best_metrics = metrics

    return best_gains, best_cost, best_metrics


def generate_dataset(output_path="data/training_data.csv", verbose=True):
    """
    Main dataset generation loop.

    For every wireless condition combination:
        1. Try N_PID_RANDOM_SAMPLES PID gain combinations
        2. Keep the one with the lowest cost
        3. Save as one row in the dataset

    Saves CSV to output_path.
    """
    # All wireless condition combinations
    all_scenarios = list(itertools.product(
        DELAY_MS_VALUES,
        PACKET_LOSS_VALUES,
        NOISE_STD_VALUES
    ))
    total = len(all_scenarios)

    if verbose:
        print(f"Generating dataset...")
        print(f"  Wireless scenarios : {total}")
        print(f"  PID samples/scenario: {N_PID_RANDOM_SAMPLES}")
        print(f"  Total simulations  : {total * N_PID_RANDOM_SAMPLES}")
        print(f"  Output             : {output_path}\n")

    rows = []

    for idx, (delay_ms, packet_loss, noise_std) in enumerate(all_scenarios):

        if verbose and idx % 20 == 0:
            print(f"  [{idx+1}/{total}]  delay={delay_ms}ms  "
                  f"loss={packet_loss:.0%}  noise={noise_std}")

        best_gains, best_cost, best_metrics = find_best_pid(
            delay_ms, packet_loss, noise_std
        )
        Kp, Ki, Kd = best_gains

        row = {
            # Inputs (wireless conditions)
            "delay_ms"      : delay_ms,
            "packet_loss"   : packet_loss,
            "noise_std"     : noise_std,

            # Outputs (best PID gains found)
            "best_kp"       : Kp,
            "best_ki"       : Ki,
            "best_kd"       : Kd,

            # Extra info (not used for training, useful for analysis)
            "best_cost"     : best_cost,
            "iae"           : best_metrics["iae"]           if best_metrics else None,
            "overshoot_pct" : best_metrics["overshoot_pct"] if best_metrics else None,
            "settling_time" : best_metrics["settling_time"] if best_metrics else None,
        }
        rows.append(row)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"\nDone. Dataset saved → {output_path}")
        print(f"  Rows    : {len(df)}")
        print(f"  Columns : {list(df.columns)}")
        print(f"\nSample rows:")
        print(df.head(5).to_string(index=False))
        print(f"\nBest cost range  : {df['best_cost'].min():.2f} → {df['best_cost'].max():.2f}")
        print(f"Kp range         : {df['best_kp'].min()} → {df['best_kp'].max()}")
        print(f"Ki range         : {df['best_ki'].min()} → {df['best_ki'].max()}")
        print(f"Kd range         : {df['best_kd'].min()} → {df['best_kd'].max()}")

    return df


if __name__ == "__main__":
    np.random.seed(42)   # reproducible results
    df = generate_dataset(output_path="data/training_data.csv")