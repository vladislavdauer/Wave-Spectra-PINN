from __future__ import annotations

import copy
import random
from argparse import ArgumentParser
from csv import DictWriter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from scipy.optimize import brentq

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PhysicsConfig:
    a: float = 1.0
    b: float = 1.0
    mu: float = 10.0
    density: float = 1.0
    thickness: float = 1.0
    stiffness: float = 1.0
    lattice_terms: int = 8
    eps: float = 1e-10


@dataclass
class TrainConfig:
    bands: int = 1
    points_per_segment: int = 100
    kappa_min: float = 0.05
    kappa_max: float = 10.0
    root_scan_points: int = 4000
    hidden_dim: int = 128
    hidden_layers: int = 5
    epochs: int = 10000
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    data_weight: float = 1.0
    physics_weight_start: float = 1e-4
    physics_weight_end: float = 1e-2
    smooth_weight: float = 1e-4
    order_weight: float = 1e-3
    min_band_distance: float = 2e-3
    residual_clip: float = 30.0
    residual_scale: float = 10.0
    lbfgs_steps: int = 30
    lbfgs_iterations: int = 20
    lbfgs_lr: float = 0.5
    log_every: int = 100
    min_band_coverage: float = 0.55


class WaveSpectraPINN(nn.Module):
    def __init__(
            self,
            k_mean: torch.Tensor,
            k_std: torch.Tensor,
            kappa_low: float,
            kappa_high: float,
            config: TrainConfig,
    ) -> None:
        super().__init__()

        self.kappa_low = float(kappa_low)
        self.kappa_high = float(kappa_high)

        self.register_buffer("k_mean", k_mean)
        self.register_buffer("k_std", k_std)

        layers: list[nn.Module] = []
        input_dim = 2

        for layer_number in range(config.hidden_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            layers.append(nn.Tanh())

            input_dim = config.hidden_dim

        layers.append(nn.Linear(config.hidden_dim, config.bands))

        self.network = nn.Sequential(*layers)

    def forward(self, k_points: torch.Tensor) -> torch.Tensor:
        normalized = (k_points - self.k_mean) / self.k_std
        raw = self.network.forward(normalized)
        scaled = torch.sigmoid(raw)

        kappa = self.kappa_low + (self.kappa_high - self.kappa_low) * scaled
        kappa, _ = torch.sort(kappa, dim=1)

        return kappa


def make_k_path(
        points_per_segment: int,
        physics: PhysicsConfig,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    gamma = np.array([0.0, 0.0])

    x_point = np.array([np.pi / physics.a, 0.0])
    m_point = np.array([np.pi / physics.a, np.pi / physics.b])

    corners = [gamma, x_point, m_point, gamma]

    distances = []
    k_points = []
    ticks = [0.0]
    current_distance = 0.0

    for start, end in zip(corners[:-1], corners[1:]):
        segment = end - start
        segment_length = float(np.linalg.norm(segment))
        alphas = np.linspace(0.0, 1.0, points_per_segment, endpoint=False)

        for alpha in alphas:
            k_points.append(start + alpha * segment)
            distances.append(current_distance + alpha * segment_length)

        current_distance += segment_length
        ticks.append(current_distance)

    return np.asarray(distances), np.asarray(k_points), ticks


def lattice_sum_numpy(
        kappa: float,
        kx: float,
        ky: float,
        physics: PhysicsConfig,
) -> float:
    q_value = kappa * physics.b
    y_phase = ky * physics.a

    result = 0.0 + 0.0j

    for index in range(-physics.lattice_terms, physics.lattice_terms + 1):
        kx_shifted = kx + 2.0 * np.pi * index / physics.b
        ratio = complex(kx_shifted / kappa, 0.0)

        lambda_value = np.sqrt(ratio * ratio - 1.0 + 0.0j)
        gamma_value = np.sqrt(ratio * ratio + 1.0 + 0.0j)

        if abs(lambda_value) < physics.eps:
            lambda_value = complex(physics.eps, 0.0)

        if abs(gamma_value) < physics.eps:
            gamma_value = complex(physics.eps, 0.0)

        gamma_argument = gamma_value * q_value * physics.a / physics.b
        lambda_argument = lambda_value * q_value * physics.a / physics.b

        gamma_denominator = np.cosh(gamma_argument) - np.cos(y_phase)
        lambda_denominator = np.cosh(lambda_argument) - np.cos(y_phase)

        if abs(gamma_denominator) < physics.eps:
            gamma_denominator = complex(physics.eps, 0.0)

        if abs(lambda_denominator) < physics.eps:
            lambda_denominator = complex(physics.eps, 0.0)

        gamma_term = (
                np.sinh(gamma_argument)
                / gamma_value
                / gamma_denominator
        )

        lambda_term = (
                np.sinh(lambda_argument)
                / lambda_value
                / lambda_denominator
        )

        result += gamma_term - lambda_term

    if not np.isfinite(result.real):
        return float("nan")

    return float(result.real)


def lattice_sum_torch(
        kappa: torch.Tensor,
        kx: torch.Tensor,
        ky: torch.Tensor,
        physics: PhysicsConfig,
) -> torch.Tensor:
    q_value = kappa.to(torch.complex128) * physics.b
    y_phase = (ky * physics.a).to(torch.complex128)

    result = torch.zeros_like(q_value, dtype=torch.complex128)

    for index in range(-physics.lattice_terms, physics.lattice_terms + 1):
        kx_shifted = kx + 2.0 * np.pi * index / physics.b
        ratio = (kx_shifted / kappa).to(torch.complex128)

        lambda_value = torch.sqrt(ratio ** 2 - 1.0 + 0.0j)
        gamma_value = torch.sqrt(ratio ** 2 + 1.0 + 0.0j)

        eps_tensor = torch.full_like(lambda_value.real, physics.eps)
        lambda_value = torch.where(
            torch.abs(lambda_value) < physics.eps,
            eps_tensor.to(torch.complex128),
            lambda_value,
        )

        gamma_value = torch.where(
            torch.abs(gamma_value) < physics.eps,
            eps_tensor.to(torch.complex128),
            gamma_value,
        )

        lambda_argument = lambda_value * q_value * physics.a / physics.b
        gamma_argument = gamma_value * q_value * physics.a / physics.b

        lambda_denominator = torch.cosh(lambda_argument) - torch.cos(y_phase)
        gamma_denominator = torch.cosh(gamma_argument) - torch.cos(y_phase)

        lambda_denominator = torch.where(
            torch.abs(lambda_denominator) < physics.eps,
            eps_tensor.to(torch.complex128),
            lambda_denominator,
        )

        gamma_denominator = torch.where(
            torch.abs(gamma_denominator) < physics.eps,
            eps_tensor.to(torch.complex128),
            gamma_denominator,
        )

        gamma_term = (
                torch.sinh(gamma_argument)
                / gamma_value
                / gamma_denominator
        )

        lambda_term = (
                torch.sinh(lambda_argument)
                / lambda_value
                / lambda_denominator
        )

        result = result + gamma_term - lambda_term

    return result.real.to(dtype=DTYPE)


def track_bands(
        candidates_by_point: list[list[float]],
        config: TrainConfig,
) -> np.ndarray:
    bands = np.full((len(candidates_by_point), config.bands), np.nan, dtype=float)
    previous = np.full(config.bands, np.nan, dtype=float)
    max_jump = 0.75

    for point_id, candidates in enumerate(candidates_by_point):
        if not candidates:
            continue

        unused = candidates.copy()
        assigned = np.full(config.bands, np.nan, dtype=float)

        for band_id in range(config.bands):
            if np.isfinite(previous[band_id]) and unused:
                distances = np.abs(np.asarray(unused) - previous[band_id])
                best_id = int(np.argmin(distances))

                if distances[best_id] <= max_jump:
                    assigned[band_id] = unused.pop(best_id)

        for band_id in range(config.bands):
            if not np.isfinite(assigned[band_id]) and unused:
                assigned[band_id] = unused.pop(0)

        assigned = np.sort(assigned[np.isfinite(assigned)])
        bands[point_id, :len(assigned)] = assigned[:config.bands]

        for band_id in range(config.bands):
            if np.isfinite(bands[point_id, band_id]):
                previous[band_id] = bands[point_id, band_id]

    return bands


def find_numerical_bands(
        k_points: np.ndarray,
        physics: PhysicsConfig,
        config: TrainConfig,
) -> tuple[np.ndarray, list[float]]:
    scan_grid = np.linspace(config.kappa_min, config.kappa_max, config.root_scan_points)
    candidates_by_point: list[list[float]] = []
    log_step = max(1, len(k_points) // 12)

    for point_id, (kx_value, ky_value) in enumerate(k_points):
        if point_id % log_step == 0 or point_id == len(k_points) - 1:
            print(f"root finding: {point_id + 1:4d}/{len(k_points)} k-points")

        values = []
        for kappa in scan_grid:
            left_side = 4.0 * physics.b * float(kappa) ** 3 / physics.mu
            right_side = lattice_sum_numpy(
                float(kappa),
                float(kx_value),
                float(ky_value),
                physics,
            )

            values.append(left_side - right_side)

        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values) & (np.abs(values) < 1e8)

        values = np.where(finite, values, np.nan)
        roots: list[float] = []

        for left, right, f_left, f_right in zip(
                scan_grid[:-1],
                scan_grid[1:],
                values[:-1],
                values[1:],
        ):
            if not np.isfinite(f_left) or not np.isfinite(f_right):
                continue

            if abs(f_left - f_right) > 1e6:
                continue

            if f_left * f_right > 0.0:
                continue

            try:
                root_result = brentq(
                    lambda value: (
                            4.0 * physics.b * float(value) ** 3 / physics.mu
                            - lattice_sum_numpy(
                                float(value),
                                float(kx_value),
                                float(ky_value),
                                physics,
                            )
                    ),
                    float(left),
                    float(right),
                    maxiter=100,
                    full_output=False,
                )
            except ValueError:
                continue

            if isinstance(root_result, tuple):
                root = float(root_result[0])
            else:
                root = float(root_result)

            if all(abs(root - old_root) > 2e-3 for old_root in roots):
                roots.append(root)

            if len(roots) >= max(config.bands + 4, 6):
                break

        candidates_by_point.append(sorted(roots))

    kappa_bands = track_bands(candidates_by_point, config)
    coverages = []

    for band_id in range(config.bands):
        coverage = float(np.isfinite(kappa_bands[:, band_id]).mean())
        coverages.append(coverage)

        if band_id == 0:
            print(f"\nband {band_id + 1} coverage: {100.0 * coverage:.2f}%")
        else:
            print(f"band {band_id + 1} coverage: {100.0 * coverage:.2f}%")

    for band_id, coverage in enumerate(coverages):
        if coverage < config.min_band_coverage:
            raise RuntimeError(
                f"Band {band_id + 1} coverage is too low: "
                f"{100.0 * coverage:.2f}%. Reduce --bands or tune root search."
            )

    return kappa_bands, coverages


def model_losses(
        model: WaveSpectraPINN,
        k_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        physics: PhysicsConfig,
        config: TrainConfig,
        physics_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = model.forward(k_tensor)
    known = torch.isfinite(target_tensor)

    if known.any():
        data_loss = functional.smooth_l1_loss(
            prediction[known],
            target_tensor[known],
            beta=0.03,
        )
    else:
        data_loss = torch.zeros((), dtype=DTYPE, device=k_tensor.device)

    kx_values = k_tensor[:, 0:1]
    ky_values = k_tensor[:, 1:2]

    left_side = 4.0 * physics.b * prediction ** 3 / physics.mu
    right_side = lattice_sum_torch(prediction, kx_values, ky_values, physics)

    residual = torch.clamp(
        left_side - right_side,
        -config.residual_clip,
        config.residual_clip,
    )

    physics_loss = torch.mean((residual / config.residual_scale) ** 2)

    if prediction.shape[0] > 2:
        second_diff = prediction[2:] - 2.0 * prediction[1:-1] + prediction[:-2]
        smooth_loss = torch.mean(second_diff ** 2)
    else:
        smooth_loss = torch.zeros((), dtype=DTYPE, device=k_tensor.device)

    if prediction.shape[1] > 1:
        distances = prediction[:, 1:] - prediction[:, :-1]
        order_loss = torch.mean(

            functional.relu(config.min_band_distance - distances) ** 2,
        )
    else:
        order_loss = torch.zeros((), dtype=DTYPE, device=k_tensor.device)

    loss = (
            config.data_weight * data_loss
            + physics_weight * physics_loss
            + config.smooth_weight * smooth_loss
            + config.order_weight * order_loss
    )

    parts = {
        "data_loss": data_loss,
        "physics_loss": physics_loss,
        "smooth_loss": smooth_loss,
        "order_loss": order_loss,
    }

    return loss, parts


def train_model(
        model: WaveSpectraPINN,
        k_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        physics: PhysicsConfig,
        config: TrainConfig,
        output_dir: Path,
) -> list[dict[str, float | int | str]]:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate * 0.05,
    )

    history = []
    best_loss = float("inf")

    best_state = copy.deepcopy(model.state_dict())
    sample_count = k_tensor.shape[0]

    for epoch in range(config.epochs):
        model.train()
        permutation = torch.randperm(sample_count, device=k_tensor.device)
        progress = epoch / max(config.epochs - 1, 1)
        physics_weight = config.physics_weight_start * (
                config.physics_weight_end / config.physics_weight_start
        ) ** progress

        for start in range(0, sample_count, config.batch_size):
            ids = permutation[start:start + config.batch_size]
            loss, _ = model_losses(
                model,
                k_tensor[ids],
                target_tensor[ids],
                physics,
                config,
                physics_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        scheduler.step()

        if epoch % config.log_every == 0 or epoch == config.epochs - 1:
            model.eval()

            with torch.enable_grad():
                full_loss, parts = model_losses(
                    model,
                    k_tensor,
                    target_tensor,
                    physics,
                    config,
                    physics_weight,
                )

            loss_value = float(full_loss.detach().cpu())

            row = {
                "stage": "adam",
                "epoch": epoch,
                "loss": loss_value,
                "data_loss": float(parts["data_loss"].detach().cpu()),
                "physics_loss": float(parts["physics_loss"].detach().cpu()),
                "smooth_loss": float(parts["smooth_loss"].detach().cpu()),
                "order_loss": float(parts["order_loss"].detach().cpu()),
                "physics_weight": float(physics_weight),
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }

            history.append(row)

            if loss_value < best_loss:
                best_loss = loss_value
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, checkpoint_dir / "best_model.pt")

            print(
                f"adam {epoch:5d} | loss={row['loss']:.4e} | "
                f"data={row['data_loss']:.4e} | "
                f"physics={row['physics_loss']:.4e} | "
                f"smooth={row['smooth_loss']:.4e} | "
                f"w_phys={row['physics_weight']:.2e}"
            )

    model.load_state_dict(best_state)

    if config.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=config.lbfgs_lr,
            max_iter=config.lbfgs_iterations,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        for step in range(config.lbfgs_steps):
            def closure() -> torch.Tensor:
                lbfgs.zero_grad(set_to_none=True)
                loss, _ = model_losses(
                    model,
                    k_tensor,
                    target_tensor,
                    physics,
                    config,
                    config.physics_weight_end,
                )
                loss.backward()

                return loss

            lbfgs.step(closure)

            model.eval()
            with torch.enable_grad():
                current_loss, _ = model_losses(
                    model,
                    k_tensor,
                    target_tensor,
                    physics,
                    config,
                    config.physics_weight_end,
                )

            current_value = float(current_loss.detach().cpu())
            history.append(
                {
                    "stage": "lbfgs",
                    "epoch": step,
                    "loss": current_value,
                    "data_loss": float("nan"),
                    "physics_loss": float("nan"),
                    "smooth_loss": float("nan"),
                    "order_loss": float("nan"),
                    "physics_weight": config.physics_weight_end,
                    "learning_rate": config.lbfgs_lr,
                }
            )

            if step == 0:
                print(f"\nlbfgs {step:4d} | loss={current_value:.4e}")
            else:
                print(f"lbfgs {step:4d} | loss={current_value:.4e}")

            if current_value < best_loss:
                best_loss = current_value
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, checkpoint_dir / "best_model.pt")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), checkpoint_dir / "final_model.pt")

    return history


def save_results(
        distances: np.ndarray,
        k_points: np.ndarray,
        ticks: list[float],
        coverages: list[float],
        kappa_numerical: np.ndarray,
        kappa_pinn: np.ndarray,
        history: list[dict[str, float | int | str]],
        physics: PhysicsConfig,
        output_dir: Path,
) -> None:
    log_dir = output_dir / "logs"
    plot_dir = output_dir / "plots"
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    omega_numerical = np.sqrt(
        physics.stiffness
        * np.asarray(kappa_numerical) ** 4
        / (physics.density * physics.thickness)
    )

    omega_pinn = np.sqrt(
        physics.stiffness
        * np.asarray(kappa_pinn) ** 4
        / (physics.density * physics.thickness)
    )

    finite = np.isfinite(omega_numerical) & np.isfinite(omega_pinn)
    error = omega_pinn[finite] - omega_numerical[finite]

    metrics = {
        "MSE": float(np.mean(error ** 2)),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "rMAE": float(
            np.mean(np.abs(error))
            / (np.mean(np.abs(omega_numerical[finite])) + 1e-12),
        ),
        "rRMSE": float(
            np.sqrt(np.mean(error ** 2))
            / (np.sqrt(np.mean(omega_numerical[finite] ** 2)) + 1e-12),
        ),
    }

    band_gaps = []
    for band_id in range(omega_numerical.shape[1] - 1):
        lower_max = float(np.nanmax(omega_numerical[:, band_id]))
        upper_min = float(np.nanmin(omega_numerical[:, band_id + 1]))
        gap = upper_min - lower_max
        band_gaps.append((band_id + 1, band_id + 2, lower_max, upper_min, gap))

    with open(log_dir / "history.csv", "w", encoding="utf-8", newline="") as file:
        writer = DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    with open(log_dir / "metrics.txt", "w", encoding="utf-8") as file:
        file.write("Metrics PINN and brentq\n")
        for name, value in metrics.items():
            file.write(f"{name}: {value:.10e}\n")

        file.write("\nReference coverage\n")
        for band_id, coverage in enumerate(coverages):
            file.write(f"band {band_id + 1}: {100.0 * coverage:.4f}%\n")

        file.write("\nBand gaps from numerical solution\n")
        for left_band, right_band, lower_max, upper_min, gap in band_gaps:
            file.write(
                f"bands {left_band}-{right_band}: "
                f"lower_max={lower_max:.8e}, "
                f"upper_min={upper_min:.8e}, "
                f"gap={gap:.8e}, "
                f"exists={gap > 0.0}\n"
            )

    np.savez(
        output_dir / "data.npz",
        distances=distances,
        k_points=k_points,
        ticks=np.asarray(ticks),
        coverages=np.asarray(coverages),
        kappa_numerical=kappa_numerical,
        kappa_pinn=kappa_pinn,
        omega_numerical=omega_numerical,
        omega_pinn=omega_pinn,
    )

    plot_results(
        distances,
        ticks,
        omega_numerical,
        omega_pinn,
        band_gaps,
        history,
        plot_dir,
    )

    print("\nMetrics PINN and brentq:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.8e}")


def plot_results(
        distances: np.ndarray,
        ticks: list[float],
        omega_numerical: np.ndarray,
        omega_pinn: np.ndarray,
        band_gaps: list[tuple[int, int, float, float, float]],
        history: list[dict[str, float | int | str]],
        plots_dir: Path,
) -> None:
    labels = ["$\\Gamma$", "X", "M", "$\\Gamma$"]

    plt.figure(figsize=(16, 8))

    for band_id in range(omega_numerical.shape[1]):
        plt.plot(
            distances,
            omega_numerical[:, band_id],
            "--",
            linewidth=2.4,
            label=f"brentq band {band_id + 1}",
        )

        plt.plot(
            distances,
            omega_pinn[:, band_id],
            linewidth=2.2,
            label=f"PINN band {band_id + 1}",
        )

    for left_band, right_band, lower_max, upper_min, gap in band_gaps:
        if gap <= 0.0:
            continue
        plt.axhspan(lower_max, upper_min, alpha=0.18, label=f"band gap {left_band}-{right_band}")

    for tick in ticks:
        plt.axvline(tick, linewidth=1.0, alpha=0.35)

    plt.xticks(ticks, labels, fontsize=18)
    plt.yticks(fontsize=16)
    plt.xlabel("$\\Gamma - X - M - \\Gamma$", fontsize=20)
    plt.ylabel("$\\omega$", fontsize=20)
    plt.title("PINN and brentq baseline", fontsize=20)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=14, ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(plots_dir / "bands_pinn_brentq.png", dpi=300)
    plt.close()

    plt.figure(figsize=(16, 8))

    for band_id in range(omega_numerical.shape[1]):
        plt.plot(
            distances,
            np.abs(omega_pinn[:, band_id] - omega_numerical[:, band_id]),
            linewidth=2.2,
            label=f"band {band_id + 1}",
        )

    for tick in ticks:
        plt.axvline(tick, linewidth=1.0, alpha=0.35)

    plt.xticks(ticks, labels, fontsize=18)
    plt.yticks(fontsize=16)
    plt.xlabel("$\\Gamma - X - M - \\Gamma$", fontsize=20)
    plt.ylabel("$|\\omega_{PINN} - \\omega_{brentq}|$", fontsize=20)
    plt.title("Absolute error", fontsize=20)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / "error.png", dpi=300)
    plt.close()

    plt.figure(figsize=(16, 8))
    steps = np.arange(len(history))

    plt.semilogy(
        steps,
        [row["loss"] for row in history],
        linewidth=2.2,
        label="total",
    )
    plt.semilogy(
        steps,
        [row["data_loss"] for row in history],
        linewidth=2.2,
        label="data",
    )
    plt.semilogy(
        steps,
        [row["physics_loss"] for row in history],
        linewidth=2.2,
        label="physics",
    )
    plt.semilogy(
        steps,
        [row["smooth_loss"] for row in history],
        linewidth=2.2,
        label="smooth",
    )

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.title("PINN training history", fontsize=20)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / "loss.png", dpi=300)
    plt.close()


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="output",
        type=Path,
        help="Path to output directory.",
    )

    parser.add_argument(
        "--bands",
        default=1,
        type=int,
        help="Number of bands to use.",
    )

    parser.add_argument(
        "--points-per-segment",
        default=100,
        type=int,
        help="Number of points per segment.",
    )

    parser.add_argument(
        "--root-scan-points",
        default=4000,
        type=int,
        help="Number of root scan points.",
    )

    parser.add_argument(
        "--lattice-terms",
        default=8,
        type=int,
        help="Number of lattice terms.",
    )

    parser.add_argument(
        "--mu",
        default=10.0,
        type=float,
        help="Mean lattice parameter.",
    )

    parser.add_argument(
        "--kappa-min",
        default=0.05,
        type=float,
        help="Minimum kappa value.",
    )

    parser.add_argument(
        "--kappa-max",
        default=10.0,
        type=float,
        help="Maximum kappa value.",
    )

    parser.add_argument(
        "--epochs",
        default=10000,
        type=int,
        help="Number of Adam training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="Training batch size.",
    )

    parser.add_argument(
        "--hidden-dim",
        default=128,
        type=int,
        help="Hidden layer width.",
    )

    parser.add_argument(
        "--hidden-layers",
        default=5,
        type=int,
        help="Number of hidden layers.",
    )

    parser.add_argument(
        "--learning-rate",
        default=1e-3,
        type=float,
        help="Adam learning rate.",
    )

    parser.add_argument(
        "--lbfgs-steps",
        default=30,
        type=int,
        help="Number of L-BFGS outer steps.",
    )

    parser.add_argument(
        "--min-band-coverage",
        default=0.55,
        type=float,
        help="Minimum allowed numerical coverage for each band.",
    )

    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Run a short sanity check.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    physics = PhysicsConfig(mu=args.mu, lattice_terms=args.lattice_terms)
    config = TrainConfig(
        bands=args.bands,
        points_per_segment=args.points_per_segment,
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
        root_scan_points=args.root_scan_points,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lbfgs_steps=args.lbfgs_steps,
        min_band_coverage=args.min_band_coverage,
    )

    if args.sanity:
        config.bands = 1
        config.points_per_segment = 5
        config.root_scan_points = 250
        config.hidden_dim = 32
        config.hidden_layers = 3
        config.epochs = 10
        config.batch_size = 16
        config.lbfgs_steps = 0
        physics.lattice_terms = 3

    output_dir = Path(args.output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nConfig: {config}")
    print(f"\nPhysics: {physics}\n")

    distances, k_points, ticks = make_k_path(config.points_per_segment, physics)
    kappa_numerical, coverages = find_numerical_bands(k_points, physics, config)

    k_tensor = torch.tensor(k_points, dtype=DTYPE, device=DEVICE)
    target_tensor = torch.tensor(kappa_numerical, dtype=DTYPE, device=DEVICE)
    k_mean = k_tensor.mean(dim=0, keepdim=True)
    k_std = k_tensor.std(dim=0, keepdim=True).clamp_min(1e-8)

    finite_values = kappa_numerical[np.isfinite(kappa_numerical)]
    if finite_values.size == 0:
        raise RuntimeError("Root search did not find any numerical branches.")

    min_kappa = float(np.nanmin(finite_values))
    max_kappa = float(np.nanmax(finite_values))
    kappa_low = float(max(config.kappa_min, 0.95 * min_kappa))
    kappa_high = float(min(config.kappa_max, 1.05 * max_kappa))

    model = WaveSpectraPINN(
        k_mean=k_mean,
        k_std=k_std,
        kappa_low=kappa_low,
        kappa_high=kappa_high,
        config=config,
    ).to(DEVICE).to(DTYPE)

    print(f"\n{model}\n")

    history = train_model(
        model,
        k_tensor,
        target_tensor,
        physics,
        config,
        output_dir,
    )

    model.eval()
    with torch.no_grad():
        kappa_pinn = model.forward(k_tensor).detach().cpu().numpy()

    save_results(
        distances=distances,
        k_points=k_points,
        ticks=ticks,
        coverages=coverages,
        kappa_numerical=kappa_numerical,
        kappa_pinn=kappa_pinn,
        history=history,
        physics=physics,
        output_dir=output_dir,
    )

    print(f"\nResults are saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
