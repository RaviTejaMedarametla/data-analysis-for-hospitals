from __future__ import annotations


def estimate_energy_joules(runtime_s: float, power_watts: float) -> float:
    return runtime_s * power_watts


def compare_precision_energy(runtime_s: float, batch_size: int) -> dict[str, float]:
    fp32_energy = estimate_energy_joules(runtime_s, power_watts=45 + batch_size * 0.05)
    fp16_energy = estimate_energy_joules(runtime_s * 0.8, power_watts=35 + batch_size * 0.03)
    return {"fp32_joules": fp32_energy, "fp16_joules": fp16_energy, "energy_saving_ratio": 1 - (fp16_energy / fp32_energy)}
