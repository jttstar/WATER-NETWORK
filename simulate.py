# src/simulate.py
import argparse, yaml
from pathlib import Path
import numpy as np
import pandas as pd

def daily_profile(hours: int):
    import math
    return np.array([1.0 + 0.5*math.sin(2*math.pi*(h%24)/24.0) for h in range(hours)])

def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    outdir = Path(cfg['output']['processed_dir']); outdir.mkdir(parents=True, exist_ok=True)
    tag = cfg['output'].get('tag', 'demo')

    hours = int(cfg['simulation']['duration_hours'])
    sinks = cfg['nodes']['sinks']
    profile = daily_profile(hours)
    rng = np.random.default_rng(cfg['scenarios'].get('random_seed', 42))

    # Series de demanda
    ts = pd.DataFrame({'t': np.arange(hours)})
    for node, base_demand in sinks.items():
        ts[node] = base_demand * profile * (1 + rng.normal(0, cfg['scenarios']['demand_noise_std'], hours))

    ts.to_csv(outdir / f"timeseries_{tag}.csv", index=False)

    # Etiquetas mínimas (dummy: usamos directamente x_g como p)
    labels = ts.copy().drop(columns=['t'])
    labels.to_csv(outdir / f"labels_min_cost_{tag}.csv", index=False)

    # Dataset para ML
    X = ts.drop(columns=['t']).to_numpy(dtype=np.float32)
    Y = labels.to_numpy(dtype=np.float32)
    np.savez(outdir / f"training_set_{tag}.npz", X=X, Y=Y)
    print(f"[OK] Dataset guardado en {outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
