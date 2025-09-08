# training/infer.py
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(d_in, hidden), nn.ReLU()]
        for _ in range(depth-1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, d_out)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def load_scalers(scalers_path: Path):
    sc = np.load(scalers_path)
    return sc["x_mean"], sc["x_std"], sc["y_mean"], sc["y_std"]

def standardize(X, mean, std):
    return (X - mean) / (std + 1e-8)

def parse_inputs(args):
    names = None
    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        # Remove column 't' if present
        cols = [c for c in df.columns if c.lower() != 't']
        df = df[cols]
        X = df.to_numpy(dtype=np.float32)
        names = list(df.columns)
    elif args.json:
        obj = json.loads(Path(args.json).read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            names = list(obj.keys())
            X = np.array([list(obj.values())], dtype=np.float32)
        else:
            X = np.array(obj, dtype=np.float32)
            names = [f"x{i}" for i in range(X.shape[1])]
    else:
        raise ValueError("Proporciona --csv o --json con las entradas x_g.")
    return X, names

def validate_and_map_columns(X, names, expected_names, do_map=False):
    set_in, set_exp = set(names), set(expected_names)
    if set_in != set_exp:
        missing = list(set_exp - set_in)
        extra = list(set_in - set_exp)
        msg = {"error": "Los nombres de columnas no coinciden con los esperados.",
               "faltan": missing, "sobran": extra, "esperados": expected_names, "recibidos": names}
        raise ValueError(json.dumps(msg, ensure_ascii=False, indent=2))
    if names != expected_names:
        if not do_map:
            msg = {"error": "El orden de columnas difiere del esperado.",
                   "esperados": expected_names, "recibidos": names,
                   "sugerencia": "Re-ejecuta con --map para reordenar automáticamente."}
            raise ValueError(json.dumps(msg, ensure_ascii=False, indent=2))
        idx = [names.index(n) for n in expected_names]
        X = X[:, idx]
        names = expected_names[:]
    return X, names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ruta a best_model.pt")
    ap.add_argument("--scalers", required=True, help="Ruta a scalers.npz")
    ap.add_argument("--mode", choices=["p","alpha"], default="p")
    ap.add_argument("--csv", help="CSV con filas de x_g (columnas=features)")
    ap.add_argument("--json", help="JSON con x_g (dict o lista de listas)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dout", type=int, default=None, help="Dimensión de salida si mode=p")
    ap.add_argument("--feature_names", type=str, default=None, help="Ruta a x_feature_names.json")
    ap.add_argument("--map", action="store_true", help="Reordenar automáticamente columnas al orden esperado")
    ap.add_argument("--save", type=str, default=None, help="Ruta para guardar resultados en CSV")
    args = ap.parse_args()

    # Load expected feature names
    if args.feature_names:
        exp_names = json.loads(Path(args.feature_names).read_text(encoding="utf-8"))
    else:
        model_dir = Path(args.model).resolve().parent
        feat_file = model_dir / "x_feature_names.json"
        if not feat_file.exists():
            raise FileNotFoundError("No se encontró x_feature_names.json. Proporciona --feature_names o re-entrena guardando el archivo.")
        exp_names = json.loads(feat_file.read_text(encoding="utf-8"))

    X, names = parse_inputs(args)
    X, names = validate_and_map_columns(X, names, exp_names, do_map=args.map)

    x_mean, x_std, y_mean, y_std = load_scalers(Path(args.scalers))
    d_in = X.shape[1]
    d_out = 1 if args.mode=="alpha" else (args.dout if args.dout is not None else y_mean.shape[1])

    model = MLP(d_in=d_in, d_out=d_out, hidden=args.hidden, depth=args.depth).to(DEVICE)
    sd = torch.load(args.model, map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()

    Xs = standardize(X, x_mean, x_std)
    with torch.no_grad():
        y_hat = model(torch.from_numpy(Xs).to(DEVICE)).cpu().numpy()

    if args.mode == "alpha":
        y_hat = y_hat * y_std + y_mean
        result = {"alpha_pred": y_hat.squeeze().tolist()}
        cols = ["alpha_pred"]
        rows = [[float(v)] for v in y_hat.squeeze()]
    else:
        result = {"p_pred": y_hat.tolist(), "columns": [f"p{i}" for i in range(d_out)]}
        cols = result["columns"]
        rows = y_hat.tolist()

    print(json.dumps(result, indent=2))

    if args.save:
        import pandas as pd
        df = pd.DataFrame(rows, columns=cols)
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[OK] Resultados guardados en {out_path}")

if __name__ == "__main__":
    main()
