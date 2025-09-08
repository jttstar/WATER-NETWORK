# training/train.py
import argparse, os, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 128, depth: int = 2):
        super().__init__()
        layers = [nn.Linear(d_in, hidden), nn.ReLU()]
        for _ in range(depth-1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, d_out)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def standardize(train: np.ndarray, val: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-8
    return (train-mean)/std, (val-mean)/std, (test-mean)/std, mean, std

def mae(y_true, y_pred): return float(np.mean(np.abs(y_true - y_pred)))
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Ruta a training_set_*.npz")
    ap.add_argument("--outdir", default="artifacts", help="Directorio de salida")
    ap.add_argument("--mode", choices=["p","alpha"], default="p")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--test_split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--x_names_csv", type=str, default=None, help="CSV de donde tomar el orden y los nombres de X (encabezados de columnas).")
    ap.add_argument("--x_names", type=str, default=None, help="Lista separada por comas con los nombres de features de X.")
    args = ap.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz)
    X = data["X"].astype(np.float32)
    Y = data["Y"].astype(np.float32)
    n, d_in = X.shape
    d_out = 1 if args.mode=="alpha" else Y.shape[1]

    # Determinar nombres de features de entrada
    x_feature_names = None
    if args.x_names_csv:
        import pandas as pd
        df = pd.read_csv(args.x_names_csv, nrows=1)
        x_feature_names = [c for c in df.columns if c.lower() != 't']
        if len(x_feature_names) != d_in:
            print(f"[ADVERTENCIA] #features en CSV ({len(x_feature_names)}) != d_in ({d_in}). Se usará secuencia genérica.")
            x_feature_names = None
    elif args.x_names:
        x_feature_names = [s.strip() for s in args.x_names.split(',') if s.strip()]
        if len(x_feature_names) != d_in:
            print(f"[ADVERTENCIA] #features en --x_names ({len(x_feature_names)}) != d_in ({d_in}). Se usará secuencia genérica.")
            x_feature_names = None
    if x_feature_names is None:
        x_feature_names = [f"x{i}" for i in range(d_in)]

    # splits
    n_test = int(n * args.test_split)
    n_val  = int((n - n_test) * args.val_split)
    n_train = n - n_test - n_val
    idx = np.arange(n); np.random.shuffle(idx)
    X, Y = X[idx], Y[idx]
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val,   Y_val   = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test,  Y_test  = X[n_train+n_val:], Y[n_train+n_val:]

    # standardize inputs
    X_train_s, X_val_s, X_test_s, x_mean, x_std = standardize(X_train, X_val, X_test)
    if args.mode == "alpha":
        y_mean = Y_train.mean(axis=0, keepdims=True); y_std = Y_train.std(axis=0, keepdims=True)+1e-8
        Y_train_s = (Y_train - y_mean)/y_std
        Y_val_s   = (Y_val - y_mean)/y_std
        Y_test_s  = (Y_test - y_mean)/y_std
    else:
        Y_train_s, Y_val_s, Y_test_s = Y_train, Y_val, Y_test
        y_mean = np.zeros((1, d_out), dtype=np.float32); y_std = np.ones((1, d_out), dtype=np.float32)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(Y_train_s)), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(Y_val_s)), batch_size=args.batch_size)
    test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test_s), torch.from_numpy(Y_test_s)), batch_size=args.batch_size)

    model = MLP(d_in=d_in, d_out=d_out, hidden=args.hidden, depth=args.depth).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf"); best_path = outdir / "best_model.pt"
    for epoch in range(1, args.epochs+1):
        model.train(); train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb); loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += float(loss.item()) * xb.size(0)
        train_loss /= len(train_loader.dataset)
        # val
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb); val_loss += float(loss_fn(pred, yb).item()) * xb.size(0)
        val_loss /= len(val_loader.dataset)
        if val_loss < best_val:
            best_val = val_loss; torch.save(model.state_dict(), best_path)
        print(f"Epoch {epoch:03d} | train_mse={train_loss:.6f} | val_mse={val_loss:.6f}")

    # test
    model.load_state_dict(torch.load(best_path, map_location=DEVICE)); model.eval()
    Y_true, Y_pred = [], []
    with torch.no_grad():
        for xb, yb in DataLoader(TensorDataset(torch.from_numpy(X_test_s), torch.from_numpy(Y_test_s)), batch_size=args.batch_size):
            xb = xb.to(DEVICE)
            pred = model(xb).cpu().numpy()
            Y_pred.append(pred); Y_true.append(yb.numpy())
    Y_true = np.concatenate(Y_true, axis=0); Y_pred = np.concatenate(Y_pred, axis=0)
    if args.mode == "alpha":
        Y_pred = Y_pred * y_std + y_mean
        Y_true = Y_true * y_std + y_mean

    metrics = { "test_mae": float(np.mean(np.abs(Y_true - Y_pred))), "test_rmse": float(np.sqrt(np.mean((Y_true - Y_pred)**2))), "best_val_mse": best_val }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(outdir / "scalers.npz", x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    (outdir / "x_feature_names.json").write_text(json.dumps(x_feature_names, indent=2))

    print("Artefactos en:", outdir)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
