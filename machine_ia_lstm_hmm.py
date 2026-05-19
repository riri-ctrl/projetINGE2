# -*- coding: utf-8 -*-
"""
Modèles LSTM et HMM pour prédiction financière — v2
Améliorations :
  - 1 seul modèle LSTM pour tous les tickers (via Embedding layer)
  - Index (Date, Ticker) sur tous les DataFrames
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from hmmlearn.hmm import GaussianHMM

from traitement_de_donnee2 import fill_indice, fill_df, fill_bourse, ticker_encoder


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

SEQUENCE_LEN  = 30
HORIZON       = 5
BATCH_SIZE    = 64
EPOCHS        = 50
LEARNING_RATE = 1e-3
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
DROPOUT       = 0.2
EMBED_DIM     = 8        # Dimension de l'embedding ticker
HMM_N_STATES  = 4
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA20", "MA50", "Momentum", "RSI",
    "MACD", "MACD_Signal", "MACD_Hist",
    "Volatility_20d", "BB_Upper", "BB_Lower", "BB_Mid",
    "is_trading_day",
]
TARGET_COL = "Close"


# =============================================================================
# ÉTAPE 0 — Construction du MultiIndex (Date, Ticker)
# =============================================================================

def build_multiindex_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reçoit le DataFrame brut (colonnes Date + Ticker + features).
    Retourne un DataFrame avec un MultiIndex (Date, Ticker) trié.

    Pourquoi un MultiIndex ?
      - df.loc[pd.IndexSlice[:, ticker_id], :]  → toutes les dates d'un ticker
      - df.loc[pd.IndexSlice["2024-01-01":"2024-12-31", :], :]  → fenêtre temporelle
      - Plus aucun filtre .query() ou booléen dispersé dans le code
    """
    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = (
        df.set_index(["Date", "Ticker"])
          .sort_index()           # tri lexicographique (date, ticker)
    )
    return df


# =============================================================================
# UTILITAIRES
# =============================================================================

def select_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  ⚠️  Colonnes absentes (ignorées) : {missing}")
    return df[available].copy()


def build_sequences(data: np.ndarray, ticker_ids: np.ndarray,
                    target: np.ndarray, seq_len: int, horizon: int):
    """
    Construit des séquences glissantes (X_feat, X_ticker, y).
      X_feat   : (N, seq_len, n_features)  — features continues
      X_ticker : (N,)                       — id entier du ticker (constant sur la séquence)
      y        : (N, horizon)
    Le ticker id est extrait au dernier pas de temps de chaque fenêtre
    (il est identique sur toute la séquence d'un même titre).
    """
    X_feat, X_tick, y = [], [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X_feat.append(data[i : i + seq_len])
        X_tick.append(ticker_ids[i + seq_len - 1])   # entier : id du ticker
        y.append(target[i + seq_len : i + seq_len + horizon])
    return (
        np.array(X_feat, dtype=np.float32),
        np.array(X_tick, dtype=np.int64),
        np.array(y,      dtype=np.float32),
    )


def train_val_split_temporal(*arrays, val_ratio: float = 0.15):
    """Découpe temporelle (pas de mélange aléatoire)."""
    split = int(len(arrays[0]) * (1 - val_ratio))
    return tuple(a[:split] for a in arrays) + tuple(a[split:] for a in arrays)


# =============================================================================
# DATASET PYTORCH — accepte maintenant les ids ticker
# =============================================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, X_feat: np.ndarray, X_ticker: np.ndarray, y: np.ndarray):
        self.X_feat   = torch.from_numpy(X_feat)
        self.X_ticker = torch.from_numpy(X_ticker)   # LongTensor
        self.y        = torch.from_numpy(y)

    def __len__(self):
        return len(self.X_feat)

    def __getitem__(self, idx):
        return self.X_feat[idx], self.X_ticker[idx], self.y[idx]


# =============================================================================
# ARCHITECTURE LSTM MULTI-TICKERS
# =============================================================================

class FinancialLSTM(nn.Module):
    """
    LSTM unique pour tous les tickers.

    Fonctionnement :
      1. L'id entier du ticker passe dans une Embedding layer → vecteur (embed_dim,)
      2. Ce vecteur est concaténé à chaque pas de temps des features continues
         → entrée LSTM de taille (input_size + embed_dim)
      3. La tête linéaire prédit `horizon` valeurs futures

    Avantage de l'Embedding vs one-hot / colonne normalisée :
      - Représentation dense apprise (pas de biais de distance entre tickers)
      - Paramétrage faible : embed_dim * n_tickers au lieu de n_tickers * seq_len
    """
    def __init__(self, input_size: int, n_tickers: int, embed_dim: int,
                 hidden_size: int, num_layers: int, dropout: float, horizon: int):
        super().__init__()

        # Embedding : ticker_id (entier) → vecteur dense
        self.ticker_emb = nn.Embedding(
            num_embeddings = n_tickers,
            embedding_dim  = embed_dim,
            padding_idx    = None,
        )

        # LSTM : reçoit features + vecteur ticker à chaque pas de temps
        self.lstm = nn.LSTM(
            input_size  = input_size + embed_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, horizon)

    def forward(self, x_feat, x_ticker):
        """
        x_feat   : (batch, seq_len, input_size)
        x_ticker : (batch,)  — id entier du ticker
        """
        # Embedding → (batch, embed_dim) → répété sur toute la séquence
        emb = self.ticker_emb(x_ticker)                    # (batch, embed_dim)
        emb = emb.unsqueeze(1).expand(-1, x_feat.size(1), -1)  # (batch, seq_len, embed_dim)

        # Concaténation feature + embedding sur la dernière dimension
        x = torch.cat([x_feat, emb], dim=-1)               # (batch, seq_len, input+embed)

        out, _ = self.lstm(x)                              # (batch, seq_len, hidden)
        out     = self.dropout(out[:, -1, :])              # dernier pas de temps
        return self.fc(out)                                # (batch, horizon)


# =============================================================================
# PRÉPARATION DES DONNÉES — TOUS LES TICKERS ENSEMBLE
# =============================================================================

def prepare_all_tickers(mi_df: pd.DataFrame):
    """
    Reçoit le DataFrame MultiIndex (Date, Ticker).
    Construit un seul jeu X / X_ticker / y en concaténant
    les séquences de tous les tickers.

    Points clés :
      - Les séquences d'un ticker ne chevauchent JAMAIS celles d'un autre
        (on itère ticker par ticker)
      - Un scaler global est fitté sur l'ensemble des données d'entraînement
        pour ne pas fuir d'information
      - La normalisation de la cible (Close) est séparée pour pouvoir
        dénormaliser les prédictions
    """
    tickers = mi_df.index.get_level_values("Ticker").unique()
    n_tickers = len(tickers)

    # Mapping ticker_encoded → indice compact [0, n_tickers[
    ticker_to_idx = {t: i for i, t in enumerate(sorted(tickers))}

    all_X_feat, all_X_tick, all_y = [], [], []

    # ── Collecte brute (pas encore normalisée) ────────────────────────────
    raw_feats  = []   # liste de np.ndarray (n_dates_ticker, n_feat)
    raw_target = []   # liste de np.ndarray (n_dates_ticker,)
    raw_tids   = []   # liste d'entiers (un par ligne)

    for ticker_enc in sorted(tickers):
        # .loc sur le niveau Ticker du MultiIndex
        df_t = mi_df.loc[pd.IndexSlice[:, ticker_enc], :].droplevel("Ticker")
        df_t = df_t.sort_index()   # tri chronologique
        df_t = df_t.dropna(subset=[TARGET_COL])

        feat_df = select_features(df_t, FEATURE_COLS)
        feat_df = feat_df.ffill().bfill().fillna(0)

        if len(feat_df) < SEQUENCE_LEN + HORIZON + 20:
            print(f"  ⚠️  Ticker {ticker_enc} ignoré (données insuffisantes).")
            continue

        tid = ticker_to_idx[ticker_enc]
        raw_feats.append(feat_df.values)
        raw_target.append(feat_df[TARGET_COL].values if TARGET_COL in feat_df.columns
                          else np.zeros(len(feat_df)))
        raw_tids.append(np.full(len(feat_df), tid, dtype=np.int64))

    if not raw_feats:
        raise RuntimeError("Aucun ticker avec suffisamment de données.")

    # ── Normalisation GLOBALE (fit sur toutes les données brutes) ─────────
    # On empile toutes les lignes pour fitter les scalers une seule fois.
    all_raw = np.vstack(raw_feats)
    all_tgt = np.concatenate(raw_target).reshape(-1, 1)

    feat_scaler   = MinMaxScaler().fit(all_raw)
    target_scaler = MinMaxScaler().fit(all_tgt)

    # ── Construction des séquences par ticker ─────────────────────────────
    for feat_arr, tgt_arr, tid_arr in zip(raw_feats, raw_target, raw_tids):
        scaled_feat = feat_scaler.transform(feat_arr).astype(np.float32)
        scaled_tgt  = target_scaler.transform(
            tgt_arr.reshape(-1, 1)
        ).astype(np.float32).flatten()

        X_f, X_t, y = build_sequences(
            scaled_feat, tid_arr, scaled_tgt, SEQUENCE_LEN, HORIZON
        )
        all_X_feat.append(X_f)
        all_X_tick.append(X_t)
        all_y.append(y)

    X_feat   = np.concatenate(all_X_feat, axis=0)
    X_ticker = np.concatenate(all_X_tick, axis=0)
    y        = np.concatenate(all_y,      axis=0)

    print(f"  Dataset global : {len(X_feat)} séquences | "
          f"{X_feat.shape[2]} features | {n_tickers} tickers")

    return X_feat, X_ticker, y, feat_scaler, target_scaler, n_tickers, ticker_to_idx


# =============================================================================
# ENTRAÎNEMENT DU LSTM
# =============================================================================

def train_lstm(model: nn.Module, train_loader: DataLoader,
               val_loader: DataLoader, epochs: int, lr: float) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_f, X_t, y_b in train_loader:
            X_f, X_t, y_b = X_f.to(DEVICE), X_t.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            pred  = model(X_f, X_t)
            loss  = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_f)
        train_loss /= len(train_loader.dataset)

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_f, X_t, y_b in val_loader:
                X_f, X_t, y_b = X_f.to(DEVICE), X_t.to(DEVICE), y_b.to(DEVICE)
                val_loss += criterion(model(X_f, X_t), y_b).item() * len(X_f)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


# =============================================================================
# PIPELINE LSTM GLOBAL (tous les tickers d'un coup)
# =============================================================================

def run_lstm_pipeline(mi_df: pd.DataFrame) -> dict:
    """
    Pipeline LSTM unique entraîné sur tous les tickers.
    Reçoit le DataFrame MultiIndex (Date, Ticker).
    """
    print(f"\n{'='*60}")
    print("  LSTM — Modèle unique multi-tickers")
    print(f"{'='*60}")

    X_feat, X_ticker, y, feat_scaler, target_scaler, n_tickers, ticker_to_idx = \
        prepare_all_tickers(mi_df)

    # Découpe temporelle globale
    split = int(len(X_feat) * 0.85)
    X_f_tr, X_f_val = X_feat[:split],   X_feat[split:]
    X_t_tr, X_t_val = X_ticker[:split], X_ticker[split:]
    y_tr,   y_val   = y[:split],        y[split:]

    print(f"  Train : {len(X_f_tr)} | Val : {len(X_f_val)}")

    train_dl = DataLoader(
        TimeSeriesDataset(X_f_tr, X_t_tr, y_tr),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    val_dl = DataLoader(
        TimeSeriesDataset(X_f_val, X_t_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    model = FinancialLSTM(
        input_size  = X_feat.shape[2],
        n_tickers   = n_tickers,
        embed_dim   = EMBED_DIM,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
        horizon     = HORIZON,
    ).to(DEVICE)

    model = train_lstm(model, train_dl, val_dl, EPOCHS, LEARNING_RATE)

    # ── Évaluation globale sur la validation ──────────────────────────────
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X_f, X_t, y_b in val_dl:
            pred = model(X_f.to(DEVICE), X_t.to(DEVICE)).cpu().numpy()
            preds.append(pred)
            actuals.append(y_b.numpy())

    preds   = np.concatenate(preds,   axis=0)
    actuals = np.concatenate(actuals, axis=0)

    pred_inv   = target_scaler.inverse_transform(preds[:, :1])
    actual_inv = target_scaler.inverse_transform(actuals[:, :1])

    rmse = np.sqrt(mean_squared_error(actual_inv, pred_inv))
    mae  = mean_absolute_error(actual_inv, pred_inv)
    print(f"\n  📈 RMSE global (J+1) : {rmse:.4f}")
    print(f"  📈 MAE  global (J+1) : {mae:.4f}")

    return {
        "model"          : model,
        "feat_scaler"    : feat_scaler,
        "target_scaler"  : target_scaler,
        "ticker_to_idx"  : ticker_to_idx,
        "n_tickers"      : n_tickers,
        "rmse"           : rmse,
        "mae"            : mae,
        "n_features"     : X_feat.shape[2],
    }


# =============================================================================
# PIPELINE HMM — inchangé dans la logique, adapté au MultiIndex
# =============================================================================

def prepare_hmm_features(df: pd.DataFrame) -> tuple:
    """df est déjà trié chronologiquement, index = Date uniquement."""
    obs_cols = []

    if "Close" in df.columns:
        df = df.copy()
        df["return_1d"] = df["Close"].pct_change().fillna(0)
        obs_cols.append("return_1d")
    if "Volatility_20d" in df.columns:
        obs_cols.append("Volatility_20d")
    if "Volume" in df.columns:
        df = df.copy()
        df["Volume_norm"] = (df["Volume"] - df["Volume"].mean()) / (df["Volume"].std() + 1e-9)
        obs_cols.append("Volume_norm")
    if "RSI" in df.columns:
        df = df.copy()
        df["RSI_norm"] = df["RSI"] / 100.0
        obs_cols.append("RSI_norm")

    obs = df[obs_cols].ffill().bfill().fillna(0).values
    return obs.astype(np.float32), obs_cols


def run_hmm_pipeline(mi_df: pd.DataFrame, ticker_enc: int,
                     ticker_name: str, n_states: int = HMM_N_STATES) -> dict | None:
    """
    HMM pour un seul ticker extrait du MultiIndex.
    On garde 1 HMM par ticker car les régimes de marché sont
    spécifiques à chaque titre (volatilité, liquidité…).
    """
    print(f"\n{'='*60}")
    print(f"  HMM — {ticker_name}  |  États : {n_states}")
    print(f"{'='*60}")

    # Extraction propre via le MultiIndex
    df = (
        mi_df.loc[pd.IndexSlice[:, ticker_enc], :]
             .droplevel("Ticker")
             .sort_index()
             .dropna(subset=["Close"])
    )

    if len(df) < 100:
        print("  ⚠️  Données insuffisantes pour le HMM.")
        return None

    obs, obs_cols = prepare_hmm_features(df)
    print(f"  Features HMM : {obs_cols}")

    split     = int(len(obs) * 0.85)
    obs_train = obs[:split]
    obs_test  = obs[split:]

    model = GaussianHMM(
        n_components    = n_states,
        covariance_type = "full",
        n_iter          = 200,
        random_state    = 42,
        tol             = 1e-4,
    )
    model.fit(obs_train)
    print(f"  ✅ Log-vraisemblance train : {model.score(obs_train):.2f}")

    hs_train = model.predict(obs_train)
    hs_test  = model.predict(obs_test)

    # On ré-attache les états au DataFrame MultiIndex d'origine
    # en reconstruisant un MultiIndex sur les sous-ensembles
    df_train = df.iloc[:split].copy()
    df_test  = df.iloc[split:].copy()
    df_train["hmm_state"] = hs_train
    df_test["hmm_state"]  = hs_test

    # Remettre le ticker dans l'index pour cohérence avec mi_df
    df_train.index = pd.MultiIndex.from_arrays(
        [df_train.index, [ticker_enc] * len(df_train)], names=["Date", "Ticker"]
    )
    df_test.index = pd.MultiIndex.from_arrays(
        [df_test.index, [ticker_enc] * len(df_test)], names=["Date", "Ticker"]
    )

    # Statistiques par état
    print(f"\n  {'État':<6} {'Nb jours':<10} {'Rdt moy (%)':<15} {'Rdt std (%)':<15} {'Label'}")
    print("  " + "-"*65)

    df_train_flat = df_train.copy()
    if "return_1d" not in df_train_flat.columns:
        df_train_flat["return_1d"] = df_train_flat["Close"].pct_change().fillna(0)

    state_stats   = df_train_flat.groupby("hmm_state")["return_1d"].agg(["count","mean","std"])
    state_labels  = {}

    for state, row in state_stats.iterrows():
        m = row["mean"] * 100
        s = row["std"]  * 100 if not np.isnan(row["std"]) else 0
        label = (
            "📈 Haussier calme"    if m >  0.05 and s < 1.5  else
            "📈 Haussier volatile" if m >  0.05               else
            "📉 Baissier volatile" if m < -0.05 and s >= 1.5 else
            "📉 Baissier calme"    if m < -0.05               else
            "➡️  Neutre / range"
        )
        state_labels[state] = label
        print(f"  {state:<6} {int(row['count']):<10} {m:<15.4f} {s:<15.4f} {label}")

    last_state = hs_test[-1]
    next_state = int(np.argmax(model.transmat_[last_state]))
    print(f"\n  🔮 Dernier état : {last_state} ({state_labels.get(last_state,'?')})")
    print(f"  🔮 Prochain état : {next_state} ({state_labels.get(next_state,'?')})")

    return {
        "model"        : model,
        "state_labels" : state_labels,
        "df_train"     : df_train,
        "df_test"      : df_test,
        "last_state"   : last_state,
        "next_state"   : next_state,
    }


# =============================================================================
# PIPELINE COMBINÉ
# =============================================================================

def run_combined_pipeline(raw_df: pd.DataFrame, n_tickers: int = None) -> dict:
    """
    1. Construit le MultiIndex (Date, Ticker)
    2. Entraîne 1 LSTM global
    3. Entraîne 1 HMM par ticker
    """
    # ── MultiIndex ─────────────────────────────────────────────────────────
    mi_df = build_multiindex_df(raw_df)

    all_tickers = mi_df.index.get_level_values("Ticker").unique()
    if n_tickers is not None:
        all_tickers = all_tickers[:n_tickers]
        mi_df = mi_df.loc[pd.IndexSlice[:, all_tickers], :]

    # ── LSTM unique ────────────────────────────────────────────────────────
    lstm_result = run_lstm_pipeline(mi_df)

    # ── HMM par ticker ─────────────────────────────────────────────────────
    hmm_results = {}
    for ticker_enc in all_tickers:
        try:
            name = ticker_encoder.inverse_transform([ticker_enc])[0]
        except Exception:
            name = str(ticker_enc)

        hmm_results[name] = run_hmm_pipeline(mi_df, ticker_enc, name)

    return {"lstm": lstm_result, "hmm": hmm_results}


# =============================================================================
# INFÉRENCE
# =============================================================================

def predict_lstm(lstm_result: dict, recent_df: pd.DataFrame,
                 ticker_enc: int, seq_len: int = SEQUENCE_LEN) -> np.ndarray:
    """
    Prédit les `HORIZON` prochains cours de clôture pour un ticker donné.

    recent_df : DataFrame brut (non indexé) des dernières `seq_len` lignes
    ticker_enc : valeur encodée du ticker (pour récupérer son index compact)
    """
    model          = lstm_result["model"]
    feat_scaler    = lstm_result["feat_scaler"]
    target_scaler  = lstm_result["target_scaler"]
    ticker_to_idx  = lstm_result["ticker_to_idx"]

    if ticker_enc not in ticker_to_idx:
        raise ValueError(f"Ticker {ticker_enc} inconnu du modèle.")

    feat_df = select_features(recent_df.tail(seq_len), FEATURE_COLS)
    feat_df = feat_df.ffill().bfill().fillna(0)

    if len(feat_df) < seq_len:
        raise ValueError(f"Il faut au moins {seq_len} lignes pour l'inférence.")

    scaled  = feat_scaler.transform(feat_df.values).astype(np.float32)
    X_feat  = torch.from_numpy(scaled).unsqueeze(0).to(DEVICE)      # (1, seq, feat)
    X_tick  = torch.tensor([ticker_to_idx[ticker_enc]], dtype=torch.long).to(DEVICE)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_feat, X_tick).cpu().numpy()

    return target_scaler.inverse_transform(pred_scaled)[0]  # (horizon,)


def predict_hmm_next_state(hmm_result: dict) -> dict:
    model      = hmm_result["model"]
    last_state = hmm_result["last_state"]
    probs      = model.transmat_[last_state]
    next_state = int(np.argmax(probs))
    return {
        "next_state" : next_state,
        "label"      : hmm_result["state_labels"].get(next_state, "?"),
        "probs"      : {i: round(float(p), 4) for i, p in enumerate(probs)},
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("🚀 Démarrage du pipeline LSTM + HMM (v2 — modèle unique)")
    print(f"   Device    : {DEVICE}")
    print(f"   Séquence  : {SEQUENCE_LEN} jours  |  Horizon : {HORIZON} jours")
    print(f"   Embedding : dim={EMBED_DIM}")
    print(f"   Tickers   : {fill_df['Ticker'].nunique()} disponibles")

    results = run_combined_pipeline(fill_df, n_tickers=5)

    print("\n" + "="*60)
    print("  RÉSUMÉ FINAL")
    print("="*60)

    lstm = results["lstm"]
    print(f"\n  LSTM global → RMSE={lstm['rmse']:.4f} | MAE={lstm['mae']:.4f}")
    print(f"  Embedding appris pour {lstm['n_tickers']} tickers")

    for ticker_name, hmm_res in results["hmm"].items():
        if hmm_res:
            nxt = predict_hmm_next_state(hmm_res)
            print(f"\n  HMM [{ticker_name}] → prochain état : "
                  f"{nxt['next_state']} ({nxt['label']})")
            print(f"    Probabilités : {nxt['probs']}")

    print("\n✅ Pipeline terminé.")
