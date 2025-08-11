#!/usr/bin/env python3
"""
Hybrid ALFA-Transformer Training Script v4.6 (Optimized 6-Pair Set)
====================================================================
- Reduces the set of currency pairs to the essential 6 required for
  the MQL5 EA's hard-coded feature engineering.
- This provides maximum efficiency while ensuring the script runs correctly.
- Includes the definitive positional fix for all data loading errors.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import joblib
import traceback
import warnings
import yfinance as yf
from datetime import datetime, timedelta
import glob

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DOWNLOAD_PATH = os.path.join(SCRIPT_DIR, "forex_data_h1")
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "models")

# Model & Training Parameters
INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN = 15, 128, 3, 20
OUTPUT_STEPS, NUM_CLASSES, EPOCHS, BATCH_SIZE, LR = 5, 3, 50, 64, 0.001
PROFIT_THRESHOLD_ATR = 0.75

# --- DATA DOWNLOADER (Optimized for 6 pairs) ---
# This is the minimal set of pairs REQUIRED by the feature engineering logic.
CURRENCY_PAIRS = [
    "GBPUSD=X",
    "USDJPY=X",
    "USDCAD=X",
    "USDCHF=X",
    "EURJPY=X",
    "EURGBP=X",
]

def download_all_data():
    """Downloads 1-hour data for the essential set of currency pairs."""
    print("🚀 Starting currency data download for the essential 6-pair set...")
    os.makedirs(DATA_DOWNLOAD_PATH, exist_ok=True)
    start_date = (datetime.now() - timedelta(days=729)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    print(f"Fetching data from {start_date} to {end_date}.")

    for pair in CURRENCY_PAIRS:
        clean_pair_name = pair.replace('=X', '')
        file_path = os.path.join(DATA_DOWNLOAD_PATH, f"{clean_pair_name}.csv")
        print(f"--- Downloading {clean_pair_name} ---")
        try:
            data = yf.download(tickers=pair, start=start_date, end=end_date, interval="1h", progress=False)
            if data.empty:
                print(f"⚠️ No data for {clean_pair_name}. Skipping.")
                continue
            data.to_csv(file_path)
            print(f"✅ Saved to {file_path}")
        except Exception as e:
            print(f"❌ Failed to download {clean_pair_name}: {e}")
    print("\n🎉 All downloads complete!")


# --- MODEL ARCHITECTURE and OTHER FUNCTIONS (No changes needed) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
class HybridModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_regression_outputs, nhead=8, dropout=0.2):
        super(HybridModel, self).__init__()
        self.hidden_size = hidden_size
        self.alfa_input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.lstm_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=nhead, dropout=dropout, batch_first=True)
        self.alfa_norm = nn.LayerNorm(hidden_size)
        self.transformer_input_embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.transformer_norm = nn.LayerNorm(hidden_size)
        self.fusion_layer = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.GELU(), nn.Dropout(dropout))
        self.regression_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, num_regression_outputs))
        self.classification_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, num_classes))
        self.uncertainty_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, num_regression_outputs))
        self.confidence_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
    def forward(self, x):
        alfa_x = self.alfa_input_norm(x); lstm_out, _ = self.lstm(alfa_x); lstm_out = self.lstm_norm(lstm_out)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out); alfa_features = self.alfa_norm(attn_out)
        transformer_x = self.transformer_input_embedding(x) * math.sqrt(self.hidden_size); transformer_x = self.pos_encoder(transformer_x)
        transformer_out = self.transformer_encoder(transformer_x); transformer_features = self.transformer_norm(transformer_out)
        alfa_last_step = alfa_features[:, -1, :]; transformer_last_step = transformer_features[:, -1, :]
        combined_features = torch.cat((alfa_last_step, transformer_last_step), dim=1); fused_representation = self.fusion_layer(combined_features)
        regression_output = self.regression_head(fused_representation); classification_logits = self.classification_head(fused_representation)
        uncertainty = torch.exp(self.uncertainty_head(fused_representation)); model_confidence = self.confidence_head(fused_representation)
        return regression_output, classification_logits, uncertainty, model_confidence, None
def create_mql5_features(main_df, all_data):
    print(f"🔧 Creating MQL5-aligned features for {main_df.name}...")
    features_df = pd.DataFrame(index=main_df.index)
    features_df['price_return'] = (main_df['Close'] / main_df['Close'].shift(1)) - 1.0; features_df['Volume'] = main_df['Volume']
    tr = pd.concat([main_df['High'] - main_df['Low'], abs(main_df['High'] - main_df['Close'].shift(1)), abs(main_df['Low'] - main_df['Close'].shift(1))], axis=1).max(axis=1)
    features_df['atr'] = tr.rolling(14).mean()
    ema12 = main_df['Close'].ewm(span=12, adjust=False).mean(); ema26 = main_df['Close'].ewm(span=26, adjust=False).mean()
    features_df['macd'] = ema12 - ema26
    delta = main_df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean(); loss = -delta.clip(upper=0).abs().rolling(window=14).mean()
    features_df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    low14, high14 = main_df['Low'].rolling(14).min(), main_df['High'].rolling(14).max()
    features_df['stoch_k'] = 100 * (main_df['Close'] - low14) / (high14 - low14 + 1e-10)
    tp = (main_df['High'] + main_df['Low'] + main_df['Close']) / 3
    tp_ma = tp.rolling(20).mean(); tp_md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    features_df['cci'] = (tp - tp_ma) / (0.015 * tp_md + 1e-10)
    features_df['hour'] = main_df.index.hour; features_df['day_of_week'] = main_df.index.dayofweek
    main_ret = features_df['price_return']
    uj_ret = all_data['USDJPY']['Close'].pct_change(); uc_ret = all_data['USDCAD']['Close'].pct_change()
    uchf_ret = all_data['USDCHF']['Close'].pct_change(); gu_ret = all_data['GBPUSD']['Close'].pct_change()
    ej_ret = all_data['EURJPY']['Close'].pct_change(); eg_ret = all_data['EURGBP']['Close'].pct_change()
    features_df['usd_strength_proxy'] = (uj_ret + uc_ret + uchf_ret) - (main_ret + gu_ret)
    features_df['eur_strength_proxy'] = (main_ret + ej_ret + eg_ret); features_df['jpy_strength_proxy'] = -(ej_ret + uj_ret)
    close_ma20 = main_df['Close'].rolling(20).mean(); bb_std = main_df['Close'].rolling(20).std()
    features_df['bb_width'] = (bb_std * 4) / (close_ma20 + 1e-10)
    features_df['volume_change'] = main_df['Volume'].diff(periods=5)
    body = abs(main_df['Close'] - main_df['Open']); price_range = main_df['High'] - main_df['Low']
    bar_type = pd.Series(0.0, index=main_df.index)
    bar_type[price_range > 0] = (body / price_range).apply(lambda x: 1.0 if x < 0.1 else 0.0)
    bar_type[(main_df['Close'] > main_df['Open']) & (main_df['Open'] < main_df['Open'].shift(1)) & (main_df['Close'] > main_df['Close'].shift(1))] = 2.0
    bar_type[(main_df['Close'] < main_df['Open']) & (main_df['Open'] > main_df['Open'].shift(1)) & (main_df['Close'] < main_df['Close'].shift(1))] = -2.0
    bar_type += (main_df['Open'] - main_df['Close'].shift(1)) / (features_df['atr'] + 1e-10)
    features_df['candle_type'] = bar_type
    features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()
    aligned_main_df = main_df.loc[features_df.index].copy()
    print(f"✅ Created {len(features_df.columns)} features. Final data points: {len(features_df):,}")
    return aligned_main_df, features_df
class UncertaintyLoss(nn.Module):
    def forward(self, pred, targ, unc): return torch.mean(0.5 * torch.exp(-unc) * (pred - targ)**2 + 0.5 * unc)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0): super().__init__(); self.alpha, self.gamma = alpha, gamma
    def forward(self, inp, targ):
        ce = F.cross_entropy(inp, targ, reduction='none'); pt = torch.exp(-ce)
        return torch.mean(self.alpha * (1 - pt)**self.gamma * ce)
def create_targets(main_df):
    print("🎯 Creating targets..."); regr_targets = [main_df['Close'].shift(-i) for i in range(1, OUTPUT_STEPS + 1)]
    regr_df = pd.concat(regr_targets, axis=1); regr_df.columns = [f'target_{i}' for i in range(OUTPUT_STEPS)]
    atr = (pd.concat([main_df['High'] - main_df['Low'], abs(main_df['High'] - main_df['Close'].shift(1)), abs(main_df['Low'] - main_df['Close'].shift(1))], axis=1).max(axis=1)).rolling(14).mean()
    future_price = main_df['Close'].shift(-OUTPUT_STEPS)
    conditions = [future_price > main_df['Close'] + atr * PROFIT_THRESHOLD_ATR, future_price < main_df['Close'] - atr * PROFIT_THRESHOLD_ATR]
    class_s = pd.Series(np.select(conditions, [2, 0], default=1), index=main_df.index, name='target_class')
    combined = main_df.join(regr_df).join(class_s).dropna()
    print(f"📊 Class distribution: {np.bincount(combined['target_class'].astype(int).values)}")
    return combined
def prepare_sequences(features_df, targets_df):
    print("🔗 Building sequences..."); common_index = features_df.index.intersection(targets_df.index)
    X = features_df.loc[common_index].values
    y_regr = targets_df.loc[common_index][[f'target_{i}' for i in range(OUTPUT_STEPS)]].values
    y_class = targets_df.loc[common_index]['target_class'].values
    f_scaler = StandardScaler().fit(X); t_scaler = StandardScaler().fit(y_regr)
    X_scaled, y_regr_scaled = f_scaler.transform(X), t_scaler.transform(y_regr)
    X_seq, y_r_seq, y_c_seq = [], [], []
    for i in range(len(X_scaled) - SEQ_LEN):
        X_seq.append(X_scaled[i:i + SEQ_LEN]); y_r_seq.append(y_regr_scaled[i + SEQ_LEN - 1]); y_c_seq.append(y_class[i + SEQ_LEN - 1])
    print(f"✅ Created {len(X_seq):,} sequences")
    return (torch.tensor(np.array(X_seq), dtype=torch.float32), torch.tensor(np.array(y_r_seq), dtype=torch.float32), torch.tensor(np.array(y_c_seq), dtype=torch.long), f_scaler, t_scaler)
def train_model(model, train_loader, val_loader, device, model_name="main"):
    print(f"\n🧠 Training Hybrid Model ({model_name})..."); regr_loss_fn, class_loss_fn = UncertaintyLoss(), FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR); scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    best_val_loss = float('inf'); best_model_state = None
    for epoch in range(EPOCHS):
        model.train(); train_loss = 0
        for X_b, y_r_b, y_c_b in train_loader:
            X_b, y_r_b, y_c_b = X_b.to(device), y_r_b.to(device), y_c_b.to(device)
            optimizer.zero_grad(); p_r, p_c, p_u, _, _ = model(X_b); loss = regr_loss_fn(p_r, y_r_b, p_u) + class_loss_fn(p_c, y_c_b)
            loss.backward(); optimizer.step(); train_loss += loss.item()
        model.eval(); val_loss = 0
        with torch.no_grad():
            for X_b, y_r_b, y_c_b in val_loader:
                X_b, y_r_b, y_c_b = X_b.to(device), y_r_b.to(device), y_c_b.to(device)
                p_r, p_c, p_u, _, _ = model(X_b); val_loss += (regr_loss_fn(p_r, y_r_b, p_u) + class_loss_fn(p_c, y_c_b)).item()
        val_loss /= len(val_loader); scheduler.step(val_loss)
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss; best_model_state = model.state_dict().copy(); print(f"   -> New best model found!")
    if best_model_state: model.load_state_dict(best_model_state)
    return model

# --- MAIN EXECUTION ---
def main():
    print("🚀 Hybrid ALFA-Transformer Training Script v4.6 (Optimized 6-Pair Set)")
    print("=" * 70)
    
    download_all_data()

    all_data = {}
    print("\nLoading all currency data into memory...")
    
    # +++ START OF DEFINITIVE FIX +++
    for pair_file in glob.glob(os.path.join(DATA_DOWNLOAD_PATH, "*.csv")):
        pair_name = os.path.basename(pair_file).replace('.csv', '')
        try:
            with open(pair_file, 'r') as f:
                f.readline(); second_line = f.readline().strip()

            if 'Ticker' in second_line or 'Price' in second_line:
                print(f"   -> Detected non-standard header for {pair_name}. Applying positional fix...")
                df = pd.read_csv(pair_file, header=None, skiprows=3)
                df.columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']
                df.set_index('Datetime', inplace=True)
                df.index = pd.to_datetime(df.index)
            else:
                df = pd.read_csv(pair_file, header=0, index_col=0, parse_dates=True)

            df.columns = [col.title() for col in df.columns]
            if 'Adj Close' in df.columns:
                df.rename(columns={'Adj Close': 'Close'}, inplace=True)

            required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
            if not required_cols.issubset(set(df.columns)):
                 raise ValueError(f"Could not find required columns. Available: {df.columns.tolist()}")

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

        except Exception as e:
            print(f"❌ Could not process file {pair_name}. Error: {e}")
            print("   This file might be corrupted or in an unhandled format. Skipping.")
            continue
        
        df.name = pair_name
        all_data[pair_name] = df
    # +++ END OF DEFINITIVE FIX +++
    
    # Check if all essential pairs were loaded successfully
    if len(all_data) < len(CURRENCY_PAIRS):
        print("💥 Not all essential data files were processed. Cannot proceed with feature creation. Exiting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 Using device: {device}")
    
    for currency_pair_name, main_df in all_data.items():
        print("\n" + "="*30 + f" Training for {currency_pair_name} " + "="*30)
        try:
            # Align all dataframes to the shortest common index to avoid errors
            # from mismatched data lengths during feature creation.
            common_index = main_df.index
            for name, df_to_align in all_data.items():
                common_index = common_index.intersection(df_to_align.index)
            
            # Create a temporary, aligned dictionary for feature creation
            aligned_all_data = {name: df.loc[common_index] for name, df in all_data.items()}
            main_df_aligned = main_df.loc[common_index]
            
            main_df_processed, features_df = create_mql5_features(main_df_aligned, aligned_all_data)
            targets_df = create_targets(main_df_processed)
            X, y_r, y_c, f_scaler, t_scaler = prepare_sequences(features_df, targets_df)
            
            if len(X) < 100:
                print(f"⚠️ Insufficient data for {currency_pair_name} after processing. Skipping.")
                continue
                
            train_dataset = TensorDataset(X, y_r, y_c); train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
            
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            model = HybridModel(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, OUTPUT_STEPS).to(device)
            model = train_model(model, train_loader, val_loader, device, model_name=currency_pair_name)
            
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"hybrid_model_{currency_pair_name}.pth"))
            joblib.dump(f_scaler, os.path.join(MODEL_SAVE_PATH, f"scaler_{currency_pair_name}.pkl"))
            joblib.dump(t_scaler, os.path.join(MODEL_SAVE_PATH, f"scaler_regression_{currency_pair_name}.pkl"))
            print(f"✅ Models and scalers for {currency_pair_name} saved successfully.")
        except Exception as e:
            print(f"❌ FAILED to train model for {currency_pair_name}. Error: {e}")
            traceback.print_exc()

    print("\n\n🎉🎉🎉 All training processes complete! 🎉🎉🎉")

if __name__ == "__main__":
    main()