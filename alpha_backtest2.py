#!/usr/bin/env python3
"""
Hybrid ALFA-Transformer Backtesting File Generator v4.6 (Dynamic File Fix)
===========================================================================
- Fixes the "FATAL: Missing required data file" error.
- The list of required files is now built dynamically based on the
  MODEL_SYMBOL, ensuring it only looks for files that were actually
  downloaded by the optimized training script.
- Includes the definitive positional fix for all data loading errors.
"""
import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import traceback
import math
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---

# 1. SET THE SYMBOL YOU WANT TO BACKTEST HERE
# IMPORTANT: This symbol MUST be one of the 6 pairs from the training script.
# (GBPUSD, USDJPY, USDCAD, USDCHF, EURJPY, EURGBP)
MODEL_SYMBOL = "GBPUSD" 

# 2. VERIFY THESE PATHS
try:
    TERMINAL_COMMON_PATH = os.path.join(os.environ['APPDATA'], "MetaQuotes", "Terminal", "Common", "Files")
except KeyError:
    TERMINAL_COMMON_PATH = os.path.expanduser("~/.wine/drive_c/users/user/AppData/Roaming/MetaQuotes/Terminal/Common/Files")

OUTPUT_FILE_PATH = os.path.join(TERMINAL_COMMON_PATH, "backtest_predictions.csv")
MODEL_PATH = "models"
DATA_DOWNLOAD_PATH = "forex_data_h1"

# --- MODEL AND DATA PARAMETERS (Must match training) ---
SEQ_LEN = 20
FEATURE_COUNT = 15
BATCH_SIZE = 512

# ================================================================= #
#  BELOW THIS LINE, NO MODIFICATIONS ARE GENERALLY NEEDED           #
# ================================================================= #

# --- 1. HYBRID MODEL ARCHITECTURE (Copied from training script) ---
# ... (The model architecture code is unchanged) ...
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
        self.hidden_size = hidden_size; self.alfa_input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.lstm_norm = nn.LayerNorm(hidden_size); self.attention = nn.MultiheadAttention(hidden_size, num_heads=nhead, dropout=dropout, batch_first=True)
        self.alfa_norm = nn.LayerNorm(hidden_size); self.transformer_input_embedding = nn.Linear(input_size, hidden_size)
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

# --- 2. MQL5-ALIGNED FEATURE CREATION (Copied from training script) ---
def create_mql5_features(main_df, all_data, pair_name):
    # This function is identical to the one in the training script
    print(f"🔧 Creating MQL5-aligned features for {pair_name}...")
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
    print(f"✅ Created {len(features_df.columns)} features. Final data points: {len(features_df):,}")
    return features_df

# --- 3. MAIN EXECUTION SCRIPT ---
def run_backtest_generator():
    """The main function to generate the backtesting file."""
    print("🚀 Backtesting File Generator v4.6 (Dynamic File Fix)")
    print("="*60)

    # --- Step 1: Load all historical data required for features ---
    all_data = {}
    print("\n[Step 1/6] Loading all currency data for feature engineering...")
    
    # +++ START OF FIX +++
    # Dynamically build the list of required pairs based on the MODEL_SYMBOL.
    # These are the base pairs needed for the strength formulas.
    base_required_pairs = {"GBPUSD", "USDJPY", "USDCAD", "USDCHF", "EURJPY", "EURGBP"}
    # Add the symbol we are testing to the set.
    required_pairs = base_required_pairs.union({MODEL_SYMBOL})
    print(f"   -> Needing data for: {sorted(list(required_pairs))}")
    # +++ END OF FIX +++

    for pair_name in required_pairs:
        file_path = os.path.join(DATA_DOWNLOAD_PATH, f"{pair_name}.csv")
        if not os.path.exists(file_path):
            print(f"❌ FATAL: Missing required data file: {file_path}")
            print(f"   Please ensure the training script has been run and downloaded data for {pair_name}.")
            return
            
        try:
            with open(file_path, 'r') as f:
                f.readline(); second_line = f.readline().strip()
            
            if 'Ticker' in second_line or 'Price' in second_line:
                print(f"   -> Detected non-standard header for {pair_name}. Applying positional fix...")
                df = pd.read_csv(file_path, header=None, skiprows=3)
                df.columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']
                df.set_index('Datetime', inplace=True)
                df.index = pd.to_datetime(df.index)
            else:
                df = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True)

            df.columns = [col.title() for col in df.columns]
            if 'Adj Close' in df.columns:
                df.rename(columns={'Adj Close': 'Close'}, inplace=True)

            required_cols_check = {'Open', 'High', 'Low', 'Close', 'Volume'}
            if not required_cols_check.issubset(set(df.columns)):
                 raise ValueError(f"Could not find required columns. Available: {df.columns.tolist()}")

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
            
        except Exception as e:
            print(f"❌ FATAL: Could not process required file {pair_name}. Error: {e}")
            return

        all_data[pair_name] = df
    print("✅ All required data loaded successfully.")

    # --- Step 2: Load the specific trained model and scalers ---
    print(f"\n[Step 2/6] Loading model and scalers for {MODEL_SYMBOL}...")
    model_file = os.path.join(MODEL_PATH, f"hybrid_model_{MODEL_SYMBOL}.pth")
    scaler_file = os.path.join(MODEL_PATH, f"scaler_{MODEL_SYMBOL}.pkl")
    reg_scaler_file = os.path.join(MODEL_PATH, f"scaler_regression_{MODEL_SYMBOL}.pkl")

    if not all(os.path.exists(f) for f in [model_file, scaler_file, reg_scaler_file]):
        print(f"❌ FATAL: Model or scaler files for '{MODEL_SYMBOL}' not found in '{MODEL_PATH}'.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridModel(input_size=15, hidden_size=128, num_layers=3, num_classes=3, num_regression_outputs=5).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    f_scaler = joblib.load(scaler_file)
    t_scaler = joblib.load(reg_scaler_file)
    print(f"✅ Model for {MODEL_SYMBOL} loaded onto {device}.")

    # --- Step 3: Generate features for the entire history ---
    print(f"\n[Step 3/6] Generating features for the entire {MODEL_SYMBOL} dataset...")
    main_df = all_data[MODEL_SYMBOL]
    try:
        common_index = main_df.index
        for df_to_align in all_data.values():
            common_index = common_index.intersection(df_to_align.index)
        
        main_df_aligned = main_df.loc[common_index]
        aligned_all_data = {name: df.loc[common_index] for name, df in all_data.items()}

        features_df = create_mql5_features(main_df_aligned, aligned_all_data, MODEL_SYMBOL)
        print(f"   -> Data aligned. Final feature count: {len(features_df)}")
    except Exception as e:
        print(f"❌ FATAL: Failed to create features. Error: {e}")
        traceback.print_exc()
        return

    # --- Step 4: Create sequences from the features ---
    print("\n[Step 4/6] Building sequences for the entire dataset...")
    X = features_df.values; X_scaled = f_scaler.transform(X)
    X_seq = []; sequence_timestamps = [] 
    for i in range(len(X_scaled) - SEQ_LEN + 1):
        X_seq.append(X_scaled[i : i + SEQ_LEN])
        sequence_timestamps.append(features_df.index[i + SEQ_LEN - 1])

    X_seq_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
    seq_dataset = TensorDataset(X_seq_tensor)
    seq_loader = DataLoader(seq_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"✅ Created {len(X_seq):,} sequences to predict.")

    # --- Step 5: Iterate and get all predictions ---
    print("\n[Step 5/6] Generating all predictions... (This may take a moment)")
    all_predictions = []
    with torch.no_grad():
        for i, (X_batch,) in enumerate(seq_loader):
            X_batch = X_batch.to(device)
            p_r, p_c, _, p_conf, _ = model(X_batch)
            prices_unscaled = t_scaler.inverse_transform(p_r.cpu().numpy())
            probabilities = F.softmax(p_c.cpu(), dim=1).numpy()
            conf_scores = p_conf.cpu().numpy()
            for j in range(X_batch.size(0)):
                global_idx = i * BATCH_SIZE + j
                record = {
                    'timestamp': sequence_timestamps[global_idx],
                    'buy_prob': probabilities[j, 2], 'sell_prob': probabilities[j, 0],
                    'hold_prob': probabilities[j, 1], 'confidence_score': conf_scores[j, 0],
                    'predicted_prices_0': prices_unscaled[j, 0], 'predicted_prices_1': prices_unscaled[j, 1],
                    'predicted_prices_2': prices_unscaled[j, 2], 'predicted_prices_3': prices_unscaled[j, 3],
                    'predicted_prices_4': prices_unscaled[j, 4],
                }
                all_predictions.append(record)
    print("✅ All predictions generated.")

    # --- Step 6: Save results to the CSV file required by the EA ---
    print(f"\n[Step 6/6] Saving {len(all_predictions)} predictions to CSV...")
    if not all_predictions:
        print("❌ No predictions were generated. Cannot create file.")
        return
        
    results_df = pd.DataFrame(all_predictions)
    results_df['timestamp'] = results_df['timestamp'].dt.strftime('%Y.%m.%d %H:%M:%S')
    
    try:
        os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
        results_df.to_csv(OUTPUT_FILE_PATH, sep=';', index=False, header=True, float_format='%.8f')
        print("\n" + "="*60)
        print("🎉 SUCCESS! Backtesting file created.")
        print(f"   Location: {OUTPUT_FILE_PATH}")
        print("   You can now run the EA in the MT5 Strategy Tester.")
        print("="*60)
    except Exception as e:
        print(f"❌ FATAL: Failed to write the final CSV file.")
        print(f"   Error: {e}\n   Please check permissions for the path: {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    run_backtest_generator()