#!/usr/bin/env python3
"""
ALFA / Transformer Training Script
==================================
Trains the selected model (Attention-based LSTM or Transformer).

Features:
- Model selection via global configuration
- AttentionLSTM (ALFA) with 8-head attention mechanism
- TransformerModel variant
- Enhanced loss functions and training techniques
- Unified feature creation for consistency
- Ensemble training support for the ALFA model
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import joblib
import sys
import traceback
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --- MODEL SELECTION ---
# Choose between 'ALFA' (AttentionLSTM) and 'TRANSFORMER'
# IMPORTANT: This must match the setting in daemon.py and generate_backtest.py
SELECTED_MODEL = 'ALFA'

# --- ATTENTION LSTM (ALFA) ARCHITECTURE ---
class AttentionLSTM(nn.Module):
    """Enhanced LSTM with attention mechanism and uncertainty estimation"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_regression_outputs, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=False)
        self.lstm_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.fusion = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Dropout(dropout))
        self.regression_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout * 0.5), nn.Linear(hidden_size // 2, hidden_size // 4), nn.ReLU(), nn.Linear(hidden_size // 4, num_regression_outputs))
        self.classification_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout * 0.5), nn.Linear(hidden_size // 2, hidden_size // 4), nn.ReLU(), nn.Linear(hidden_size // 4, num_classes))
        self.uncertainty_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, num_regression_outputs))
        self.confidence_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 4), nn.ReLU(), nn.Linear(hidden_size // 4, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        attn_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attention_norm(attn_out)
        combined = torch.cat([lstm_out, attn_out], dim=-1)
        fused = self.fusion(combined)
        last_hidden = fused[:, -1, :]
        avg_hidden = torch.mean(fused, dim=1)
        max_hidden, _ = torch.max(fused, dim=1)
        final_hidden = (last_hidden + avg_hidden + max_hidden) / 3
        regression_output = self.regression_head(final_hidden)
        classification_logits = self.classification_head(final_hidden)
        uncertainty = torch.exp(self.uncertainty_head(final_hidden))
        model_confidence = self.confidence_head(final_hidden)
        return regression_output, classification_logits, uncertainty, model_confidence, attention_weights

# --- TRANSFORMER MODEL ARCHITECTURE ---
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

class TransformerModel(nn.Module):
    """Transformer-based model for time series forecasting"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_regression_outputs, nhead=8, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.hidden_size = hidden_size
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        self.input_embedding = nn.Linear(input_size, hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, hidden_size)
        self.regression_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, num_regression_outputs))
        self.classification_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, num_classes))
        self.confidence_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, src):
        src = self.input_embedding(src) * math.sqrt(self.hidden_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output[:, -1, :] # Use the last output token for prediction
        decoded_output = self.decoder(output)
        
        regression_output = self.regression_head(decoded_output)
        classification_logits = self.classification_head(decoded_output)
        model_confidence = self.confidence_head(decoded_output)
        
        # Transformer doesn't inherently produce uncertainty, so we return a dummy tensor
        dummy_uncertainty = torch.ones_like(regression_output) * 0.1 # Low, constant uncertainty
        dummy_attention_weights = None # Not directly comparable to LSTM's attention
        
        return regression_output, classification_logits, dummy_uncertainty, model_confidence, dummy_attention_weights

# --- UNIFIED FEATURE CREATION ---
def create_unified_features(df):
    """Creates a consistent 15-feature set for all models."""
    print("🔧 Creating unified features...")
    features_df = pd.DataFrame(index=df.index)
    
    # Ensure OHLC columns exist, even if dummy
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            df[col] = df['Close'] if 'Close' in df.columns else 1.0

    # Price and Volume
    features_df['price_return'] = df['Close'].pct_change()
    features_df['Volume'] = df['Volume']
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features_df['atr'] = tr.rolling(14).mean()
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    features_df['macd'] = ema12 - ema26
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).abs().rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    features_df['rsi'] = 100 - (100 / (1 + rs))

    # Stochastic Oscillator (%K)
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    features_df['stoch_k'] = 100 * (df['Close'] - low14) / (high14 - low14 + 1e-10)
    
    # CCI
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    tp_ma = tp.rolling(20).mean()
    tp_md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    features_df['cci'] = (tp - tp_ma) / (0.015 * tp_md + 1e-10)
    
    # Time Features
    features_df['hour'] = df.index.hour
    features_df['day_of_week'] = df.index.dayofweek
    
    # Simplified Strength (placeholders, as MQL5 calculates them differently but this provides a signal)
    features_df['usd_strength'] = df['Close'].pct_change().rolling(5).mean()
    features_df['eur_strength'] = -df['Close'].pct_change().rolling(5).mean()
    features_df['jpy_strength'] = 0.0 # Placeholder
    
    # Bollinger Band Width (use bb_width to avoid name clash with EA)
    bb_std = df['Close'].rolling(20).std()
    features_df['bb_width'] = bb_std / (df['Close'].rolling(20).mean() + 1e-10)

    # Volume Change
    features_df['volume_change'] = df['Volume'].pct_change(periods=5)
    
    # Candle Type
    body = abs(df['Close'] - df['Open'])
    range_size = df['High'] - df['Low']
    features_df['candle_type'] = (body / (range_size + 1e-10))
    
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df.dropna(inplace=True)
    
    feature_names = [
        'price_return', 'Volume', 'atr', 'macd', 'rsi', 'stoch_k', 'cci',
        'hour', 'day_of_week', 'usd_strength', 'eur_strength', 'jpy_strength',
        'bb_width', 'volume_change', 'candle_type'
    ]
    
    # Ensure final feature order and count, and align the original dataframe
    features_df = features_df[feature_names]
    aligned_main_df = df.loc[features_df.index].copy()
    
    print(f"✅ Created {len(features_df.columns)} features. Final data points: {len(features_df):,}")
    return aligned_main_df, features_df, feature_names

# --- ENHANCED LOSS FUNCTIONS ---
class UncertaintyLoss(nn.Module):
    def __init__(self): super(UncertaintyLoss, self).__init__()
    def forward(self, predictions, targets, uncertainties):
        return torch.mean((predictions - targets) ** 2 / (2 * uncertainties + 1e-6) + 0.5 * torch.log(uncertainties + 1e-6))

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return torch.mean(self.alpha * (1 - pt) ** self.gamma * ce_loss)

# --- TRAINING CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "models")
DATA_DIR = SCRIPT_DIR
DATA_FILES = ["EURUSD60.csv", "EURUSD.csv", "eurusd_h1.csv"]
INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN = 15, 128, 3, 20
OUTPUT_STEPS, NUM_CLASSES, EPOCHS, BATCH_SIZE, LEARNING_RATE = 5, 3, 50, 64, 0.001
LOOKAHEAD_BARS, PROFIT_THRESHOLD_ATR = 5, 0.75
USE_ROBUST_SCALING = True
CREATE_ENSEMBLE = True # Only applies if SELECTED_MODEL is 'ALFA'

def load_and_prepare_data():
    """Load and prepare training data."""
    data_file = None
    for filename in DATA_FILES:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            data_file = filepath
            break
    if not data_file:
        print(f"❌ No data file found. Looking for: {DATA_FILES}")
        return None, None
    
    print(f"📁 Using data file: {os.path.basename(data_file)}")
    df = pd.read_csv(data_file)
    
    # Handle various possible date column names and formats
    date_col = next((col for col in ['Date', 'Datetime', 'timestamp'] if col in df.columns), None)
    if not date_col:
        print("❌ Date column not found in CSV.")
        return None, None
        
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df.dropna(subset=[date_col], inplace=True)
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)
    
    # Handle various column name conventions from MT5
    df.rename(columns=lambda x: x.strip().capitalize(), inplace=True)
    if 'Tickvol' in df.columns and 'Volume' not in df.columns:
        df.rename(columns={'Tickvol': 'Volume'}, inplace=True)

    main_df, features_df, feature_names = create_unified_features(df)
    
    if main_df is None or len(main_df) < SEQ_LEN + OUTPUT_STEPS:
        print(f"💥 Insufficient data: {len(main_df) if main_df is not None else 0} rows")
        return None, None
    
    return main_df, features_df, feature_names

def create_targets(main_df):
    """Create regression and classification targets."""
    print("🎯 Creating targets...")
    price_col = 'Close'
    
    regr_targets = []
    for i in range(1, OUTPUT_STEPS + 1):
        regr_targets.append(main_df[price_col].shift(-i))
    regr_target_df = pd.concat(regr_targets, axis=1)
    regr_target_df.columns = [f'target_regr_{i}' for i in range(OUTPUT_STEPS)]
    
    high_low = main_df['High'] - main_df['Low']
    high_close = abs(main_df['High'] - main_df[price_col].shift(1))
    low_close = abs(main_df['Low'] - main_df[price_col].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()
    
    future_price = main_df[price_col].shift(-LOOKAHEAD_BARS)
    atr_threshold = atr * PROFIT_THRESHOLD_ATR
    
    conditions = [
        future_price > main_df[price_col] + atr_threshold,
        future_price < main_df[price_col] - atr_threshold
    ]
    choices = [2, 0]  # 2=Buy, 0=Sell
    class_target_s = pd.Series(np.select(conditions, choices, default=1), index=main_df.index, name='target_class')
    
    combined_df = main_df.join(regr_target_df).join(class_target_s)
    combined_df.dropna(inplace=True)
    
    print(f"📊 Class distribution: {np.bincount(combined_df['target_class'].astype(int).values)}")
    print(f"📈 Final training data: {len(combined_df):,} samples")
    return combined_df

def prepare_sequences(features_df, targets_df, feature_names):
    """Prepare training sequences."""
    print("🔗 Building sequences...")
    X = features_df[feature_names].values
    y_regr = targets_df[[f'target_regr_{i}' for i in range(OUTPUT_STEPS)]].values
    y_class = targets_df['target_class'].values
    
    scaler_class = RobustScaler if USE_ROBUST_SCALING else StandardScaler
    feature_scaler = scaler_class()
    X_scaled = feature_scaler.fit_transform(X)
    
    target_scaler = StandardScaler()
    y_regr_scaled = target_scaler.fit_transform(y_regr)
    
    X_seq, y_regr_seq, y_class_seq = [], [], []
    for i in range(len(X_scaled) - SEQ_LEN):
        X_seq.append(X_scaled[i:i + SEQ_LEN])
        y_regr_seq.append(y_regr_scaled[i + SEQ_LEN - 1])
        y_class_seq.append(y_class[i + SEQ_LEN - 1])
    
    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_regr_tensor = torch.tensor(np.array(y_regr_seq), dtype=torch.float32)
    y_class_tensor = torch.tensor(np.array(y_class_seq), dtype=torch.long)
    
    print(f"✅ Created {len(X_tensor):,} sequences")
    return X_tensor, y_regr_tensor, y_class_tensor, feature_scaler, target_scaler

def create_data_loaders(X_tensor, y_regr_tensor, y_class_tensor):
    train_size = int(0.8 * len(X_tensor))
    train_dataset = TensorDataset(X_tensor[:train_size], y_regr_tensor[:train_size], y_class_tensor[:train_size])
    val_dataset = TensorDataset(X_tensor[train_size:], y_regr_tensor[train_size:], y_class_tensor[train_size:])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"📊 Train: {len(train_dataset):,}, Validation: {len(val_dataset):,}")
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, device, model_type="ALFA", model_name="main"):
    print(f"🎯 Training {model_type} {model_name} model...")
    regr_criterion = UncertaintyLoss() if model_type == "ALFA" else nn.MSELoss()
    class_criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    best_val_loss, patience, max_patience, best_model_state = float('inf'), 0, 10, None
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_regr_batch, y_class_batch in train_loader:
            X_batch, y_regr_batch, y_class_batch = X_batch.to(device), y_regr_batch.to(device), y_class_batch.to(device)
            optimizer.zero_grad()
            pred_regr, pred_class, uncertainty, confidence, _ = model(X_batch)
            regr_loss = regr_criterion(pred_regr, y_regr_batch, uncertainty) if model_type == "ALFA" else regr_criterion(pred_regr, y_regr_batch)
            class_loss = class_criterion(pred_class, y_class_batch)
            total_loss = regr_loss + class_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += total_loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_regr_batch, y_class_batch in val_loader:
                X_batch, y_regr_batch, y_class_batch = X_batch.to(device), y_regr_batch.to(device), y_class_batch.to(device)
                pred_regr, pred_class, _, _, _ = model(X_batch)
                v_regr_loss = nn.MSELoss()(pred_regr, y_regr_batch)
                v_class_loss = nn.CrossEntropyLoss()(pred_class, y_class_batch)
                val_loss += (v_regr_loss + v_class_loss).item()
        
        avg_train_loss, avg_val_loss = train_loss / len(train_loader), val_loss / len(val_loader)
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss, best_model_state, patience = avg_val_loss, model.state_dict().copy(), 0
        else:
            patience += 1
        if patience >= max_patience:
            print(f"⏹️  Early stopping at epoch {epoch+1}")
            break
            
    if best_model_state: model.load_state_dict(best_model_state)
    return model

def save_models_and_scalers(model, feature_scaler, target_scaler, ensemble_models=None):
    print("💾 Saving models and scalers...")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    torch.save({"model_state": model.state_dict()}, os.path.join(MODEL_SAVE_PATH, "lstm_model_regression.pth"))
    if ensemble_models:
        for i, em in enumerate(ensemble_models):
            torch.save({"model_state": em.state_dict()}, os.path.join(MODEL_SAVE_PATH, f"lstm_ensemble_{i+1}.pth"))
            
    joblib.dump(target_scaler, os.path.join(MODEL_SAVE_PATH, "scaler_regression.pkl"))
    joblib.dump(feature_scaler, os.path.join(MODEL_SAVE_PATH, "scaler.pkl"))
    print("✅ Models and scalers saved successfully.")

def main():
    print(f"🚀 ALFA / Transformer Training Script v3.1")
    print(f"🔥 Selected Model: {SELECTED_MODEL}")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")
    
    main_df, features_df, feature_names = load_and_prepare_data()
    if main_df is None: return
    
    combined_df = create_targets(main_df)
    if combined_df is None: return

    # Align features with the final combined dataframe index
    features_df = features_df.loc[combined_df.index]
    
    X_tensor, y_regr_tensor, y_class_tensor, feature_scaler, target_scaler = prepare_sequences(features_df, combined_df, feature_names)
    train_loader, val_loader = create_data_loaders(X_tensor, y_regr_tensor, y_class_tensor)
    
    print(f"\n" + "="*60 + f"\n🎯 TRAINING MAIN {SELECTED_MODEL} MODEL\n" + "="*60)
    
    if SELECTED_MODEL == 'ALFA':
        model = AttentionLSTM(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, OUTPUT_STEPS).to(device)
    elif SELECTED_MODEL == 'TRANSFORMER':
        model = TransformerModel(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, OUTPUT_STEPS).to(device)
    else:
        raise ValueError(f"Invalid model '{SELECTED_MODEL}' configured.")
        
    model = train_model(model, train_loader, val_loader, device, model_type=SELECTED_MODEL, model_name="main")
    
    ensemble_models = []
    if CREATE_ENSEMBLE and SELECTED_MODEL == 'ALFA':
        print(f"\n" + "="*60 + "\n🎭 TRAINING ENSEMBLE MODELS (ALFA ONLY)\n" + "="*60)
        for i in range(3):
            print(f"\n🎭 Training ensemble model {i+1}/3...")
            ensemble_model = AttentionLSTM(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, OUTPUT_STEPS, dropout=0.2 + i * 0.05).to(device)
            ensemble_model = train_model(ensemble_model, train_loader, val_loader, device, "ALFA", f"ensemble_{i+1}")
            ensemble_models.append(ensemble_model)
    
    save_models_and_scalers(model, feature_scaler, target_scaler, ensemble_models)
    
    print(f"\n" + "="*70 + "\n🎉 TRAINING COMPLETE!\n" + "="*70)
    print(f"📊 Model type trained: {SELECTED_MODEL}")
    if ensemble_models: print(f"🎭 Ensemble models created: {len(ensemble_models)}")
    print(f"💾 Models saved to: {MODEL_SAVE_PATH}")
    print(f"\n🚀 Next steps:")
    print(f"1. Ensure daemon.py has SELECTED_MODEL = '{SELECTED_MODEL}'")
    print(f"2. Run the daemon: python daemon.py")
    print(f"3. Load your EA in MT5")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        traceback.print_exc()