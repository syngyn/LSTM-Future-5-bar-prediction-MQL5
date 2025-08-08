#!/usr/bin/env python3
"""
ALFA / Transformer Backtest Data Generator
==========================================
Generates backtest_predictions.csv for the MT5 Strategy Tester.

This script:
1. Loads your trained ALFA or Transformer models and scalers.
2. Uses the historical EURUSD60.csv data your model was trained on.
3. Creates the exact same features as the training script.
4. Generates predictions for each historical bar.
5. Outputs a CSV file in the precise format required by the EA.
"""

import os
import pandas as pd
import numpy as np
import torch
import joblib
import math
from datetime import datetime, timedelta
import warnings
import traceback # <-- FIX: Added the missing traceback import

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
OUTPUT_FILE = "backtest_predictions.csv"

# --- MODEL SELECTION ---
# Choose between 'ALFA' (AttentionLSTM) and 'TRANSFORMER'
# IMPORTANT: This must match the model you trained.
SELECTED_MODEL = 'ALFA'

# Model parameters (must match your training script)
INPUT_FEATURES = 15
HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN = 128, 3, 20
OUTPUT_STEPS = 5
NUM_CLASSES = 3

# Backtest parameters
MIN_CONFIDENCE = 0.3  # Minimum confidence for realistic backtesting

# --- ATTENTION LSTM (ALFA) ARCHITECTURE ---
class AttentionLSTM(torch.nn.Module):
    """Enhanced LSTM with attention mechanism and uncertainty estimation"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_regression_outputs, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        self.input_norm = torch.nn.LayerNorm(input_size)
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=False)
        self.lstm_norm = torch.nn.LayerNorm(hidden_size)
        self.attention = torch.nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout, batch_first=True)
        self.attention_norm = torch.nn.LayerNorm(hidden_size)
        self.fusion = torch.nn.Sequential(torch.nn.Linear(hidden_size * 2, hidden_size), torch.nn.ReLU(), torch.nn.Dropout(dropout))
        self.regression_head = torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size // 2), torch.nn.ReLU(), torch.nn.Dropout(dropout * 0.5), torch.nn.Linear(hidden_size // 2, hidden_size // 4), torch.nn.ReLU(), torch.nn.Linear(hidden_size // 4, num_regression_outputs))
        self.classification_head = torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size // 2), torch.nn.ReLU(), torch.nn.Dropout(dropout * 0.5), torch.nn.Linear(hidden_size // 2, hidden_size // 4), torch.nn.ReLU(), torch.nn.Linear(hidden_size // 4, num_classes))
        self.uncertainty_head = torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size // 2), torch.nn.ReLU(), torch.nn.Linear(hidden_size // 2, num_regression_outputs))
        self.confidence_head = torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size // 4), torch.nn.ReLU(), torch.nn.Linear(hidden_size // 4, 1), torch.nn.Sigmoid())

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
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
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

class TransformerModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_regression_outputs, nhead=8, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.hidden_size = hidden_size
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        self.input_embedding = torch.nn.Linear(input_size, hidden_size)
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = torch.nn.Linear(hidden_size, hidden_size)
        self.regression_head = torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size // 2), torch.nn.ReLU(), torch.nn.Linear(hidden_size // 2, num_regression_outputs))
        self.classification_head = torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size // 2), torch.nn.ReLU(), torch.nn.Linear(hidden_size // 2, num_classes))
        self.confidence_head = torch.nn.Sequential(torch.nn.Linear(hidden_size, 1), torch.nn.Sigmoid())

    def forward(self, src):
        src = self.input_embedding(src) * math.sqrt(self.hidden_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output[:, -1, :]
        decoded_output = self.decoder(output)
        regression_output = self.regression_head(decoded_output)
        classification_logits = self.classification_head(decoded_output)
        model_confidence = self.confidence_head(decoded_output)
        dummy_uncertainty = torch.ones_like(regression_output) * 0.1
        dummy_attention_weights = None
        return regression_output, classification_logits, dummy_uncertainty, model_confidence, dummy_attention_weights

# --- UNIFIED FEATURE CREATION ---
def create_unified_features(df):
    """Creates a consistent 15-feature set for all models."""
    print("🔧 Creating unified features for backtest...")
    features_df = pd.DataFrame(index=df.index)
    
    # Ensure OHLC columns exist
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
    
    # Simplified Strength
    features_df['usd_strength'] = df['Close'].pct_change().rolling(5).mean()
    features_df['eur_strength'] = -df['Close'].pct_change().rolling(5).mean()
    features_df['jpy_strength'] = 0.0
    
    # Bollinger Band Width
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
    
    features_df = features_df[feature_names]
    aligned_main_df = df.loc[features_df.index].copy()
    
    print(f"✅ Created {len(features_df.columns)} features. Final data points: {len(features_df):,}")
    return aligned_main_df, features_df

class BacktestGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        
        self.model = None
        self.ensemble_models = []
        self.scaler_feature = None
        self.scaler_regressor_target = None
        self.model_type = None
        self.feature_names = None
        
        self._load_models()
        
    def _load_models(self):
        """Load trained models and scalers based on configuration."""
        print(f"🔄 Loading {SELECTED_MODEL} model and scalers...")
        
        try:
            scaler_feature_path = os.path.join(MODEL_DIR, "scaler.pkl")
            scaler_target_path = os.path.join(MODEL_DIR, "scaler_regression.pkl")
            model_path = os.path.join(MODEL_DIR, "lstm_model_regression.pth")
            
            self.scaler_feature = joblib.load(scaler_feature_path)
            self.scaler_regressor_target = joblib.load(scaler_target_path)
            print("✅ Scalers loaded successfully")
            
            if SELECTED_MODEL == 'ALFA':
                self.model = AttentionLSTM(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, OUTPUT_STEPS)
            elif SELECTED_MODEL == 'TRANSFORMER':
                self.model = TransformerModel(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, OUTPUT_STEPS)
            else:
                raise ValueError(f"Invalid model '{SELECTED_MODEL}' configured.")

            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.to(self.device).eval()
            self.model_type = SELECTED_MODEL
            print(f"✅ {self.model_type} model loaded successfully")

            if self.model_type == 'ALFA':
                self._load_ensemble_models()
            
        except Exception as e:
            print(f"💥 FATAL: Could not load models/scalers: {e}")
            raise
            
    def _load_ensemble_models(self):
        """Load ensemble models if available (for ALFA)."""
        ensemble_count = 0
        for i in range(1, 6):
            ensemble_path = os.path.join(MODEL_DIR, f"lstm_ensemble_{i}.pth")
            if os.path.exists(ensemble_path):
                try:
                    ensemble_model = AttentionLSTM(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, OUTPUT_STEPS)
                    checkpoint = torch.load(ensemble_path, map_location=self.device)
                    ensemble_model.load_state_dict(checkpoint['model_state'])
                    ensemble_model.to(self.device).eval()
                    self.ensemble_models.append(ensemble_model)
                    ensemble_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to load ensemble model {i}: {e}")
        
        if ensemble_count > 0:
            print(f"✅ Loaded {ensemble_count} ALFA ensemble models")
        else:
            print("ℹ️ No ensemble models found.")
            
    def download_data(self):
        """Load historical EURUSD data from the training dataset."""
        print("📊 Loading EURUSD data from training file...")
        data_path = os.path.join(SCRIPT_DIR, "EURUSD60.csv")
        
        if not os.path.exists(data_path):
            print(f"❌ EURUSD60.csv not found. Please place it in the script directory.")
            return None
        
        try:
            df = pd.read_csv(data_path)
            df.rename(columns=lambda x: x.strip().capitalize(), inplace=True)
            date_col = next((col for col in ['Date', 'Datetime', 'timestamp'] if col in df.columns), None)
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df.set_index(date_col, inplace=True)
            df.sort_index(inplace=True)
            if 'Tickvol' in df.columns and 'Volume' not in df.columns:
                df.rename(columns={'Tickvol': 'Volume'}, inplace=True)
            df = df.loc[~df.index.duplicated(keep='first')]
            print(f"✅ Loaded {len(df)} hourly bars from {df.index[0]} to {df.index[-1]}")
            return df
        except Exception as e:
            print(f"💥 Failed to load EURUSD60.csv: {e}")
            return None
    
    def create_sequences(self, main_df, features_df):
        """Create sequences for model prediction."""
        print("🔗 Creating sequences...")
        sequences, timestamps, current_prices = [], [], []
        
        # Get the aligned price data from the main dataframe
        price_data = main_df['Close']

        for i in range(SEQ_LEN, len(features_df)):
            sequence_data = features_df.iloc[i-SEQ_LEN:i].values
            sequences.append(sequence_data)
            timestamps.append(features_df.index[i])
            current_prices.append(price_data.iloc[i])
        
        sequences = np.array(sequences)
        print(f"✅ Created {len(sequences)} sequences of shape {sequences.shape}")
        return sequences, timestamps, current_prices
    
    def generate_predictions(self, sequences, current_prices):
        """Generate predictions for all sequences."""
        print(f"🎯 Generating predictions using {self.model_type} model...")
        all_predictions = []
        batch_size = 100
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            batch_current_prices = current_prices[i:i+batch_size]
            
            # Scale features
            flat_batch = batch_sequences.reshape(-1, INPUT_FEATURES)
            batch_scaled = self.scaler_feature.transform(flat_batch).reshape(batch_sequences.shape)
            batch_tensor = torch.tensor(batch_scaled, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                if self.ensemble_models: # Use ensemble if available
                    batch_preds = self._get_ensemble_predictions(batch_tensor, batch_current_prices)
                else: # Use single model
                    batch_preds = self._get_single_predictions(batch_tensor, batch_current_prices)
            all_predictions.extend(batch_preds)

        print(f"✅ Generated {len(all_predictions)} predictions")
        return all_predictions
    
    def _process_model_output(self, reg_out, class_logits, confidence, current_price):
        """Helper to denormalize and package model output."""
        predictions = reg_out.cpu().numpy()[0]
        classification_probs = torch.softmax(class_logits, dim=1)[0].cpu().numpy()
        model_confidence = float(confidence.cpu().numpy().item())
        
        try:
            unscaled_predictions = self.scaler_regressor_target.inverse_transform(predictions.reshape(1, -1))[0]
            if not all(0.5 < price < 2.5 for price in unscaled_predictions):
                raise ValueError("Unrealistic price")
        except Exception:
            atr_estimate = current_price * 0.002
            unscaled_predictions = [current_price + p * atr_estimate * 2.0 for p in predictions]
        
        sell_prob, hold_prob, buy_prob = classification_probs
        return {
            'predicted_prices': unscaled_predictions,
            'buy_prob': float(buy_prob), 'sell_prob': float(sell_prob), 'hold_prob': float(hold_prob),
            'confidence': max(MIN_CONFIDENCE, model_confidence)
        }

    def _get_single_predictions(self, batch_tensor, batch_current_prices):
        """Generate predictions for a batch using a single model."""
        batch_predictions = []
        reg_out, class_logits, _, confidence, _ = self.model(batch_tensor)
        for i in range(len(batch_tensor)):
            pred = self._process_model_output(reg_out[i:i+1], class_logits[i:i+1], confidence[i:i+1], batch_current_prices[i])
            batch_predictions.append(pred)
        return batch_predictions

    def _get_ensemble_predictions(self, batch_tensor, batch_current_prices):
        """Generate predictions for a batch using ensemble models (ALFA only)."""
        batch_predictions = []
        all_reg, all_class, all_conf = [], [], []
        models_to_use = [self.model] + self.ensemble_models

        for model in models_to_use:
            reg_out, class_logits, _, confidence, _ = model(batch_tensor)
            all_reg.append(reg_out.unsqueeze(0))
            all_class.append(torch.softmax(class_logits, dim=1).unsqueeze(0))
            all_conf.append(confidence.unsqueeze(0))
        
        avg_reg = torch.mean(torch.cat(all_reg, dim=0), dim=0)
        avg_class_logits = torch.log(torch.mean(torch.cat(all_class, dim=0), dim=0) + 1e-10) # Average probabilities, then back to logits
        avg_conf = torch.mean(torch.cat(all_conf, dim=0), dim=0)
        
        for i in range(len(batch_tensor)):
            pred = self._process_model_output(avg_reg[i:i+1], avg_class_logits[i:i+1], avg_conf[i:i+1], batch_current_prices[i])
            batch_predictions.append(pred)
        return batch_predictions
    
    def save_csv(self, timestamps, predictions):
        """Save predictions in MT5 backtest format."""
        print(f"💾 Saving predictions to {OUTPUT_FILE}...")
        rows = []
        for timestamp, pred in zip(timestamps, predictions):
            row = [timestamp.strftime("%Y.%m.%d %H:%M:%S"),
                   pred['buy_prob'], pred['sell_prob'], pred['hold_prob'], pred['confidence']]
            row.extend(pred['predicted_prices'])
            rows.append(row)
        
        columns = ['timestamp', 'buy_prob', 'sell_prob', 'hold_prob', 'confidence_score'] + \
                  [f'price_{i+1}' for i in range(OUTPUT_STEPS)]
        
        df = pd.DataFrame(rows, columns=columns)
        output_path = os.path.join(SCRIPT_DIR, OUTPUT_FILE)
        df.to_csv(output_path, sep=';', index=False, float_format='%.8f')
        print(f"✅ Saved {len(df)} predictions to: {output_path}")
        return output_path
    
    def generate_backtest_data(self):
        """Main function to generate complete backtest data."""
        print("\n" + "="*60)
        print(f"🚀 Generating backtest data using {self.model_type} model...")
        print("="*60)
        
        try:
            data = self.download_data()
            if data is None: 
                print("\n💥 FATAL ERROR: Could not load data.")
                return None
            
            # <-- FIX: Correctly call the standalone function
            main_df_aligned, features_df = create_unified_features(data)
            
            sequences, timestamps, current_prices = self.create_sequences(main_df_aligned, features_df)
            if len(sequences) == 0: raise ValueError("No valid sequences created.")
            
            predictions = self.generate_predictions(sequences, current_prices)
            if len(predictions) == 0: raise ValueError("No predictions generated.")
            
            output_path = self.save_csv(timestamps, predictions)
            
            print("\n" + "="*60 + "\n🎉 BACKTEST DATA GENERATION COMPLETE!\n" + "="*60)
            print(f"📁 Output file: {output_path}")
            print(f"📋 Next steps:")
            print(f"1. Copy '{OUTPUT_FILE}' to your MT5/Common/Files/ folder.")
            print(f"2. Run a backtest in the MT5 Strategy Tester with your EA.")
            
            return output_path
            
        except Exception as e:
            print(f"\n💥 FATAL ERROR: {e}")
            traceback.print_exc() # <-- FIX: This will now work
            return None

def main():
    """Main execution function."""
    print(f"🎯 ALFA / Transformer Backtest Data Generator v3.1")
    print(f"🔥 Selected Model for Backtesting: {SELECTED_MODEL}")
    print("="*50)
    
    try:
        generator = BacktestGenerator()
        result = generator.generate_backtest_data()
        if not result:
            print(f"\n❌ FAILED! Check error messages above.")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc() # <-- FIX: This will now work

if __name__ == "__main__":
    main()