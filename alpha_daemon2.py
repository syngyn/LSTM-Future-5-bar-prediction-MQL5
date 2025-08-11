#!/usr/bin/env python3
"""
Hybrid ALFA-Transformer Prediction Daemon v1.0
==============================================
Listens for requests from the MQL5 EA, processes features
through a trained PyTorch model, and returns predictions.
"""
import os
import json
import time
import glob
import traceback
import torch
import torch.nn.functional as F
import joblib
import numpy as np
import math
import torch.nn as nn

# --- CONFIGURATION ---

# More robust way to find the MT5 Common/Files directory on Windows
try:
    # This is the standard Windows path
    APPDATA_PATH = os.environ['APPDATA']
    TERMINAL_COMMON_PATH = os.path.join(APPDATA_PATH, "MetaQuotes", "Terminal", "Common", "Files")
except KeyError:
    # Fallback for non-Windows or unusual setups
    print("⚠️ Could not find APPDATA environment variable. Using default user path.")
    TERMINAL_COMMON_PATH = os.path.expanduser("~/AppData/Roaming/MetaQuotes/Terminal/Common/Files")

# This path must match the MQL5 EA's #define DATA_FOLDER
DATA_FOLDER = os.path.join(TERMINAL_COMMON_PATH, "LSTM_Trading", "data")

# Path to the trained models, relative to this script's location
MODEL_PATH = "models" 

# Model parameters (must match the training script)
SEQ_LEN = 20
FEATURE_COUNT = 15
PREDICTION_STEPS = 5

# --- MODEL ARCHITECTURE (Must be defined to load model weights) ---
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


# --- DYNAMIC MODEL LOADER WITH CACHING ---
# Global dictionary to hold loaded models to avoid reloading from disk
MODEL_CACHE = {}

def load_model_and_scalers(symbol):
    """
    Dynamically loads a model and its scalers for a given symbol.
    Caches the loaded objects in memory for fast reuse.
    """
    if symbol in MODEL_CACHE:
        # Return the cached model if it's already loaded
        return MODEL_CACHE[symbol]

    print(f"🧠 Caching new model for '{symbol}'...")
    model_file = os.path.join(MODEL_PATH, f"hybrid_model_{symbol}.pth")
    scaler_file = os.path.join(MODEL_PATH, f"scaler_{symbol}.pkl")
    reg_scaler_file = os.path.join(MODEL_PATH, f"scaler_regression_{symbol}.pkl")

    if not all(os.path.exists(f) for f in [model_file, scaler_file, reg_scaler_file]):
        print(f"❌ Error: Model or scaler files for '{symbol}' not found in '{MODEL_PATH}'")
        return None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model parameters must match the training script
    model = HybridModel(input_size=15, hidden_size=128, num_layers=3, num_classes=3, num_regression_outputs=5).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    feature_scaler = joblib.load(scaler_file)
    regression_scaler = joblib.load(reg_scaler_file)
    
    # Store the loaded objects in the cache
    MODEL_CACHE[symbol] = (model, feature_scaler, regression_scaler)
    
    print(f"✅ Model and scalers for '{symbol}' loaded and cached successfully on {device}.")
    return model, feature_scaler, regression_scaler


def process_request(file_path):
    """Processes a single request file and generates a response."""
    request_id = os.path.basename(file_path).replace("request_", "").replace(".json", "")
    response_file = os.path.join(DATA_FOLDER, f"response_{request_id}.json")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Get the symbol from the request to load the correct model
        symbol = data.get('symbol')
        if not symbol:
            raise ValueError("Request JSON is missing the 'symbol' key.")

        # Load the appropriate model (or get it from the cache)
        model, f_scaler, t_scaler = load_model_and_scalers(symbol)
        if model is None:
            raise FileNotFoundError(f"Could not load model for symbol: {symbol}")

        # The EA sends a flat list of 300 features
        features_flat = np.array(data['features'], dtype=np.float32)
        if features_flat.shape[0] != SEQ_LEN * FEATURE_COUNT:
            raise ValueError(f"Expected {SEQ_LEN * FEATURE_COUNT} features, got {features_flat.shape[0]}")
            
        features = features_flat.reshape(SEQ_LEN, FEATURE_COUNT)

        # Scale features
        scaled_features = f_scaler.transform(features)
        
        # Create a batch of 1 and send to the correct device
        device = next(model.parameters()).device
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            p_r, p_c, _, p_conf, _ = model(input_tensor)
        
        # Process predictions
        predicted_prices = t_scaler.inverse_transform(p_r.cpu().numpy()).flatten().tolist()
        probabilities = F.softmax(p_c.cpu(), dim=1).flatten().tolist()
        sell_prob, hold_prob, buy_prob = probabilities[0], probabilities[1], probabilities[2]
        confidence_score = p_conf.cpu().item()

        # Create response
        response_data = {
            "status": "ok",
            "request_id": request_id,
            "predicted_prices": predicted_prices,
            "confidence_score": confidence_score,
            "buy_probability": buy_prob,
            "sell_probability": sell_prob,
            "hold_probability": hold_prob
        }
    
    except Exception as e:
        print(f"❌ Error processing {request_id}: {e}")
        traceback.print_exc()
        response_data = {"status": "error", "message": str(e), "request_id": request_id}

    # Write response file
    with open(response_file, 'w') as f:
        json.dump(response_data, f)
    
    # Clean up request file
    os.remove(file_path)
    print(f"Processed request {request_id}")


def main():
    """Main daemon loop."""
    print("🧠 Hybrid Prediction Daemon v1.1 (Dynamic & Fixed)")
    print("="*50)
    print(f"Watching for requests in: {DATA_FOLDER}")
    
    if not os.path.exists(DATA_FOLDER):
        print(f"⚠️ Data folder not found. Creating it...")
        os.makedirs(DATA_FOLDER)
    
    print("\n🚀 Daemon is running. Waiting for EA requests... (Press Ctrl+C to stop)")
    while True:
        try:
            request_files = glob.glob(os.path.join(DATA_FOLDER, "request_*.json"))
            if request_files:
                for req_file in request_files:
                    process_request(req_file)
            time.sleep(0.1)  # Sleep for 100ms to prevent high CPU usage
        except KeyboardInterrupt:
            print("\n🛑 Daemon stopped by user.")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()