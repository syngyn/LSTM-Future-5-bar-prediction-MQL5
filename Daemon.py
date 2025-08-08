import os
import json
import time
import torch
import joblib
import numpy as np
import math
import traceback
from torch import nn
from datetime import datetime
import sys

# --- MODEL SELECTION ---
# Choose between 'ALFA' (AttentionLSTM) and 'TRANSFORMER'
# IMPORTANT: This must match the setting in train_enhanced_model.py
SELECTED_MODEL = 'ALFA'

# --- Configuration ---
def find_mql5_files_path():
    """Finds the MQL5 'Files' directory on Windows."""
    appdata = os.getenv('APPDATA')
    if not appdata or 'win' not in sys.platform:
        return None
    metaquotes_path = os.path.join(appdata, 'MetaQuotes', 'Terminal')
    if not os.path.isdir(metaquotes_path):
        return None
    for entry in os.listdir(metaquotes_path):
        terminal_path = os.path.join(metaquotes_path, entry)
        if os.path.isdir(terminal_path) and len(entry) > 30:
            mql5_files_path = os.path.join(terminal_path, 'MQL5', 'Files')
            if os.path.isdir(mql5_files_path):
                return mql5_files_path
    return None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
COMM_DIR_BASE = find_mql5_files_path() or SCRIPT_DIR
DATA_DIR = os.path.join(COMM_DIR_BASE, "LSTM_Trading", "data")
print(f"--> Using Model Path: {MODEL_DIR}")
print(f"--> Using Communication Path: {DATA_DIR}")

# --- Model File Paths ---
MODEL_FILE = os.path.join(MODEL_DIR, "lstm_model_regression.pth")
SCALER_FILE_FEATURE = os.path.join(MODEL_DIR, "scaler.pkl")
SCALER_FILE_REGRESSION_TARGET = os.path.join(MODEL_DIR, "scaler_regression.pkl")

# --- Constants ---
INPUT_FEATURES = 15
HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN = 128, 3, 20
POLL_INTERVAL = 0.1
OUTPUT_STEPS = 5
NUM_CLASSES = 3

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


# --- DAEMON CLASS ---
class AlfaTransformerDaemon:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        
        self.model = None
        self.model_type = None
        self.scaler_feature = None
        self.scaler_regressor_target = None
        self.prediction_history = []
        self.market_conditions = {"volatility": "normal", "trend": "neutral"}
        
        self._load_models()

    def _load_models(self):
        """Load the selected model and scalers."""
        print(f"🔄 Loading {SELECTED_MODEL} model and scalers...")
        try:
            # Load scalers first
            self.scaler_feature = joblib.load(SCALER_FILE_FEATURE)
            self.scaler_regressor_target = joblib.load(SCALER_FILE_REGRESSION_TARGET)
            print("✅ Scalers loaded successfully")
            
            # Instantiate the selected model architecture
            if SELECTED_MODEL == 'ALFA':
                self.model = AttentionLSTM(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, OUTPUT_STEPS)
            elif SELECTED_MODEL == 'TRANSFORMER':
                self.model = TransformerModel(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, OUTPUT_STEPS)
            else:
                raise ValueError(f"Invalid model '{SELECTED_MODEL}' configured in daemon.py.")
            
            # Load the trained model state
            checkpoint = torch.load(MODEL_FILE, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.to(self.device).eval()
            self.model_type = SELECTED_MODEL
            print(f"✅ {self.model_type} model loaded successfully")
            
        except Exception as e:
            print(f"💥 FATAL: Could not load models/scalers for {SELECTED_MODEL}: {e}")
            traceback.print_exc()
            sys.exit(1)

    def _validate_features(self, features):
        """Comprehensive feature validation."""
        try:
            features_array = np.array(features, dtype=np.float32)
            if not np.all(np.isfinite(features_array)):
                return False, f"Found invalid values (NaN/Inf)"
            if np.any(np.abs(features_array) > 10000):
                return False, "Features outside expected range"
            return True, "Features validated"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _detect_market_conditions(self, features):
        """Detect current market volatility and trend."""
        try:
            features_array = np.array(features).reshape(SEQ_LEN, INPUT_FEATURES)
            price_returns = features_array[:, 0][-10:] # Price return is the first feature
            
            volatility = np.std(price_returns) * np.sqrt(252 * 24) # Annualized hourly volatility
            if volatility > 0.15: vol_regime = "high"
            elif volatility < 0.05: vol_regime = "low"
            else: vol_regime = "normal"
            
            self.market_conditions = {"volatility": vol_regime, "trend": "neutral"}
        except Exception:
            self.market_conditions = {"volatility": "unknown", "trend": "unknown"}

    def _apply_smoothing(self, prices):
        """Simple exponential moving average smoothing."""
        if len(prices) < 2:
            return prices
        alpha = 0.4 # Smoothing factor
        smoothed = [prices[0]]
        for i in range(1, len(prices)):
            smoothed.append(alpha * prices[i] + (1 - alpha) * smoothed[-1])
        return smoothed

    def _calculate_confidence(self, prices, uncertainties, model_confidence):
        """Multi-factor confidence calculation."""
        try:
            price_changes = np.diff(prices)
            consistency = 1.0 / (1.0 + np.std(price_changes) * 1000)
            avg_uncertainty = np.mean(uncertainties)
            uncertainty_conf = 1.0 / (1.0 + avg_uncertainty)
            
            # Weighted average of model confidence, consistency, and uncertainty
            final_confidence = (model_confidence * 0.5) + (consistency * 0.25) + (uncertainty_conf * 0.25)
            
            return np.clip(final_confidence, 0.05, 0.95)
        except Exception:
            return 0.3 # Fallback

    def _get_combined_prediction(self, features: list, current_price: float, atr: float) -> dict:
        """Main prediction function with error handling."""
        if not all([self.model, self.scaler_feature, self.scaler_regressor_target]):
            raise RuntimeError("Models or scalers not properly loaded")
        
        is_valid, validation_message = self._validate_features(features)
        if not is_valid:
            return self._get_fallback_prediction(current_price, validation_message)
        
        try:
            self._detect_market_conditions(features)
            
            arr = np.array(features, dtype=np.float32).reshape(1, SEQ_LEN, INPUT_FEATURES)
            scaled_features = self.scaler_feature.transform(arr.reshape(-1, INPUT_FEATURES))
            scaled_sequence = scaled_features.reshape(1, SEQ_LEN, INPUT_FEATURES)
            tensor = torch.tensor(scaled_sequence, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                predictions, class_logits, uncertainties, model_confidence, _ = self.model(tensor)
                
                predictions = predictions.cpu().numpy()[0]
                uncertainties = uncertainties.cpu().numpy()[0]
                model_confidence = float(model_confidence.cpu().numpy().flatten()[0])
                classification_probs = torch.softmax(class_logits, dim=1)[0].cpu().numpy()
            
            try:
                unscaled_predictions = self.scaler_regressor_target.inverse_transform(predictions.reshape(1, -1))[0]
                if not all(0.5 < price < 2.5 for price in unscaled_predictions):
                    raise ValueError("Unrealistic price from scaler")
            except Exception:
                unscaled_predictions = [current_price + (pred * atr) for pred in predictions]

            smoothed_prices = self._apply_smoothing(unscaled_predictions)
            
            confidence_score = self._calculate_confidence(smoothed_prices, uncertainties, model_confidence)
            
            sell_prob, hold_prob, buy_prob = classification_probs
            
            return {
                "predicted_prices": [float(p) for p in smoothed_prices],
                "confidence_score": float(confidence_score),
                "buy_probability": float(buy_prob),
                "sell_probability": float(sell_prob), 
                "hold_probability": float(hold_prob),
                "model_confidence": float(model_confidence),
                "market_conditions": self.market_conditions,
                "prediction_source": self.model_type,
                "validation_status": "passed"
            }
            
        except Exception as e:
            print(f"💥 Prediction failed: {e}")
            traceback.print_exc()
            return self._get_fallback_prediction(current_price, f"Prediction error: {str(e)}")

    def _get_fallback_prediction(self, current_price, error_message):
        """Conservative fallback prediction."""
        fallback_prices = [current_price + (np.random.randn() * 0.0001) for _ in range(OUTPUT_STEPS)]
        return {
            "predicted_prices": fallback_prices,
            "confidence_score": 0.1,
            "buy_probability": 0.33,
            "sell_probability": 0.33,
            "hold_probability": 0.34,
            "model_confidence": 0.1,
            "market_conditions": {"volatility": "unknown", "trend": "unknown"},
            "prediction_source": "fallback",
            "validation_status": "failed",
            "error_message": error_message
        }

    def _handle_request(self, filepath: str):
        """Process an incoming prediction request file."""
        request_id = "unknown"
        response = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            request_id = data.get("request_id", os.path.basename(filepath))
            action = data.get("action")
            
            if action == "predict_combined":
                features = data["features"]
                current_price = float(data["current_price"])
                atr = float(data["atr"])
                
                prediction_result = self._get_combined_prediction(features, current_price, atr)
                response = {"request_id": request_id, "status": "success", **prediction_result}
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {request_id} | Conf: {prediction_result.get('confidence_score', 0):.3f} | Source: {prediction_result.get('prediction_source', 'N/A')}")
            else:
                raise ValueError(f"Unsupported action: '{action}'")
                
        except Exception as e:
            error_msg = f"Error processing {os.path.basename(filepath)}: {e}"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ {request_id} | Error: {error_msg}")
            response = {"request_id": request_id, "status": "error", "error_message": error_msg}
        
        # Write response file
        try:
            resp_final = os.path.join(DATA_DIR, f"response_{request_id}.json")
            with open(resp_final, 'w', encoding='utf-8') as f:
                json.dump(response, f)
        except Exception as e:
            print(f"💥 Failed to write response for {request_id}: {e}")

    def run(self):
        """Main daemon loop to monitor for request files."""
        print(f"\n🎯 ALFA / Transformer Daemon v3.1 is running!")
        print(f"🔧 Model: {self.model_type}")
        print(f"📁 Monitoring: {DATA_DIR}")
        print(f"⏱️ Poll interval: {POLL_INTERVAL}s")
        print("=" * 60)
        
        while True:
            try:
                for filename in os.listdir(DATA_DIR):
                    if filename.startswith("request_") and filename.endswith(".json"):
                        filepath = os.path.join(DATA_DIR, filename)
                        self._handle_request(filepath)
                        try:
                            os.remove(filepath)
                        except OSError as e:
                            print(f"⚠️ Could not remove request file {filepath}: {e}")
                
                time.sleep(POLL_INTERVAL)
                
            except KeyboardInterrupt:
                print(f"\n🛑 Daemon shutting down gracefully...")
                break
            except Exception as e:
                print(f"💥 Unexpected error in main loop: {e}")
                traceback.print_exc()
                time.sleep(5)

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    try:
        daemon = AlfaTransformerDaemon()
        daemon.run()
    except Exception as e:
        print(f"💥 Fatal error starting daemon: {e}")
        traceback.print_exc()
        sys.exit(1)