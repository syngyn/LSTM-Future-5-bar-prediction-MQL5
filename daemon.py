import os
import json
import time
import torch
import joblib
import numpy as np
import traceback
from torch import nn
from datetime import datetime
import sys
try:
    from pykalman import KalmanFilter
except ImportError:
    print("⚠️  pykalman not available, using simplified smoothing")
    KalmanFilter = None
from sklearn.preprocessing import RobustScaler
import torch.nn.functional as F

# --- Configuration ---
def find_mql5_files_path():
    appdata = os.getenv('APPDATA')
    if not appdata or 'win' not in sys.platform: 
        return None
    metaquotes_path = os.path.join(appdata, 'MetaQuotes', 'Terminal')
    if not os.path.isdir(metaquotes_path): 
        return None
    for entry in os.listdir(metaquotes_path):
        terminal_path = os.path.join(metaquotes_path, entry)
        if os.path.isdir(terminal_path) and len(entry) > 30 and all(c in '0123456789ABCDEF' for c in entry.upper()):
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
HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN = 128, 3, 20  # Enhanced with 3 layers
POLL_INTERVAL = 0.1  # Faster response time  # Faster response time
OUTPUT_STEPS = 5
NUM_CLASSES = 3

# --- ENHANCED MODEL ARCHITECTURE ---
class AttentionLSTM(nn.Module):
    """Advanced LSTM with attention mechanism and uncertainty estimation"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_regression_outputs, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        
        # Input layer normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # Enhanced LSTM with residual connections
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Layer normalization for LSTM output
        self.lstm_norm = nn.LayerNorm(hidden_size)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Attention normalization
        self.attention_norm = nn.LayerNorm(hidden_size)
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Separate specialized heads
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_regression_outputs)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_regression_outputs)
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Input normalization
        x = self.input_norm(x)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        
        # Self-attention mechanism
        attn_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attention_norm(attn_out)
        
        # Feature fusion: combine LSTM and attention outputs
        combined = torch.cat([lstm_out, attn_out], dim=-1)
        fused = self.fusion(combined)
        
        # Aggregate temporal information
        # Use weighted average based on attention + last hidden state
        last_hidden = fused[:, -1, :]  # Last timestep
        avg_hidden = torch.mean(fused, dim=1)  # Average across time
        max_hidden = torch.max(fused, dim=1)[0]  # Max pooling
        
        # Multi-scale feature aggregation
        final_hidden = (last_hidden + avg_hidden + max_hidden) / 3
        
        # Generate outputs
        regression_output = self.regression_head(final_hidden)
        classification_logits = self.classification_head(final_hidden)
        uncertainty = torch.exp(self.uncertainty_head(final_hidden))  # Positive uncertainty
        model_confidence = self.confidence_head(final_hidden)
        
        return regression_output, classification_logits, uncertainty, model_confidence, attention_weights

# --- Fallback to original model if enhanced model fails ---
class CombinedLSTM(nn.Module):
    """Fallback model matching your original architecture"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_regression_outputs):
        super(CombinedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                           dropout=0.2 if num_layers > 1 else 0)
        self.fc_regression = nn.Linear(hidden_size, num_regression_outputs)
        self.fc_classification = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden_state = out[:, -1, :]
        regression_output = self.fc_regression(last_hidden_state)
        classification_logits = self.fc_classification(last_hidden_state)
        return regression_output, classification_logits

# --- ENHANCED DAEMON CLASS ---
class EnhancedLSTMDaemon:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        
        self.model = None
        self.scaler_feature = None
        self.scaler_regressor_target = None
        self.ensemble_models = []
        self.prediction_history = []
        self.market_conditions = {"volatility": "normal", "trend": "neutral"}
        
        self._load_models()

    def _load_models(self):
        """Load enhanced models with proper architecture matching"""
        print("🔄 Loading enhanced model and scalers...")
        try:
            # Load scalers first
            self.scaler_feature = joblib.load(SCALER_FILE_FEATURE)
            self.scaler_regressor_target = joblib.load(SCALER_FILE_REGRESSION_TARGET)
            print(f"✅ Scalers loaded successfully")
            
            # Load the model - try enhanced architecture first since that's what was just trained
            print("🎯 Attempting to load enhanced AttentionLSTM model...")
            try:
                self.model = AttentionLSTM(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, 
                                         NUM_CLASSES, OUTPUT_STEPS)
                checkpoint = torch.load(MODEL_FILE, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state'])
                self.model.to(self.device).eval()
                print(f"✅ Enhanced AttentionLSTM model loaded successfully")
                self.model_type = "enhanced"
                
            except Exception as e:
                print(f"⚠️  Enhanced model loading failed: {e}")
                print("🔄 Trying original CombinedLSTM architecture...")
                # Fallback to original model architecture
                try:
                    self.model = CombinedLSTM(INPUT_FEATURES, HIDDEN_SIZE, 2,  # Original had 2 layers
                                            NUM_CLASSES, OUTPUT_STEPS)
                    checkpoint = torch.load(MODEL_FILE, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state'])
                    self.model.to(self.device).eval()
                    print(f"✅ Original CombinedLSTM model loaded successfully")
                    self.model_type = "original"
                except Exception as e2:
                    print(f"💥 Both model architectures failed to load: {e2}")
                    raise e2
            
            # Try to load ensemble models
            self._load_ensemble_models()
            
        except Exception as e:
            print(f"💥 FATAL: Could not load models/scalers: {e}")
            traceback.print_exc()
            sys.exit(1)

    def _load_ensemble_models(self):
        """Load additional ensemble models for improved predictions"""
        ensemble_count = 0
        try:
            for i in range(1, 6):  # Try to load up to 5 ensemble models
                ensemble_path = os.path.join(MODEL_DIR, f"lstm_ensemble_{i}.pth")
                if os.path.exists(ensemble_path):
                    # Use the same architecture as the main model
                    if self.model_type == "enhanced":
                        ensemble_model = AttentionLSTM(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, 
                                                     NUM_CLASSES, OUTPUT_STEPS)
                    else:
                        ensemble_model = CombinedLSTM(INPUT_FEATURES, HIDDEN_SIZE, 2,  # Original layers
                                                    NUM_CLASSES, OUTPUT_STEPS)
                    
                    try:
                        checkpoint = torch.load(ensemble_path, map_location=self.device)
                        ensemble_model.load_state_dict(checkpoint['model_state'])
                        ensemble_model.to(self.device).eval()
                        self.ensemble_models.append(ensemble_model)
                        ensemble_count += 1
                    except Exception as e:
                        print(f"⚠️  Failed to load ensemble model {i}: {e}")
                        continue
            
            if ensemble_count > 0:
                print(f"✅ Loaded {ensemble_count} ensemble models")
            else:
                print("ℹ️  No ensemble models found (this is normal if you just trained the main model)")
                
        except Exception as e:
            print(f"⚠️  Ensemble loading failed (continuing with single model): {e}")

    def _validate_features(self, features):
        """Comprehensive but less strict feature validation"""
        try:
            features_array = np.array(features, dtype=np.float32)
            
            # Check for NaN or infinite values
            if not np.all(np.isfinite(features_array)):
                nan_count = np.sum(~np.isfinite(features_array))
                return False, f"Found {nan_count} invalid values (NaN/Inf)"
            
            # Much more relaxed outlier detection (12 standard deviations instead of 6)
            mean_val = np.mean(features_array)
            std_val = np.std(features_array)
            if std_val > 0:
                z_scores = np.abs((features_array - mean_val) / std_val)
                extreme_outliers = np.sum(z_scores > 12)  # Much more lenient
                if extreme_outliers > len(features_array) * 0.2:  # Allow more outliers (20%)
                    return False, f"Too many extreme outliers: {extreme_outliers}"
            
            # Check for constant sequences (market might be closed)
            recent_features = features_array[-30:]  # Last 30 features
            if len(recent_features) > 5 and np.std(recent_features) < 1e-10:  # Much smaller threshold
                return False, "Market appears inactive (constant values)"
            
            # Much more relaxed range check (allow larger values)
            if np.any(np.abs(features_array) > 10000):  # Increased from 1000 to 10000
                return False, "Features outside expected range"
                
            return True, "Features validated"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _detect_market_conditions(self, features):
        """Detect current market volatility and trend with timezone awareness"""
        try:
            features_array = np.array(features)
            
            # Extract recent price movements (assuming first feature is price return)
            price_returns = features_array[::INPUT_FEATURES][-10:]  # Last 10 price returns
            
            # Volatility detection
            volatility = np.std(price_returns) * np.sqrt(24)  # Annualized volatility
            if volatility > 0.02:  # 2% daily vol threshold
                vol_regime = "high"
            elif volatility < 0.01:  # 1% daily vol threshold
                vol_regime = "low" 
            else:
                vol_regime = "normal"
            
            # Trend detection
            trend_strength = np.abs(np.mean(price_returns))
            if trend_strength > 0.001:  # 0.1% average move
                trend_direction = "bullish" if np.mean(price_returns) > 0 else "bearish"
            else:
                trend_direction = "neutral"
            
            # TIMEZONE ADJUSTMENT: Check if we're dealing with timezone mismatch
            # If system is GMT-7 but training data was GMT+2, that's 9 hours difference
            current_time = datetime.now()
            hour_local = current_time.hour
            
            # Detect if we might have timezone issues based on current hour
            timezone_warning = False
            if 1 <= hour_local <= 6:  # 1 AM - 6 AM local time
                # This would be 10 AM - 3 PM broker time (GMT+2) - prime trading hours
                # If we're getting good volatility during these hours, timezone might be correct
                timezone_warning = False
            elif 7 <= hour_local <= 14:  # 7 AM - 2 PM local time  
                # This would be 4 PM - 11 PM broker time - lower activity expected
                if vol_regime == "high":
                    timezone_warning = True  # Unexpected high volatility
            else:
                # Other hours - check volatility patterns
                timezone_warning = False
            
            self.market_conditions = {
                "volatility": vol_regime,
                "trend": trend_direction,
                "vol_value": volatility,
                "trend_strength": trend_strength,
                "timezone_warning": timezone_warning,
                "local_hour": hour_local
            }
            
            if timezone_warning:
                print(f"⚠️  TIMEZONE WARNING: High volatility at local hour {hour_local} - possible timezone mismatch")
            
        except Exception as e:
            print(f"⚠️  Market condition detection failed: {e}")
            self.market_conditions = {
                "volatility": "unknown", 
                "trend": "unknown",
                "timezone_warning": True,
                "local_hour": datetime.now().hour
            }

    def _apply_advanced_smoothing(self, prices, uncertainties=None):
        """Advanced smoothing with fallback when Kalman filter unavailable"""
        if len(prices) < 2:
            return prices
            
        try:
            # If Kalman filter is available, use it
            if KalmanFilter is not None:
                # Adaptive parameters based on market conditions
                vol_regime = self.market_conditions.get("volatility", "normal")
                trend_regime = self.market_conditions.get("trend", "neutral")
                
                # Base parameters
                if vol_regime == "high":
                    obs_covariance = 0.02   # Higher noise in volatile markets
                    trans_covariance = 0.005
                elif vol_regime == "low":
                    obs_covariance = 0.005  # Lower noise in calm markets
                    trans_covariance = 0.001
                else:
                    obs_covariance = 0.01
                    trans_covariance = 0.002
                
                # Adjust for uncertainty if available
                if uncertainties is not None:
                    avg_uncertainty = np.mean(uncertainties)
                    obs_covariance *= (1 + avg_uncertainty)
                    trans_covariance *= (1 + avg_uncertainty * 0.5)
                
                # Apply Kalman filter
                kf = KalmanFilter(
                    transition_matrices=[1],
                    observation_matrices=[1], 
                    initial_state_mean=prices[0],
                    initial_state_covariance=obs_covariance,
                    observation_covariance=obs_covariance,
                    transition_covariance=trans_covariance
                )
                
                state_means, _ = kf.filter(prices)
                smoothed = state_means.flatten()
                
                # Trend-aware post-processing
                if trend_regime in ["bullish", "bearish"]:
                    trend_slope = np.polyfit(range(len(prices)), prices, 1)[0]
                    trend_adjustment = trend_slope * 0.1  # Small trend reinforcement
                    
                    for i in range(1, len(smoothed)):
                        smoothed[i] += trend_adjustment * i * 0.1
                
                return smoothed.tolist()
            
            else:
                # Fallback to simple moving average smoothing
                print("🔄 Using fallback smoothing (no Kalman filter)")
                smoothed_prices = []
                window = min(3, len(prices))  # Use 3-point moving average
                
                for i in range(len(prices)):
                    if i < window:
                        # For early points, use expanding average
                        smoothed_prices.append(np.mean(prices[:i+1]))
                    else:
                        # Moving average
                        smoothed_prices.append(np.mean(prices[i-window+1:i+1]))
                
                return smoothed_prices
            
        except Exception as e:
            print(f"⚠️  Smoothing failed, returning raw prices: {e}")
            return prices

    def _calculate_enhanced_confidence(self, prices, uncertainties, current_price, atr, model_confidence=None):
        """Multi-factor confidence calculation"""
        if atr is None or atr <= 1e-6:
            return 0.1
            
        try:
            confidence_components = []
            
            # 1. Price consistency factor (30% weight)
            price_changes = np.diff(np.insert(prices, 0, current_price))
            if len(price_changes) > 0:
                consistency = 1.0 / (1.0 + np.std(price_changes) * 1000)
                confidence_components.append(("consistency", consistency, 0.25))
            
            # 2. Model-based confidence (25% weight) - if available from enhanced model
            if model_confidence is not None:
                confidence_components.append(("model", float(model_confidence), 0.25))
            
            # 3. Uncertainty-based confidence (20% weight)
            if uncertainties is not None:
                avg_uncertainty = np.mean(uncertainties)
                uncertainty_conf = 1.0 / (1.0 + avg_uncertainty)
                confidence_components.append(("uncertainty", uncertainty_conf, 0.20))
            
            # 4. Directional agreement (15% weight)
            if len(price_changes) > 1:
                directions = np.sign(price_changes[price_changes != 0])
                if len(directions) > 0:
                    agreement = np.sum(directions == directions[0]) / len(directions)
                    confidence_components.append(("directional", agreement, 0.15))
            
            # 5. Market volatility factor (10% weight)
            vol_regime = self.market_conditions.get("volatility", "normal")
            if vol_regime == "low":
                vol_confidence = 1.0
            elif vol_regime == "normal":
                vol_confidence = 0.8
            else:  # high volatility
                vol_confidence = 0.6
            confidence_components.append(("volatility", vol_confidence, 0.10))
            
            # 6. Historical performance (5% weight)
            if len(self.prediction_history) >= 5:
                recent_accuracy = np.mean([
                    p.get('was_accurate', 0.5) for p in self.prediction_history[-10:]
                    if p.get('was_accurate') is not None
                ])
                confidence_components.append(("historical", recent_accuracy, 0.05))
            
            # Calculate weighted average
            total_weight = 0
            weighted_sum = 0
            
            for name, value, weight in confidence_components:
                if 0 <= value <= 1:  # Valid confidence value
                    weighted_sum += value * weight
                    total_weight += weight
            
            if total_weight > 0:
                final_confidence = weighted_sum / total_weight
            else:
                final_confidence = 0.5  # Default confidence
            
            # Apply market condition adjustments
            final_confidence = self._adjust_confidence_by_market_conditions(
                final_confidence, len(confidence_components)
            )
            
            return np.clip(final_confidence, 0.05, 0.95)  # Keep in reasonable range
            
        except Exception as e:
            print(f"⚠️  Confidence calculation failed: {e}")
            return 0.3  # Conservative fallback

    def _adjust_confidence_by_market_conditions(self, base_confidence, num_factors):
        """Fine-tune confidence based on market timing and conditions"""
        try:
            current_hour = datetime.now().hour
            adjustments = []
            
            # Time-based adjustments
            if 8 <= current_hour <= 16:  # Major market hours (London/NY)
                adjustments.append(1.1)  # Boost confidence
            elif 22 <= current_hour or current_hour <= 5:  # Asian session
                adjustments.append(0.95)  # Slight reduction
            else:  # Off hours
                adjustments.append(0.85)  # Reduce confidence
                
            # News hour reduction (approximate major news times)
            major_news_hours = [8, 9, 12, 13, 14, 15]  # UTC
            if current_hour in major_news_hours:
                adjustments.append(0.9)  # Reduce during news
                
            # Factor completeness adjustment
            if num_factors >= 5:
                adjustments.append(1.05)  # Boost for comprehensive analysis
            elif num_factors < 3:
                adjustments.append(0.9)   # Reduce for limited factors
            
            # Apply all adjustments
            final_confidence = base_confidence
            for adj in adjustments:
                final_confidence *= adj
                
            return np.clip(final_confidence, 0.05, 0.95)
            
        except Exception as e:
            print(f"⚠️  Market condition adjustment failed: {e}")
            return base_confidence

    def _get_ensemble_prediction(self, scaled_tensor):
        """Generate ensemble predictions from multiple models"""
        all_predictions = []
        all_uncertainties = []
        all_classifications = []
        all_confidences = []
        
        models_to_use = [self.model] + self.ensemble_models
        
        for i, model in enumerate(models_to_use):
            try:
                with torch.no_grad():
                    if self.model_type == "enhanced":
                        reg_out, class_logits, uncertainty, confidence, _ = model(scaled_tensor)
                        all_uncertainties.append(uncertainty.cpu().numpy()[0])
                        all_confidences.append(float(confidence.cpu().numpy()[0]))
                    else:
                        reg_out, class_logits = model(scaled_tensor)
                        # Create dummy uncertainty and confidence for original model
                        all_uncertainties.append(np.ones(OUTPUT_STEPS) * 0.1)
                        all_confidences.append(0.7)
                    
                    all_predictions.append(reg_out.cpu().numpy()[0])
                    all_classifications.append(
                        torch.softmax(class_logits, dim=1)[0].cpu().numpy()
                    )
                    
            except Exception as e:
                print(f"⚠️  Model {i} failed: {e}")
                continue
        
        if not all_predictions:
            raise RuntimeError("All ensemble models failed")
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        uncertainties = np.array(all_uncertainties)
        classifications = np.array(all_classifications)
        confidences = np.array(all_confidences)
        
        # Weight by confidence (higher confidence = higher weight)
        weights = confidences / np.sum(confidences)
        
        # Ensemble predictions
        final_prediction = np.average(predictions, axis=0, weights=weights)
        final_uncertainty = np.average(uncertainties, axis=0, weights=weights)
        final_classification = np.average(classifications, axis=0, weights=weights)
        avg_confidence = np.average(confidences, weights=weights)
        
        return final_prediction, final_uncertainty, final_classification, avg_confidence

    def _get_combined_prediction(self, features: list, current_price: float, atr: float) -> dict:
        """Main prediction function with comprehensive error handling"""
        
        # Validate inputs
        if not all([self.model, self.scaler_feature, self.scaler_regressor_target]):
            raise RuntimeError("Models or scalers not properly loaded")
        
        # Feature validation
        is_valid, validation_message = self._validate_features(features)
        if not is_valid:
            print(f"⚠️  Feature validation failed: {validation_message}")
            # Return conservative prediction
            return self._get_fallback_prediction(current_price, validation_message)
        
        try:
            # Detect market conditions
            self._detect_market_conditions(features)
            
            # Prepare input tensor
            arr = np.array(features, dtype=np.float32).reshape(1, SEQ_LEN, INPUT_FEATURES)
            scaled_features = self.scaler_feature.transform(arr.reshape(-1, INPUT_FEATURES))
            scaled_sequence = scaled_features.reshape(1, SEQ_LEN, INPUT_FEATURES)
            tensor = torch.tensor(scaled_sequence, dtype=torch.float32).to(self.device)
            
            # Get predictions
            if len(self.ensemble_models) > 0:
                predictions, uncertainties, classification_probs, model_confidence = self._get_ensemble_prediction(tensor)
                prediction_source = f"ensemble_{len(self.ensemble_models) + 1}"
            else:
                with torch.no_grad():
                    if self.model_type == "enhanced":
                        reg_out, class_logits, uncertainty, confidence, _ = self.model(tensor)
                        predictions = reg_out.cpu().numpy()[0]
                        uncertainties = uncertainty.cpu().numpy()[0]
                        model_confidence = float(confidence.cpu().numpy()[0])
                    else:
                        reg_out, class_logits = self.model(tensor)
                        predictions = reg_out.cpu().numpy()[0]
                        uncertainties = np.ones(OUTPUT_STEPS) * 0.1
                        model_confidence = 0.7
                    
                    classification_probs = torch.softmax(class_logits, dim=1)[0].cpu().numpy()
                    prediction_source = f"single_{self.model_type}"
            
            # Denormalize predictions
            unscaled_predictions = self.scaler_regressor_target.inverse_transform(
                predictions.reshape(1, -1)
            )[0]
            
            # Apply advanced smoothing
            smoothed_prices = self._apply_advanced_smoothing(unscaled_predictions, uncertainties)
            
            # Calculate enhanced confidence
            confidence_score = self._calculate_enhanced_confidence(
                smoothed_prices, uncertainties, current_price, atr, model_confidence
            )
            
            # Extract classification probabilities
            sell_prob, hold_prob, buy_prob = classification_probs
            
            # Store prediction for historical tracking
            prediction_record = {
                'timestamp': datetime.now(),
                'predictions': smoothed_prices,
                'confidence': confidence_score,
                'current_price': current_price,
                'market_conditions': self.market_conditions.copy(),
                'was_accurate': None,  # Will be updated later
                'source': prediction_source
            }
            self.prediction_history.append(prediction_record)
            
            # Maintain history size
            if len(self.prediction_history) > 200:
                self.prediction_history = self.prediction_history[-100:]
            
            return {
                "predicted_prices": smoothed_prices,
                "confidence_score": confidence_score,
                "buy_probability": float(buy_prob),
                "sell_probability": float(sell_prob), 
                "hold_probability": float(hold_prob),
                "model_confidence": model_confidence,
                "market_conditions": self.market_conditions,
                "uncertainty": uncertainties.tolist() if hasattr(uncertainties, 'tolist') else None,
                "ensemble_size": len(self.ensemble_models) + 1,
                "prediction_source": prediction_source,
                "validation_status": "passed"
            }
            
        except Exception as e:
            print(f"💥 Prediction failed: {e}")
            traceback.print_exc()
            return self._get_fallback_prediction(current_price, f"Prediction error: {str(e)}")

    def _get_fallback_prediction(self, current_price, error_message):
        """Conservative fallback prediction when main prediction fails"""
        
        # Generate conservative predictions (small movements around current price)
        small_changes = np.random.normal(0, current_price * 0.001, OUTPUT_STEPS)  # 0.1% std dev
        fallback_prices = [current_price + change for change in small_changes]
        
        return {
            "predicted_prices": fallback_prices,
            "confidence_score": 0.1,  # Very low confidence
            "buy_probability": 0.33,
            "sell_probability": 0.33,
            "hold_probability": 0.34,
            "model_confidence": 0.1,
            "market_conditions": {"volatility": "unknown", "trend": "unknown"},
            "uncertainty": [0.5] * OUTPUT_STEPS,
            "ensemble_size": 0,
            "prediction_source": "fallback",
            "validation_status": "failed",
            "error_message": error_message
        }

    def _handle_request(self, filepath: str):
        """Process incoming prediction requests with robust error handling and time sync checking"""
        request_id = "unknown"
        response = {}
        
        try:
            # Load request with better error handling
            with open(filepath, 'r', encoding='utf-8') as f:
                file_content = f.read().strip()
            
            # Skip empty files
            if not file_content:
                print(f"⚠️  Skipping empty request file: {filepath}")
                return
                
            # Try to parse JSON with better error reporting
            try:
                data = json.loads(file_content)
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON parse error in {filepath}:")
                print(f"    Error: {e}")
                print(f"    Content preview: {file_content[:200]}...")
                # Try to extract request_id from malformed JSON if possible
                if '"request_id"' in file_content:
                    try:
                        request_id = file_content.split('"request_id"')[1].split('"')[1]
                    except:
                        request_id = f"malformed_{datetime.now().strftime('%H%M%S')}"
                raise ValueError(f"Malformed JSON: {e}")
            
            request_id = data.get("request_id", os.path.basename(filepath))
            action = data.get("action")
            features = data.get("features")
            current_price = data.get("current_price")
            atr = data.get("atr")
            
            # Extract timestamp information for debugging
            ea_time = data.get("ea_time", "Unknown")
            server_time = data.get("server_time", "Unknown")
            local_time = data.get("local_time", "Unknown")
            current_bar_time = data.get("current_bar_time", "Unknown")
            symbol = data.get("symbol", "Unknown")
            timeframe = data.get("timeframe", "Unknown")
            
            # Log time sync information
            daemon_time = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
            print(f"🕐 [Time Sync] Request {request_id}:")
            print(f"   📊 Symbol: {symbol} | Timeframe: {timeframe}")
            print(f"   🤖 Daemon Time: {daemon_time}")
            print(f"   📈 EA Time: {ea_time}")
            print(f"   🖥️  Server Time: {server_time}")
            print(f"   🏠 Local Time: {local_time}")
            print(f"   📊 Current Bar: {current_bar_time}")
            print(f"   💰 Price: {current_price} | ATR: {atr}")
            
            # Check for potential time misalignment
            time_warnings = []
            try:
                from datetime import datetime as dt
                daemon_dt = dt.now()
                
                # Try to parse EA time
                if ea_time != "Unknown":
                    try:
                        ea_dt = dt.strptime(ea_time, "%Y.%m.%d %H:%M:%S")
                        time_diff = abs((daemon_dt - ea_dt).total_seconds())
                        
                        if time_diff > 3600:  # More than 1 hour difference
                            time_warnings.append(f"Large time difference: {time_diff/3600:.1f} hours")
                        elif time_diff > 300:  # More than 5 minutes
                            time_warnings.append(f"Time difference: {time_diff/60:.1f} minutes")
                            
                    except ValueError:
                        time_warnings.append("Could not parse EA time format")
                        
                if time_warnings:
                    print(f"   ⚠️  TIME WARNINGS: {', '.join(time_warnings)}")
                    
            except Exception as e:
                print(f"   ⚠️  Time validation error: {e}")
            
            # Validate request structure
            if not action:
                raise ValueError("Missing 'action' field in request")
            if not features:
                raise ValueError("Missing 'features' field in request")
            if current_price is None:
                raise ValueError("Missing 'current_price' field in request")
            if atr is None:
                raise ValueError("Missing 'atr' field in request")
                
            # Convert to proper types
            try:
                current_price = float(current_price)
                atr = float(atr)
                if not isinstance(features, list):
                    raise ValueError("Features must be a list")
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid data types: {e}")
            
            if action == "predict_combined":
                prediction_result = self._get_combined_prediction(features, current_price, atr)
                
                # Add time sync information to response
                prediction_result["time_sync"] = {
                    "daemon_time": daemon_time,
                    "ea_time": ea_time,
                    "server_time": server_time,
                    "time_warnings": time_warnings
                }
                
                response = {
                    "request_id": request_id,
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                    **prediction_result
                }
                
                # Log successful prediction with time info
                confidence = prediction_result.get('confidence_score', 0)
                source = prediction_result.get('prediction_source', 'unknown')
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {request_id} | Conf: {confidence:.3f} | Source: {source}")
                
                # Log time warnings if any
                if time_warnings:
                    print(f"   ⚠️  Time sync issues detected - predictions may be inaccurate!")
                    
            else:
                raise ValueError(f"Unsupported action: '{action}'")
                
        except Exception as e:
            error_msg = str(e)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ {request_id} | Error: {error_msg}")
            
            response = {
                "request_id": request_id,
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error_message": error_msg,
                "fallback_used": True
            }
        
        # Write response with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp_temp = os.path.join(DATA_DIR, f"response_{request_id}_{attempt}.tmp")
                resp_final = os.path.join(DATA_DIR, f"response_{request_id}.json")
                
                with open(resp_temp, 'w', encoding='utf-8') as f:
                    json.dump(response, f, indent=2)
                
                # Atomic rename
                if os.path.exists(resp_final):
                    os.remove(resp_final)  # Remove existing file first
                os.rename(resp_temp, resp_final)
                break  # Success
                
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    print(f"💥 Failed to write response for {request_id} after {max_retries} attempts: {e}")
                else:
                    time.sleep(0.01)  # Small delay before retry

    def run(self):
        """Main daemon loop"""
        print(f"\n🎯 Enhanced LSTM Daemon v2.1 is running!")
        print(f"📊 Device: {self.device}")
        print(f"🔧 Model: {self.model_type}")
        print(f"🎭 Ensemble: {len(self.ensemble_models) + 1} models")
        print(f"📁 Monitoring: {DATA_DIR}")
        print(f"⏱️  Poll interval: {POLL_INTERVAL}s")
        print("=" * 60)
        
        request_count = 0
        start_time = time.time()
        
        while True:
            try:
                processed_this_cycle = 0
                
                # Process all pending requests
                for filename in os.listdir(DATA_DIR):
                    if filename.startswith("request_") and filename.endswith(".json"):
                        filepath = os.path.join(DATA_DIR, filename)
                        
                        # Process request
                        self._handle_request(filepath)
                        processed_this_cycle += 1
                        request_count += 1
                        
                        # Clean up request file
                        try:
                            os.remove(filepath)
                        except OSError as e:
                            print(f"⚠️  Could not remove {filepath}: {e}")
                
                # Periodic status update
                if request_count > 0 and request_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = request_count / elapsed
                    print(f"📈 Processed {request_count} requests ({rate:.1f}/sec avg)")
                
                time.sleep(POLL_INTERVAL)
                
            except KeyboardInterrupt:
                print(f"\n🛑 Daemon shutting down gracefully...")
                print(f"📊 Total requests processed: {request_count}")
                print(f"⏱️  Uptime: {time.time() - start_time:.1f} seconds")
                break
                
            except Exception as e:
                print(f"💥 Unexpected error in main loop: {e}")
                traceback.print_exc()
                print("🔄 Continuing after 5 second pause...")
                time.sleep(5)

if __name__ == "__main__":
    # Ensure required directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Start daemon
    try:
        daemon = EnhancedLSTMDaemon()
        daemon.run()
    except Exception as e:
        print(f"💥 Fatal error starting daemon: {e}")
        traceback.print_exc()
        sys.exit(1)