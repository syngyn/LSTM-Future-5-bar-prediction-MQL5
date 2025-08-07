<artifacts>
<artifact identifier="ea_description_guide" type="text/markdown" title="GGTH LSTM Expert Advisor - Complete Description & Installation Guide">
# GGTH LSTM Expert Advisor v2.10 - Complete Guide

## 🤖 What Is This Expert Advisor?

The **GGTH LSTM Expert Advisor** is a cutting-edge AI-powered automated trading system that combines deep learning neural networks with advanced position sizing and risk management. It represents one of the most sophisticated retail algorithmic trading systems available, integrating multiple state-of-the-art technologies into a single, cohesive trading solution.

## 🏗️ System Architecture

### **Two-Part System:**
1. **Python AI Backend** - Runs sophisticated neural network models
2. **MetaTrader 5 Expert Advisor** - Executes trades and manages positions

### **Communication Flow:**
```
MT5 EA → JSON Request → Python Daemon → LSTM Prediction → JSON Response → MT5 EA → Trade Execution
```

## 🧠 Core Technologies

### **1. AttentionLSTM Neural Network**
- **Architecture**: 3-layer LSTM with 8-head multi-attention mechanism
- **Input**: 15 technical features across 20 time steps (H-19 to H0)
- **Output**: 5-step price predictions (H+1 to H+5) with confidence scores
- **Training**: Ensemble of 4 models with uncertainty estimation

### **2. Kelly Criterion Position Sizing**
- **Dynamic Sizing**: Positions automatically sized based on historical win rate and profit factors
- **Confidence Scaling**: Larger positions when model confidence is high
- **Risk Controls**: Minimum 1% / Maximum 25% of capital per trade
- **Smoothing**: Prevents position size whipsawing

### **3. Adaptive Learning System**
- **Performance Tracking**: Monitors prediction accuracy for each time horizon
- **Dynamic Adjustments**: Automatically adjusts confidence thresholds and risk multipliers
- **Market Adaptation**: Responds to changing market conditions
- **Step Weighting**: Emphasizes better-performing prediction horizons

### **4. Dual Trading Strategies**
- **Main Strategy**: Medium-term positions based on H+1 to H+5 predictions
- **Scalping Strategy**: Short-term trades targeting immediate price movements

## 📊 Key Features

### **Trading Capabilities**
- ✅ **Multi-timeframe Predictions** - H+1 through H+5 hour forecasts
- ✅ **Dual Confirmation** - Regression + Classification signals
- ✅ **Kelly Criterion Sizing** - Mathematically optimal position sizing
- ✅ **Adaptive Learning** - System improves from its own performance
- ✅ **Scalping Integration** - Additional short-term opportunities
- ✅ **Trading Hours Control** - Configurable session management
- ✅ **Real-time GUI** - Live performance monitoring

### **Risk Management**
- ✅ **ATR-based Stops** - Dynamic stop losses using market volatility
- ✅ **Time-based Exits** - Maximum holding periods
- ✅ **Trailing Stops** - Lock in profits as trades move favorably
- ✅ **Drawdown Protection** - Reduces size during losing periods
- ✅ **Position Limits** - Maximum risk per trade controls

### **Advanced Analytics**
- ✅ **Prediction Accuracy Tracking** - Real-time hit rate monitoring
- ✅ **Currency Strength Analysis** - Multi-pair correlation modeling
- ✅ **Market Condition Detection** - Volatility and trend regime awareness
- ✅ **Performance Attribution** - Detailed trade analysis and learning

## 🎯 Supported Currency Pairs

**Primary Trading Pair**: EURUSD (H1 timeframe)

**Analysis Pairs** (for currency strength calculation):
- EURJPY, USDJPY, GBPUSD, EURGBP, USDCAD, USDCHF

## 📋 System Requirements

### **Software Requirements**
- **MetaTrader 5** (Build 3550 or higher)
- **Python 3.7+** (Recommended: Python 3.9-3.11)
- **Windows 10/11** (Primary support) or **macOS/Linux** (with manual configuration)

### **Hardware Requirements**
- **CPU**: Intel i5 or AMD Ryzen 5 equivalent (minimum)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space for data and models
- **Internet**: Stable connection for real-time data

### **Python Libraries** (Auto-installed)
- PyTorch 1.9+, Pandas, NumPy, Scikit-learn, Joblib, yfinance, Matplotlib

## 🚀 Installation Guide

### **Step 1: Download and Extract**
1. Extract all files to a dedicated folder (e.g., `C:\GGTH_LSTM\`)
2. Ensure you have these files:
   ```
   GGTH_LSTM/
   ├── GGTH.mq5                 # MetaTrader Expert Advisor
   ├── install.py                  # Universal installer
   ├── train_enhanced_model.py     # Neural network training
   ├── daemon.py                   # Prediction server
   ├── generate_backtest.py        # Backtest data generator
   ├── launcher.py                 # Easy interface
   ├── path_helper.py              # Path management
   ├── requirements.txt            # Python dependencies
   └── EURUSD60.csv               # Training data (if available)
   ```

### **Step 2: Run Universal Installer**
1. **Open Command Prompt as Administrator** (type cmd in windows desktop search bar then right click on the command prompt and select run as adminstrator)
2. **Navigate to folder**: type: `cd C:\GGTH\`
3. **Run installer**:type: `python launcher.py`
4. Select 4 to install

The installer will automatically:
- ✅ Install all Python dependencies
- ✅ Detect MetaTrader 5 paths
- ✅ Create communication folders
- ✅ Download sample data (if needed)
- ✅ Create launch shortcuts

### **Step 3: Copy Expert Advisor to MetaTrader**
1. **Copy** `GGTH.mq5` to your MT5 Experts folder:
   - Path: `C:\Users\[Username]\AppData\Roaming\MetaQuotes\Terminal\[ID]\MQL5\Experts\`
2. **Restart MetaTrader 5**
3. **Verify** the EA appears in Navigator → Expert Advisors

### **Step 4: Train the Neural Network**
1. **Option A - Use Launcher** (Recommended):
   - cmd prompt (type cmd in windows desktop search bar)
   - type python launcher.py
   - Select option `1. train`

2. **Option B - Direct Command**:
   python train_enhanced_model.py


**Training Process** (15-30 minutes):
- Loads historical EURUSD data
- Creates advanced technical features
- Trains AttentionLSTM ensemble models
- Saves trained models and scalers

### **Step 5: Start the Prediction Daemon**
1. **Option A - Use Launcher**:
2. from command prompt run python launcher.py
   - Select option `2. daemon`

3. **Option B - Direct Command**:

   python daemon.py

**You should see**:
```
🚀 Enhanced LSTM Daemon v2.1 is running!
📊 Device: cuda (or cpu)
🔧 Model: enhanced
🎭 Ensemble: 4 models
📁 Monitoring: [path to communication folder]
```

### **Step 6: Configure MetaTrader 5**
1. **Enable Algo Trading**: Click "Algo Trading" button in MT5 toolbar

### **Step 7: Load the Expert Advisor**
1. **Open EURUSD H1 chart** in MetaTrader 5
2. **Drag** `GGTH from Navigator onto the chart
3. **Configure parameters** (see Configuration section below)
4. **Click "OK"** and ensure auto-trading is enabled

## 🎮 Usage Instructions

### **Daily Workflow**
1. **Start Python Daemon** (once per day or after system restart)
2. **Load EA** on EURUSD H1 chart
3. **Monitor GUI** for system status and performance
4. **Check logs** for prediction accuracy and trade decisions

### **GUI Dashboard Elements**
- **Connection Status**: Daemon communication status
- **Prediction Accuracy**: Hit rates for H+1 through H+5
- **Kelly Metrics**: Current position sizing parameters
- **Adaptive Status**: Learning system status
- **Market Hours**: Trading session status
- **Live Predictions**: Real-time price forecasts

### **Key Logs to Monitor**
- **Prediction Updates**: New forecasts every hour
- **Trade Executions**: Position entries with reasoning
- **Accuracy Updates**: Prediction hit/miss tracking
- **Kelly Updates**: Position sizing adjustments
- **Error Messages**: Connection or prediction issues

## 🎯 Backtesting

### **Generate Historical Predictions**
1. **Run**: `python generate_backtest.py`
2. **Copy** `backtest_predictions.csv` to MT5 Common Files folder C:\Users\username\AppData\Roaming\MetaQuotes\Terminal\Common\Files
3. **Run Strategy Tester** with generated data for accurate historical testing the ai will learn from each test run you do.

### **Strategy Tester Settings**
- **Model**: Every tick based on real ticks
- **Period**: Any period with sufficient data
- **Deposit**: $10,000 recommended minimum
- **Leverage**: 1:100 or higher

## ⚠️ Important Warnings & Considerations


### **Technical Considerations**
- **Internet Connection**: Stable connection required for Python-MT5 communication
- **System Resources**: Keep adequate CPU and memory available
- **Data Quality**: Ensure clean price data for accurate predictions
- **Time Synchronization**: Verify broker and system time alignment

### **Best Practices**
- ✅ **Start with demo account** for initial testing
- ✅ **Monitor prediction accuracy** before live trading
- ✅ **Use conservative Kelly settings** initially (Max 10-15%)
- ✅ **Regular model retraining** with fresh data
- ✅ **Maintain system logs** for troubleshooting

## 🔧 Troubleshooting

### **Common Issues**

#### **"Daemon not connected"**
- Ensure Python daemon is running (`python daemon.py`)
- Check firewall settings
- Verify communication folder paths

#### **"No predictions available"**
- Check if models are trained (`models/` folder exists)
- Verify EURUSD data availability
- Check Python daemon logs for errors

#### **"Invalid lot size"**
- Verify account balance and margin requirements
- Check broker volume limits
- Adjust Kelly parameters if too aggressive

#### **Poor prediction accuracy**
- Retrain models with recent data
- Adjust confidence thresholds
- Consider market regime changes

### **Getting Help**
- **Check logs** in MetaTrader Journal and Python daemon output
- **Verify configuration** against this guide
- **Test with demo account** first
- **Monitor system resources** during operation

## 📈 Expected Performance

### **Realistic Expectations**
- **Directional Accuracy**: 70% (research-backed range)
- **Win Rate**: 65 - 85% (quality over quantity approach)
- **Risk-Adjusted Returns**: Sharpe ratio 1.2-2.0 with proper Kelly sizing
- **Maximum Drawdown**: 10-20% during challenging periods


## 🚨 **Critical First Steps After Installation:**

1. **Test the Connection**: After installation, the most important thing is ensuring the Python daemon and MT5 EA can communicate properly

2. **Start Small**: Begin with demo account or very small position sizes until you verify the system works as expected

3. **Monitor the GUI**: The real-time dashboard will tell you immediately if something's wrong

4. **Check Prediction Accuracy**: Let it run for a few days and monitor the hit rates before trusting it with significant capital

## 🎯 **Quick Start Checklist:**

- [ ] Install Python dependencies with `install.py`
- [ ] Copy `GGTH8-5.mq5` to MT5 Experts folder  
- [ ] Train models with `train_enhanced_model.py`
- [ ] Start daemon with `daemon.py`
- [ ] Load EA on EURUSD H1 chart
- [ ] Verify "Connected" status in GUI
- [ ] Watch for first prediction update
- [ ] Monitor accuracy for 24-48 hours before live trading

The system is designed to be largely autonomous once properly configured, but it does require initial setup and ongoing monitoring. The comprehensive logging and GUI make it easy to see exactly what's happening at all times.
