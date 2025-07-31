# LSTM-Future-5-bar-prediction-MQL5
Updated LSTM MQL5 ai trading bot
🚀 Complete GGTH LSTM Expert Advisor Installation Guide

📋 What You'll Have After This Guide:

✅ Python installed and working
✅ LSTM trading system running
✅ Expert Advisor predicting EUR/USD prices
✅ Real-time predictions in MetaTrader
✅ Confidence scores and trading signals


⏰ Time Required: 2-3 hours (first time)

🔧 PART 1: INSTALLING PYTHON
Step 1.1: Download Python

Open your web browser (Chrome, Edge, Firefox)
Go to: https://www.python.org/downloads/
Click the big yellow button that says "Download Python 3.11.x"
Wait for download to complete (about 30MB file)

Step 1.2: Install Python

Find the downloaded file (usually in Downloads folder)
Right-click the file → "Run as administrator"
IMPORTANT: ✅ Check "Add Python to PATH" (very important!)
Click "Install Now"
Wait 5-10 minutes for installation
Click "Close" when done

Step 1.3: Test Python Installation

Press Windows key + R
Type: cmd and press Enter
Type: python --version and press Enter
Should show: Python 3.11.x (if you see this, Python is working!)
Type: exit to close

❌ If you see "python is not recognized":

Python PATH wasn't added properly
Restart computer and try again
If still broken, uninstall Python and reinstall with "Add to PATH" checked


📦 PART 2: INSTALLING REQUIRED LIBRARIES
Step 2.1: Open Command Prompt

Press Windows key
Type: cmd
Right-click "Command Prompt" → "Run as administrator"
Click "Yes" if Windows asks for permission

Step 2.2: Install Libraries (One by One)
Copy and paste each command exactly, press Enter, wait for completion:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
⏳ Wait 2-5 minutes (installing PyTorch)
pip install pandas numpy scikit-learn
⏳ Wait 1-2 minutes
bashpip install joblib watchdog
⏳ Wait 30 seconds

Step 2.3: Verify Installation
Type this command:
python -c "import torch, pandas, numpy, sklearn; "
❌ If error: One of the installations failed, try reinstalling that specific library
Close Command Prompt by typing exit

💻 PART 3: SETTING UP METATRADER 5
Step 3.1: Download MetaTrader 5

Go to: https://www.metatrader5.com/en/download
Click "Download MetaTrader 5"
Run the installer when downloaded
Follow installation wizard (default settings are fine)

Step 3.2: Open Demo Account

Open MetaTrader 5
Click "File" → "Open Account"
Select any broker (e.g., "MetaQuotes Demo")
Click "Next"
Select "Demo Account"
Fill in your details (name, email, phone)
Click "Next" and "Finish"

Step 3.3: Find Your Data Folder

In MetaTrader, clickfile → Open data Folder
You should see a folder that opens with a long name like:
C:\Users\YourName\AppData\Roaming\MetaQuotes\Terminal\ABC123DEF456\
Keep this folder open - you'll need it soon!


📁 PART 4: SETTING UP THE PROJECT
Step 4.1: Create Project Folder

In the MetaTrader data folder (from Step 3.3)
Go to: C:\Users\YourName\AppData\Roaming\MetaQuotes\Terminal\ABC123DEF456\MQL5\Files\
Create new folder: Right-click → New → Folder
Name it: LSTM_Trading
Open the LSTM_Trading folder

save daemon.py in your LSTM_trading Folder

save train_enhanced_model.py in your LSTM_trading folder

save generate_backtest.py in your LSTM_trading Folder


Step 4 : Create Required Folders
In your LSTM_Trading folder, create these folders:

models (right-click → New → Folder → name it "models")
data (right-click → New → Folder → name it "data")

Your folder should now look like:
LSTM_Trading/
├── daemon.py
├── generate_backtest.py
├── train_enhanced_model.py  
├── models/ (empty folder)
└── data/ (empty folder)

📊 PART 5: GETTING TRAINING DATA
Place all the currency pair CSV files in your LSTM_Trading folder EURUSD60.csv, USDJPY60.csv, EURGBP60.csv, USDCHF60.csv, EURJPY60.csv, USDCAD60.csv

🧠 PART 6: TRAINING THE MODEL
Step 6.1: Run Training Script
In the same Command Prompt:
python train_enhanced_model.py
⏳ This will take 30-60 minutes
You should see:
🚀 Enhanced LSTM Training Script v2.1
✅ Loaded 77,623 data points
🎯 Training enhanced main model...
Epoch   1/50 | Train: 1.0234 | Val: 0.9876
...
⏹️  Early stopping at epoch 28
🎉 TRAINING COMPLETE!
✅ Success indicators:

No red error messages
Shows "TRAINING COMPLETE!"
Creates files in the "models" folder

❌ If errors occur:

Check that all listed csv files exists and has data
Make sure you're in the correct folder
Verify Python libraries are installed

To use strategy tester 

in cmd prompt run 
pip install yfinance pandas numpy torch scikit-learn joblib

Those packages are required to run the backtest now run
python generate_backtest.py
take the .csv file and place it here
C:\Users\username\AppData\Roaming\MetaQuotes\Terminal\Common\Files

🤖 PART 7: SETTING UP THE EXPERT ADVISOR
Step 7.1: Compile Expert Advisor File

In MetaTrader 5, click on tools and select MetaQuotes Language Editor
Open the GGTH.mq5 file
Click "Compile" button (or press F7)
Should see: "0 error(s), 0 warning(s)" at bottom
Close MetaEditor

Step 7.2: Start the Daemon

Open Command Prompt as administrator
Navigate to LSTM_Trading folder (same as before)
Run daemon:
python daemon.py


Should see:
🚀 Using device: cpu
✅ Enhanced AttentionLSTM model loaded successfully
✅ Loaded 3 ensemble models
🎯 Enhanced LSTM Daemon v2.1 is running!
⚠️ LEAVE THIS WINDOW OPEN - the daemon must run continuously
Step 7.3: Attach EA to Chart

In MetaTrader, open EURUSD H1 chart
In Navigator panel (left side), find "Expert Advisors"
Drag "GGTH" onto the EURUSD chart
In settings dialog:

✅ Check "Allow automated trading"
✅ Check "Allow DLL imports"
Set IsLSTMActive = true
Set EnableDetailedLogging = true
Click OK


Should see: 😊 smiley face in top-right corner of chart


🎯 PART 8: TESTING THE SYSTEM
Step 8.1: Check if Everything Works
In MetaTrader:

Open "Experts" tab (bottom of screen)
Should see messages like:
📊 === NEW BAR DETECTED ===
🕐 TIME SYNC DEBUG:
✅ Prediction successful - Confidence: 0.847


In Command Prompt (daemon window):
✅ request_123 | Conf: 0.847 | Source: ensemble_4
Step 8.2: Understanding the Output
Good Signs: ✅

Confidence scores 0.6-0.9
Source: "ensemble_4"
No timezone warnings
Predictions every hour

Bad Signs: ❌

Confidence scores below 0.3
Source: "fallback"
Many error messages
Timezone warnings


🛠️ PART 9: TROUBLESHOOTING
Common Issues:
❌ "Python is not recognized"
Solution:

Reinstall Python with "Add to PATH" checked
Restart computer
Try again

❌ "Module not found" errors
Solution:

Open cmd as administrator
Run: pip install [missing_module_name]

❌ EA shows "LSTM inactive"
Solution:

Check that daemon.py is running
Verify files are in correct folder
Set IsLSTMActive = true in EA settings

❌ Low confidence scores (below 0.3)
Solution:

Retrain model: python train_enhanced_model.py
Check timezone conversion was successful
Ensure you have recent data (last few months)

❌ Daemon crashes or stops
Solution:

Check Command Prompt for error messages
Restart daemon: python daemon.py
Verify model files exist in models/ folder


📈 PART 10: MONITORING PERFORMANCE
Step 10.1: Daily Checks
Every day, check:

Daemon is still running (Command Prompt window open)
EA is active (smiley face on chart)
Confidence scores are reasonable (0.5+)
Predictions update every hour

Step 10.2: Weekly Maintenance
Every week:

Update training data (export new H1 data from MetaTrader)
Retrain model: python train_enhanced_model.py
Restart daemon to load new model

Step 10.3: Performance Metrics
Good Performance:

Confidence scores: 0.6-0.9 average
Prediction accuracy: 65-75%
Consistent ensemble predictions
No frequent errors

Poor Performance:

Confidence scores: Below 0.4
Many fallback predictions
Frequent timezone warnings
System crashes


🎉 CONGRATULATIONS!
You now have a complete LSTM Expert Advisor system running!
What Your System Does:
✅ Analyzes EUR/USD price patterns using AI
✅ Predicts future prices for next 5 hours
✅ Provides confidence scores for each prediction
✅ Updates automatically every hour
✅ Uses ensemble models for robust predictions
Quick Reference Commands:
Start daemon: python daemon.py
Retrain model: python train_enhanced_model.py
Fix data: python fix_csv_timezone.py
Important Files Location:
MetaTrader Data Folder\MQL5\Files\LSTM_Trading\
├── daemon.py (must be running)
├── EURUSD60.csv (your training data)
├── models\ (contains trained AI models)
└── data\ (communication with EA)
