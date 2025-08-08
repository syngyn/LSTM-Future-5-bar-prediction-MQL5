# ALFA / Transformer EA - Universal Edition

## ✅ Installation Complete!

System configured for: **Windows 10**

---

### 🚀 Quick Start Guide

1.  **IMPORTANT: Configure Your Model**
    - Open `train_enhanced_model.py` and `daemon.py` in a text editor.
    - At the top of each file, find the `SELECTED_MODEL` variable.
    - Set it to either `'ALFA'` or `'TRANSFORMER'`.
    - **You must use the same model for both training and the daemon.**

2.  **Run the Launcher**
    - **On Windows:** Double-click `ALFA_Launcher.bat`
    - **On macOS/Linux:** Open a terminal, navigate to this folder, and run `./ALFA_Launcher.sh`

3.  **Train the Model**
    - In the launcher, choose option `1` and press Enter.
    - Wait for the training process to complete. This may take some time.

4.  **Start the Daemon**
    - In the launcher, choose option `2` and press Enter.
    - The daemon will now run in the background, waiting for requests from MetaTrader 5.

5.  **Set up MetaTrader 5**
    - Copy the `ALFA_Transformer_EA.mq5` file to your MT5 `Experts` folder.
    - Open MetaEditor, open the EA file, and click "Compile".
    - Attach the compiled EA to a EURUSD, H1 chart.

---

### Backtesting

1.  After training a model, run the launcher and choose option `3` to generate `backtest_predictions.csv`.
2.  Copy this CSV file to your MT5 `Common\Files` directory.
3.  Run the EA in the Strategy Tester. It will automatically use the predictions from the file.

### Troubleshooting

- If you have issues, re-run this installer by choosing option `4` in the launcher.
- Ensure your MetaTrader 5 terminal has permission to access the file system (Tools -> Options -> Expert Advisors -> "Allow file read and write").
- Check the "Experts" and "Journal" tabs in MT5 for error messages from the EA.
