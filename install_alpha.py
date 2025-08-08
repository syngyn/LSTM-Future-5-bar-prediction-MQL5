#!/usr/bin/env python3
"""
ALFA / Transformer EA - Universal Installer
===========================================
This script installs all necessary Python packages, creates launchers,
downloads sample data, and generates a README file for the trading system.
"""
import os
import sys
import platform
import subprocess
from pathlib import Path

def install():
    """Main installation function."""
    print("=" * 50)
    print(" ALFA / Transformer EA - Universal Installer")
    print("=" * 50)
    print(f"System: {platform.system()} {platform.release()}")
    print()

    # 1. Check Python version
    if sys.version_info < (3, 7):
        print(f"❌ ERROR: Python 3.7+ is required.")
        print(f"   Your version is: {sys.version}")
        return False
    
    print(f"✅ Python version OK: {sys.version_info.major}.{sys.version_info.minor}")
    print()

    # 2. Install required Python packages
    print("📦 Installing required Python packages...")
    packages = [
        "torch>=1.9.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.0.0",
        "yfinance>=0.1.70"
    ]
    
    for package in packages:
        try:
            print(f"   - Installing {package}...")
            # Use check_call to ensure pip commands succeed
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE  # Capture stderr to show on failure
            )
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️ WARNING: Failed to install {package}.")
            print(f"      PIP Error: {e.stderr.decode('utf-8', errors='ignore').strip()}")
            print(f"      Please try installing it manually: pip install \"{package}\"")
        except Exception as e:
            print(f"   ⚠️ An unexpected error occurred while installing {package}: {e}")

    print("✅ Package installation process complete.")
    print()

    # 3. Setup paths and directories
    print("📂 Setting up required directories...")
    script_dir = Path(__file__).parent.resolve()
    
    # Create 'models' and 'data' directories if they don't exist
    (script_dir / "models").mkdir(exist_ok=True)
    (script_dir / "data").mkdir(exist_ok=True) # Used for MQL5 communication fallback
    
    print(f"   - Models directory: {script_dir / 'models'}")
    print(f"   - Data directory:   {script_dir / 'data'}")
    print("✅ Directories are ready.")
    print()

    # 4. Create the main launcher script
    print("🚀 Creating user-friendly launcher (launcher.py)...")
    launcher_py = script_dir / "launcher.py"
    launcher_content = """#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path

def main():
    script_dir = Path(__file__).parent
    
    print("==============================================")
    print("   ALFA / Transformer EA Control Panel")
    print("==============================================")
    print("  1. Train Model     (Run this first!)")
    print("  2. Start Daemon      (For live trading)")
    print("  3. Generate Backtest (For strategy testing)")
    print("  4. Re-run Installer")
    print()
    
    while True:
        choice = input("Command (1-4, or 'q' to quit): ").strip().lower()
        
        if choice in ['q', 'quit']:
            break
        elif choice in ['1', 'train']:
            subprocess.run([sys.executable, script_dir / "train_enhanced_model.py"])
        elif choice in ['2', 'daemon']:
            subprocess.run([sys.executable, script_dir / "daemon.py"])
        elif choice in ['3', 'backtest']:
            subprocess.run([sys.executable, script_dir / "generate_backtest.py"])
        elif choice in ['4', 'install']:
            subprocess.run([sys.executable, script_dir / "install.py"])
        else:
            print("Invalid choice. Please enter a number from 1 to 4.")

if __name__ == "__main__":
    main()
"""
    try:
        with open(launcher_py, 'w', encoding='utf-8') as f:
            f.write(launcher_content)
        print("✅ launcher.py created successfully.")
    except Exception as e:
        print(f"❌ Failed to create launcher.py: {e}")
    print()
    
    # 5. Create system-specific launchers (.bat or .sh)
    print("🛰️  Creating system-specific launchers...")
    if platform.system().lower() == "windows":
        batch_file = script_dir / "ALFA_Launcher.bat"
        batch_content = (
            "@echo off\n"
            "title ALFA Transformer EA Launcher\n"
            f"cd /d \"{script_dir}\"\n"
            "echo Running ALFA / Transformer EA Launcher...\n"
            "python launcher.py\n"
            "pause\n"
        )
        try:
            with open(batch_file, 'w', encoding='utf-8') as f:
                f.write(batch_content)
            print("✅ Windows launcher created: ALFA_Launcher.bat")
        except Exception as e:
            print(f"❌ Failed to create ALFA_Launcher.bat: {e}")
    else: # For Linux and macOS
        shell_file = script_dir / "ALFA_Launcher.sh"
        shell_content = (
            "#!/bin/bash\n"
            f"cd \"{script_dir}\"\n"
            "python3 launcher.py\n"
        )
        try:
            with open(shell_file, 'w', encoding='utf-8') as f:
                f.write(shell_content)
            # Make the shell script executable
            shell_file.chmod(0o755)
            print("✅ Unix/macOS launcher created: ALFA_Launcher.sh")
        except Exception as e:
            print(f"❌ Failed to create ALFA_Launcher.sh: {e}")
    print()

    # 6. Download sample training data
    print("📉 Setting up sample training data (EURUSD60.csv)...")
    data_file = script_dir / "EURUSD60.csv"
    if not data_file.exists():
        try:
            import yfinance as yf
            print("   - Downloading 2 years of hourly EURUSD data from Yahoo Finance...")
            eurusd = yf.download("EURUSD=X", period="2y", interval="1h")
            eurusd.reset_index(inplace=True)
            # Rename columns to be compatible with the EA's expectations
            eurusd.rename(columns={'Datetime': 'Date', 'Open': 'Open', 'High': 'High', 
                                   'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True)
            # Add MT5-specific columns if they don't exist
            if 'Tickvol' not in eurusd.columns: eurusd['Tickvol'] = eurusd['Volume']
            if 'Spread' not in eurusd.columns: eurusd['Spread'] = 2
            
            final_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Tickvol', 'Volume', 'Spread']
            eurusd = eurusd[final_cols]
            
            eurusd.to_csv(data_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
            print("   ✅ Sample data downloaded successfully.")
        except Exception as e:
            print(f"   ⚠️ Could not automatically download data: {e}")
            print(f"   💡 Please manually place your 'EURUSD60.csv' file in this directory: {script_dir}")
    else:
        print("   ✅ Training data file (EURUSD60.csv) already exists.")
    print()

    # 7. Create a README file
    print("📄 Creating documentation (README.md)...")
    readme = script_dir / "README.md"
    
    # *** CORRECTED README CONTENT DEFINITION ***
    # This new method avoids the triple-quoted f-string syntax error.
    readme_content = (
        "# ALFA / Transformer EA - Universal Edition\n\n"
        "## ✅ Installation Complete!\n\n"
        f"System configured for: **{platform.system()} {platform.release()}**\n\n"
        "---\n\n"
        "### 🚀 Quick Start Guide\n\n"
        "1.  **IMPORTANT: Configure Your Model**\n"
        "    - Open `train_enhanced_model.py` and `daemon.py` in a text editor.\n"
        "    - At the top of each file, find the `SELECTED_MODEL` variable.\n"
        "    - Set it to either `'ALFA'` or `'TRANSFORMER'`.\n"
        "    - **You must use the same model for both training and the daemon.**\n\n"
        "2.  **Run the Launcher**\n"
        "    - **On Windows:** Double-click `ALFA_Launcher.bat`\n"
        "    - **On macOS/Linux:** Open a terminal, navigate to this folder, and run `./ALFA_Launcher.sh`\n\n"
        "3.  **Train the Model**\n"
        "    - In the launcher, choose option `1` and press Enter.\n"
        "    - Wait for the training process to complete. This may take some time.\n\n"
        "4.  **Start the Daemon**\n"
        "    - In the launcher, choose option `2` and press Enter.\n"
        "    - The daemon will now run in the background, waiting for requests from MetaTrader 5.\n\n"
        "5.  **Set up MetaTrader 5**\n"
        "    - Copy the `ALFA_Transformer_EA.mq5` file to your MT5 `Experts` folder.\n"
        "    - Open MetaEditor, open the EA file, and click \"Compile\".\n"
        "    - Attach the compiled EA to a EURUSD, H1 chart.\n\n"
        "---\n\n"
        "### Backtesting\n\n"
        "1.  After training a model, run the launcher and choose option `3` to generate `backtest_predictions.csv`.\n"
        "2.  Copy this CSV file to your MT5 `Common\\Files` directory.\n"
        "3.  Run the EA in the Strategy Tester. It will automatically use the predictions from the file.\n\n"
        "### Troubleshooting\n\n"
        "- If you have issues, re-run this installer by choosing option `4` in the launcher.\n"
        "- Ensure your MetaTrader 5 terminal has permission to access the file system (Tools -> Options -> Expert Advisors -> \"Allow file read and write\").\n"
        "- Check the \"Experts\" and \"Journal\" tabs in MT5 for error messages from the EA.\n"
    )

    try:
        with open(readme, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print("✅ README.md created.")
    except Exception as e:
        print(f"❌ Failed to create README.md: {e}")
    print()

    # Final summary
    print("=" * 50)
    print(" INSTALLATION COMPLETE!")
    print("=" * 50)
    print("System has been configured and is ready to use.")
    print()
    print("👇 YOUR NEXT STEPS:")
    print("1. CRITICAL: Open the Python files and set your desired `SELECTED_MODEL`.")
    if platform.system().lower() == "windows":
        print("2. Double-click 'ALFA_Launcher.bat' to start.")
    else:
        print("2. Run './ALFA_Launcher.sh' in your terminal to start.")
    print("3. Follow the menu to Train your model, then Start the daemon.")
    print()

    return True

if __name__ == "__main__":
    success = install()
    if not success:
        input("Press Enter to exit...")