#!/usr/bin/env python3
"""
Hybrid ALFA-Transformer EA - Universal Installer
=================================================
This script installs all necessary Python packages, creates launchers,
automatically downloads and formats the required training data, and
generates a README file for the trading system.
"""
import os
import sys
import platform
import subprocess
from pathlib import Path

def install():
    """Main installation function."""
    print("=" * 60)
    print("   Hybrid ALFA-Transformer EA - Universal Installer")
    print("=" * 60)
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
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package, "--upgrade"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
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
    
    (script_dir / "models").mkdir(exist_ok=True)
    (script_dir / "data").mkdir(exist_ok=True)
    
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
    print("   Hybrid ALFA-Transformer EA Control Panel")
    print("==============================================")
    print("  1. Train Hybrid Model (Run this first!)")
    print("  2. Start Daemon         (For live trading)")
    print("  3. Generate Backtest    (For strategy testing)")
    print("  4. Re-run Installer / Update Data")
    print()
    
    while True:
        choice = input("Command (1-4, or 'q' to quit): ").strip().lower()
        
        if choice in ['q', 'quit']:
            break
        elif choice in ['1', 'train']:
            subprocess.run([sys.executable, script_dir / "train_alpha.py"])
        elif choice in ['2', 'daemon']:
            subprocess.run([sys.executable, script_dir / "alpha.daemon.py"])
        elif choice in ['3', 'backtest']:
            subprocess.run([sys.executable, script_dir / "generate_alpha_backtest.py"])
        elif choice in ['4', 'install']:
            subprocess.run([sys.executable, script_dir / "install_alpha.py"])
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
        batch_file = script_dir / "Hybrid_EA_Launcher.bat"
        batch_content = (
            "@echo off\n"
            "title Hybrid ALFA-Transformer EA Launcher\n"
            f"cd /d \"{script_dir}\"\n"
            "echo Running Hybrid ALFA-Transformer EA Launcher...\n"
            "python launcher.py\n"
            "pause\n"
        )
        try:
            with open(batch_file, 'w', encoding='utf-8') as f:
                f.write(batch_content)
            print(f"✅ Windows launcher created: {batch_file.name}")
        except Exception as e:
            print(f"❌ Failed to create {batch_file.name}: {e}")
    else: # For Linux and macOS
        shell_file = script_dir / "Hybrid_EA_Launcher.sh"
        shell_content = (
            "#!/bin/bash\n"
            f"cd \"{script_dir}\"\n"
            "python3 launcher.py\n"
        )
        try:
            with open(shell_file, 'w', encoding='utf-8') as f:
                f.write(shell_content)
            shell_file.chmod(0o755)
            print(f"✅ Unix/macOS launcher created: {shell_file.name}")
        except Exception as e:
            print(f"❌ Failed to create {shell_file.name}: {e}")
    print()

    # 6. AUTOMATICALLY DOWNLOAD AND FORMAT TRAINING DATA
    print("📉 Downloading and preparing training data (EURUSD60.csv)...")
    data_file = script_dir / "EURUSD60.csv"
    try:
        import yfinance as yf
        print("   - Downloading up to 2 years of hourly EURUSD data from Yahoo Finance...")
        eurusd = yf.download("EURUSD=X", period="2y", interval="1h", auto_adjust=True)
        
        if eurusd.empty:
            raise ValueError("No data downloaded from yfinance. Check ticker or network connection.")
            
        eurusd.reset_index(inplace=True)
        # Rename columns to be compatible with MT5/EA's expectations
        eurusd.rename(columns={'Datetime': 'Date', 'Open': 'Open', 'High': 'High', 
                               'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True)
        
        # Add MT5-specific columns if they don't exist
        if 'Tickvol' not in eurusd.columns: eurusd['Tickvol'] = eurusd['Volume']
        if 'Spread' not in eurusd.columns: eurusd['Spread'] = 2
        
        # Ensure correct column order
        final_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Tickvol', 'Volume', 'Spread']
        eurusd = eurusd[[col for col in final_cols if col in eurusd.columns]]
        
        eurusd.to_csv(data_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
        print(f"   ✅ Sample data downloaded and saved to {data_file.name} successfully.")
    except Exception as e:
        print(f"   ❌ FATAL: Could not automatically download data: {e}")
        print(f"      The system cannot proceed without training data.")
        print(f"      Please check your internet connection and try again.")
        return False
    print()

    # 7. Create a README file
    print("📄 Creating documentation (README.md)...")
    readme = script_dir / "README.md"
    readme_content = (
        "# Hybrid ALFA-Transformer EA - Integrated Edition\n\n"
        "## ✅ Installation Complete!\n\n"
        f"System configured for: **{platform.system()} {platform.release()}**\n\n"
        "This system uses a single, powerful **Hybrid Model** that combines the strengths of Attention-based LSTMs (ALFA) and Transformers. You no longer need to choose a model type.\n\n"
        "The required training data (`EURUSD60.csv`) has been automatically downloaded.\n\n"
        "---\n\n"
        "### 🚀 Quick Start Guide\n\n"
        "1.  **Run the Launcher**\n"
        "    - **On Windows:** Double-click `Hybrid_EA_Launcher.bat`\n"
        "    - **On macOS/Linux:** Open a terminal and run `./Hybrid_EA_Launcher.sh`\n\n"
        "2.  **Train the Hybrid Model**\n"
        "    - In the launcher, choose option `1` and press Enter.\n\n"
        "3.  **Start the Daemon**\n"
        "    - In the launcher, choose option `2` and press Enter.\n\n"
        "4.  **Set up MetaTrader 5**\n"
        "    - Copy `Hybrid_ALFA_Transformer_EA.mq5` to your MT5 `Experts` folder and compile it.\n"
        "    - Attach the EA to a EURUSD, H1 chart.\n\n"
        "---\n\n"
        "### Updating Training Data\n\n"
        "To get the latest market data for training, simply run the installer again by choosing option `4` in the launcher. This will re-download the `EURUSD60.csv` file.\n\n"
        "### Backtesting\n\n"
        "1.  After training, run the launcher and choose option `3` to generate `backtest_predictions.csv`.\n"
        "2.  Copy this CSV to your MT5 `Common\\Files` directory.\n"
        "3.  Run the EA in the Strategy Tester.\n"
    )

    try:
        with open(readme, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print("✅ README.md created.")
    except Exception as e:
        print(f"❌ Failed to create README.md: {e}")
    print()

    print("=" * 60)
    print(" INSTALLATION AND DATA DOWNLOAD COMPLETE!")
    print("=" * 60)
    print("The system has been configured with the integrated Hybrid Model.")
    print()
    print("👇 YOUR NEXT STEPS:")
    if platform.system().lower() == "windows":
        print("1. Double-click 'Hybrid_EA_Launcher.bat' to start.")
    else:
        print("1. Run './Hybrid_EA_Launcher.sh' in your terminal to start.")
    print("2. Choose option '1' to train your model.")
    print()

    return True

if __name__ == "__main__":
    success = install()
    if not success:
        input("Press Enter to exit...")