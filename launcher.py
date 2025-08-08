#!/usr/bin/env python3
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
