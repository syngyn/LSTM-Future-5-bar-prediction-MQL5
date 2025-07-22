import os
import json
import glob
from datetime import datetime

# Path to your data directory
DATA_DIR = r"C:\Users\jason\AppData\Roaming\MetaQuotes\Terminal\5C659F0E64BA794E712EE4C936BCFED5\MQL5\Files\LSTM_Trading\data"

def debug_json_files():
    """Debug JSON files to find formatting issues"""
    
    print("🔍 JSON Debug Tool")
    print("=" * 50)
    
    # Check all JSON files in data directory
    request_files = glob.glob(os.path.join(DATA_DIR, "request_*.json"))
    
    if not request_files:
        print("❌ No request files found")
        print(f"📁 Checked directory: {DATA_DIR}")
        return
    
    print(f"📁 Found {len(request_files)} request files")
    
    for i, filepath in enumerate(request_files[:5]):  # Check first 5 files
        print(f"\n📄 File {i+1}: {os.path.basename(filepath)}")
        print("-" * 30)
        
        try:
            # Read raw content
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"📏 File size: {len(content)} characters")
            print(f"📝 Content preview:")
            print(content[:300] + ("..." if len(content) > 300 else ""))
            
            # Try to parse JSON
            try:
                data = json.loads(content)
                print("✅ JSON is valid")
                print(f"🔑 Keys: {list(data.keys())}")
                
                # Check required fields
                required_fields = ["request_id", "action", "features", "current_price", "atr"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    print(f"⚠️  Missing fields: {missing_fields}")
                else:
                    print("✅ All required fields present")
                    
                # Check data types
                if "features" in data:
                    features = data["features"]
                    if isinstance(features, list):
                        print(f"📊 Features: {len(features)} items")
                        if len(features) > 0:
                            print(f"📈 Feature range: {min(features):.6f} to {max(features):.6f}")
                    else:
                        print(f"❌ Features is not a list: {type(features)}")
                
                if "current_price" in data:
                    print(f"💰 Current price: {data['current_price']}")
                    
                if "atr" in data:
                    print(f"📏 ATR: {data['atr']}")
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON Error: {e}")
                print(f"📍 Error location: line {e.lineno}, column {e.colno}")
                
                # Show the problematic line
                lines = content.split('\n')
                if e.lineno <= len(lines):
                    problematic_line = lines[e.lineno - 1]
                    print(f"🔴 Problematic line: {problematic_line}")
                    print(f"🔴 Error position: {' ' * (e.colno - 1)}^")
                
        except Exception as e:
            print(f"💥 Error reading file: {e}")
    
    print(f"\n🔧 Recommendations:")
    print("1. Check EA JSON formatting in WriteJsonRequest() function")
    print("2. Ensure all floating point numbers are properly formatted")
    print("3. Check for trailing commas or missing quotes")
    print("4. Verify array formatting for features")

if __name__ == "__main__":
    debug_json_files()