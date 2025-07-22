import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import shutil

def fix_csv_format(input_file, output_file=None):
    """Fix MetaTrader CSV format - CORRECTED VERSION"""
    
    if output_file is None:
        output_file = input_file.replace('.csv', '_fixed.csv')
    
    print(f"🔧 Fixing CSV format: {input_file}")
    
    try:
        # Read the file
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"📄 Total lines in file: {len(lines)}")
        print("📄 First 3 lines:")
        for i, line in enumerate(lines[:3]):
            print(f"   {i+1}: {repr(line.strip())}")
        
        fixed_lines = []
        skipped_lines = 0
        processed_lines = 0
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Skip header line containing column names
            if 'DATE' in line.upper() and 'TIME' in line.upper() and 'OPEN' in line.upper():
                print(f"   📋 Skipping header line {line_num + 1}")
                continue
            
            processed_lines += 1
            
            # Split by tab
            parts = line.split('\t')
            
            # Clean up parts
            parts = [part.strip().replace('<', '').replace('>', '') for part in parts]
            
            # Need at least 6 parts: DATE, TIME, OPEN, HIGH, LOW, CLOSE
            if len(parts) < 6:
                if line_num < 10:
                    print(f"   ⚠️  Line {line_num + 1}: Only {len(parts)} columns, need 6+")
                skipped_lines += 1
                continue
            
            try:
                # Parse components
                date_str = parts[0]  # e.g., "2013.01.01"
                time_str = parts[1]  # e.g., "23:00:00"
                open_str = parts[2]
                high_str = parts[3]
                low_str = parts[4]
                close_str = parts[5]
                
                # Parse date (YYYY.MM.DD format)
                try:
                    year, month, day = date_str.split('.')
                    date_obj = datetime(int(year), int(month), int(day))
                except ValueError as e:
                    if line_num < 10:
                        print(f"   ⚠️  Line {line_num + 1}: Date parse error '{date_str}': {e}")
                    skipped_lines += 1
                    continue
                
                # Parse time (HH:MM:SS format)
                try:
                    hour, minute, second = time_str.split(':')
                    time_obj = datetime.strptime(f"{hour}:{minute}:{second}", '%H:%M:%S').time()
                except ValueError as e:
                    if line_num < 10:
                        print(f"   ⚠️  Line {line_num + 1}: Time parse error '{time_str}': {e}")
                    skipped_lines += 1
                    continue
                
                # Combine date and time
                dt = datetime.combine(date_obj.date(), time_obj)
                
                # Parse and validate prices
                try:
                    open_price = float(open_str)
                    high_price = float(high_str)
                    low_price = float(low_str)
                    close_price = float(close_str)
                    
                    # Basic OHLC validation
                    if not (low_price <= min(open_price, close_price) and 
                           high_price >= max(open_price, close_price) and
                           low_price <= high_price):
                        if line_num < 10:
                            print(f"   ⚠️  Line {line_num + 1}: Invalid OHLC relationship")
                        skipped_lines += 1
                        continue
                        
                except ValueError as e:
                    if line_num < 10:
                        print(f"   ⚠️  Line {line_num + 1}: Price parse error: {e}")
                    skipped_lines += 1
                    continue
                
                # Parse volume data (optional)
                tickvol = "0"
                vol = "0" 
                spread = "0"
                
                if len(parts) > 6:
                    try:
                        tickvol = str(int(float(parts[6])))
                    except:
                        tickvol = "0"
                        
                if len(parts) > 7:
                    try:
                        vol = str(int(float(parts[7])))
                    except:
                        vol = "0"
                        
                if len(parts) > 8:
                    try:
                        spread = str(int(float(parts[8])))
                    except:
                        spread = "0"
                
                # Create CSV line
                csv_line = f"{dt.strftime('%Y-%m-%d %H:%M:%S')},{open_price:.5f},{high_price:.5f},{low_price:.5f},{close_price:.5f},{tickvol},{vol},{spread}"
                fixed_lines.append(csv_line)
                
                # Show progress
                if len(fixed_lines) % 10000 == 0:
                    print(f"   📈 Processed {len(fixed_lines):,} records...")
                
            except Exception as e:
                if line_num < 10:
                    print(f"   ⚠️  Line {line_num + 1}: General error: {e}")
                skipped_lines += 1
                continue
        
        if not fixed_lines:
            print("❌ No valid data lines found!")
            print("🔍 Debug info:")
            print(f"   Total lines: {len(lines)}")
            print(f"   Processed lines: {processed_lines}")
            print(f"   Skipped lines: {skipped_lines}")
            return None
        
        # Write the fixed CSV
        print(f"💾 Writing fixed CSV to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write('Date,Open,High,Low,Close,TickVol,Vol,Spread\n')
            # Write data
            for line in fixed_lines:
                f.write(line + '\n')
        
        print(f"✅ Successfully fixed CSV!")
        print(f"📊 Results:")
        print(f"   📈 Valid records: {len(fixed_lines):,}")
        print(f"   ⚠️  Skipped records: {skipped_lines:,}")
        print(f"   📁 Output file: {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"💥 Error fixing CSV: {e}")
        import traceback
        traceback.print_exc()
        return None

def convert_timezone(csv_file, source_tz_offset, target_tz_offset, output_file=None):
    """Convert timezone in CSV file"""
    
    if output_file is None:
        output_file = csv_file.replace('.csv', '_timezone_converted.csv')
    
    print(f"\n🌍 Converting timezone from GMT{source_tz_offset:+d} to GMT{target_tz_offset:+d}")
    
    try:
        # Read CSV
        print(f"📖 Reading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)
        
        print(f"📊 Loaded {len(df):,} records")
        print(f"🔧 Columns: {list(df.columns)}")
        
        if 'Date' not in df.columns:
            print("❌ No 'Date' column found")
            return None
        
        # Parse dates
        print(f"📅 Parsing dates...")
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"📅 Original date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Calculate timezone difference
        tz_diff_hours = target_tz_offset - source_tz_offset
        print(f"⏰ Adjusting timestamps by {tz_diff_hours:+d} hours")
        
        # Apply timezone conversion
        df['Date'] = df['Date'] + timedelta(hours=tz_diff_hours)
        
        print(f"📅 New date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Save converted file
        print(f"💾 Saving converted file...")
        df.to_csv(output_file, index=False)
        
        print(f"✅ Timezone conversion complete!")
        print(f"📁 Output file: {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"💥 Error converting timezone: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_csv_timezone(csv_file):
    """Analyze timezone patterns in CSV"""
    
    print(f"\n📊 TIMEZONE ANALYSIS: {csv_file}")
    print("-" * 50)
    
    try:
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"📈 Total records: {len(df):,}")
        print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Analyze hourly patterns
        hour_counts = df['Date'].dt.hour.value_counts().sort_index()
        
        print(f"\n📈 Hourly Activity Pattern:")
        print("Hour | Count    | %     | Chart")
        print("-" * 35)
        
        total_records = len(df)
        for hour in range(24):
            count = hour_counts.get(hour, 0)
            percentage = (count / total_records) * 100 if total_records > 0 else 0
            bar = "█" * int(percentage / 3)  # Scale bar
            print(f"{hour:2d}   | {count:8,} | {percentage:4.1f}% | {bar}")
        
        # Find peak hours
        peak_hours = hour_counts.nlargest(8).index.tolist()
        peak_hours.sort()
        
        print(f"\n🎯 Peak activity hours: {peak_hours}")
        
        # Timezone detection
        print(f"\n🌍 Timezone Detection:")
        
        # Check different timezone patterns
        timezone_patterns = {
            "GMT+0 (UTC)": [0, 1, 6, 7, 8, 13, 14, 15],
            "GMT+1 (CET)": [1, 2, 7, 8, 9, 14, 15, 16], 
            "GMT+2 (EET)": [2, 3, 8, 9, 10, 15, 16, 17],
            "GMT+3 (MSK)": [3, 4, 9, 10, 11, 16, 17, 18],
            "GMT-5 (EST)": [19, 20, 1, 2, 3, 8, 9, 10],
            "GMT-7 (MST)": [17, 18, 23, 0, 1, 6, 7, 8]
        }
        
        best_match = ""
        best_score = 0
        
        for tz_name, expected_hours in timezone_patterns.items():
            # Calculate how many peak hours match expected hours
            matches = len(set(peak_hours) & set(expected_hours))
            score = matches / len(expected_hours)
            
            print(f"   {tz_name:15s}: {matches}/{len(expected_hours)} matches ({score:.1%})")
            
            if score > best_score:
                best_score = score
                best_match = tz_name
        
        print(f"\n🎯 Best match: {best_match} ({best_score:.1%} confidence)")
        
        return best_match, best_score
        
    except Exception as e:
        print(f"💥 Error analyzing timezone: {e}")
        return None, 0

def main():
    """Main function"""
    
    print("🚀 CORRECTED CSV Fixer and Timezone Converter")
    print("=" * 60)
    
    # Find CSV files
    csv_files = [f for f in os.listdir('.') if f.lower().endswith('.csv') and 'eurusd' in f.lower()]
    
    if not csv_files:
        print("❌ No EURUSD CSV files found")
        return
    
    input_file = csv_files[0]
    print(f"📁 Processing: {input_file}")
    
    # Create backup
    backup_file = input_file.replace('.csv', '_backup.csv')
    try:
        shutil.copy2(input_file, backup_file)
        print(f"💾 Backup created: {backup_file}")
    except Exception as e:
        print(f"⚠️  Could not create backup: {e}")
    
    # Step 1: Fix CSV format
    print(f"\n" + "="*50)
    print("STEP 1: FIXING CSV FORMAT")
    print("="*50)
    
    fixed_file = fix_csv_format(input_file)
    if not fixed_file:
        print("❌ CSV format fix failed")
        return
    
    # Step 2: Analyze timezone
    print(f"\n" + "="*50)
    print("STEP 2: TIMEZONE ANALYSIS")
    print("="*50)
    
    detected_tz, confidence = analyze_csv_timezone(fixed_file)
    
    # Step 3: Timezone conversion
    print(f"\n" + "="*50)
    print("STEP 3: TIMEZONE CONVERSION")
    print("="*50)
    
    print(f"🌍 Your system timezone: GMT-7 (US Mountain Time)")
    if detected_tz:
        print(f"🏦 Detected broker timezone: {detected_tz}")
        
        if "GMT-7" in detected_tz:
            print(f"✅ Timezones already match! No conversion needed.")
            final_file = fixed_file
        else:
            print(f"⚠️  TIMEZONE MISMATCH DETECTED!")
            print(f"   Your data: {detected_tz}")
            print(f"   Your system: GMT-7")
            print(f"   This explains why predictions are 'way off'!")
            
            response = input(f"\nConvert {detected_tz} to GMT-7? (y/n): ").lower()
            
            if response == 'y':
                # Extract GMT offset from detected timezone
                if "GMT+0" in detected_tz:
                    source_offset = 0
                elif "GMT+1" in detected_tz:
                    source_offset = 1
                elif "GMT+2" in detected_tz:
                    source_offset = 2
                elif "GMT+3" in detected_tz:
                    source_offset = 3
                elif "GMT-5" in detected_tz:
                    source_offset = -5
                else:
                    source_offset = 2  # Default to GMT+2 (common broker timezone)
                
                final_file = convert_timezone(fixed_file, source_offset, -7)
                
                if final_file:
                    print(f"\n📊 Re-analyzing converted data...")
                    analyze_csv_timezone(final_file)
            else:
                final_file = fixed_file
    else:
        print(f"❓ Could not detect timezone automatically")
        final_file = fixed_file
    
    # Step 4: Replace original file
    print(f"\n" + "="*60)
    print("🎉 PROCESSING COMPLETE!")
    print("="*60)
    
    print(f"📁 Files created:")
    print(f"   💾 Backup: {backup_file}")
    print(f"   🔧 Fixed: {fixed_file}")
    if final_file != fixed_file:
        print(f"   🌍 Final: {final_file}")
    
    replace_response = input(f"\nReplace original {input_file} with corrected version? (y/n): ").lower()
    
    if replace_response == 'y':
        try:
            shutil.copy2(final_file, input_file)
            print(f"✅ Successfully replaced {input_file}")
            print(f"\n🚀 NEXT STEPS:")
            print(f"1. Retrain your model: python train_enhanced_model.py")
            print(f"2. Test predictions: python daemon.py") 
            print(f"3. Monitor accuracy - should be MUCH better!")
        except Exception as e:
            print(f"❌ Failed to replace: {e}")
    else:
        print(f"\n📋 Manual replacement:")
        print(f"   cp '{final_file}' '{input_file}'")

if __name__ == "__main__":
    main()