#!/usr/bin/env python3
"""
LSTM Trading Time Synchronization Checker
==========================================
This script analyzes time alignment between training data, system time, and broker time
to identify potential causes of inaccurate LSTM predictions.

Author: Enhanced LSTM Trading System
Version: 1.0
"""

import os
import pandas as pd
from datetime import datetime, timezone
import sys

def check_time_alignment():
    """Check time alignment between training data, server, and EA"""
    
    print("🕐 LSTM Time Synchronization Checker")
    print("=" * 60)
    
    # 1. Check current system time
    system_time = datetime.now()
    utc_time = datetime.utcnow()
    timezone_offset = system_time - utc_time
    
    print(f"🖥️  System Time (Local): {system_time.strftime('%Y.%m.%d %H:%M:%S')}")
    print(f"🌍 System Time (UTC): {utc_time.strftime('%Y.%m.%d %H:%M:%S')}")
    print(f"⏰ Timezone Offset: {timezone_offset}")
    
    # 2. Check training data timestamps
    print(f"\n📊 TRAINING DATA ANALYSIS")
    print("-" * 40)
    
    try:
        # Look for training data files
        data_files = ["EURUSD60.csv", "EURUSD.csv", "eurusd_h1.csv"]
        data_file = None
        
        for file in data_files:
            if os.path.exists(file):
                data_file = file
                break
        
        if data_file:
            print(f"📁 Found training data: {data_file}")
            df = pd.read_csv(data_file)
            
            # Try different date column names
            date_columns = ['Date', 'Time', 'Datetime', 'timestamp']
            date_col = None
            
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                latest_data = df[date_col].max()
                earliest_data = df[date_col].min()
                
                print(f"📅 Date Column: {date_col}")
                print(f"📅 Earliest Record: {earliest_data}")
                print(f"📅 Latest Record: {latest_data}")
                print(f"📈 Total Records: {len(df):,}")
                
                # Check if data is recent
                if isinstance(latest_data, pd.Timestamp):
                    latest_data = latest_data.to_pydatetime()
                
                time_since_latest = system_time - latest_data
                print(f"⏱️  Time since latest data: {time_since_latest}")
                
                if time_since_latest.days > 7:
                    print(f"   ⚠️  WARNING: Training data is {time_since_latest.days} days old!")
                elif time_since_latest.days > 1:
                    print(f"   ℹ️  Data is {time_since_latest.days} days old (acceptable)")
                else:
                    print(f"   ✅ Data is recent")
                
                # Analyze timezone pattern from data
                print(f"\n🔍 SAMPLE TIMESTAMPS (First 10 records):")
                sample_times = df[date_col].head(10)
                for i, ts in enumerate(sample_times):
                    if pd.isna(ts):
                        continue
                    hour = ts.hour
                    day_of_week = ts.strftime('%A')
                    print(f"   {i+1:2d}. {ts} ({day_of_week}, Hour: {hour:2d})")
                
                # Analyze market hours pattern
                print(f"\n📈 TRADING ACTIVITY BY HOUR:")
                hour_counts = df[date_col].dt.hour.value_counts().sort_index()
                
                # Show hours with most activity
                top_hours = hour_counts.head(12)
                for hour, count in top_hours.items():
                    percentage = (count / len(df)) * 100
                    bar_length = int(percentage / 2)  # Scale bar
                    bar = "█" * bar_length
                    print(f"   Hour {hour:2d}: {count:6,} records ({percentage:4.1f}%) {bar}")
                
                # Detect likely timezone
                print(f"\n🌍 TIMEZONE DETECTION:")
                peak_hours = hour_counts.nlargest(6).index.tolist()
                peak_hours.sort()
                
                if 8 in peak_hours and 9 in peak_hours and 14 in peak_hours:
                    print("   🎯 Likely timezone: GMT+0 (London Winter) or UTC")
                elif 9 in peak_hours and 10 in peak_hours and 15 in peak_hours:
                    print("   🎯 Likely timezone: GMT+1 (London Summer)")
                elif 10 in peak_hours and 11 in peak_hours and 16 in peak_hours:
                    print("   🎯 Likely timezone: GMT+2 (Cyprus Winter)")
                elif 11 in peak_hours and 12 in peak_hours and 17 in peak_hours:
                    print("   🎯 Likely timezone: GMT+3 (Cyprus Summer)")
                elif 3 in peak_hours and 4 in peak_hours and 9 in peak_hours:
                    print("   🎯 Likely timezone: GMT-5 (New York Winter)")
                elif 4 in peak_hours and 5 in peak_hours and 10 in peak_hours:
                    print("   🎯 Likely timezone: GMT-4 (New York Summer)")
                else:
                    print("   ❓ Timezone pattern unclear")
                    print(f"     Peak activity hours: {peak_hours}")
                
            else:
                print(f"⚠️  No recognized date column found")
                print(f"   Available columns: {list(df.columns)}")
        else:
            print(f"⚠️  No training data files found")
            print(f"   Searched for: {', '.join(data_files)}")
            print(f"   Current directory: {os.getcwd()}")
            
    except Exception as e:
        print(f"💥 Error reading training data: {e}")
    
    # 3. Current market status
    print(f"\n📊 CURRENT MARKET ANALYSIS")
    print("-" * 40)
    
    current_utc = datetime.utcnow()
    current_hour_utc = current_utc.hour
    current_day = current_utc.weekday()  # 0=Monday, 6=Sunday
    
    # Forex market sessions (UTC)
    market_sessions = {
        'Sydney': (21, 6),    # 9 PM UTC - 6 AM UTC
        'Tokyo': (0, 9),      # 12 AM UTC - 9 AM UTC  
        'London': (8, 17),    # 8 AM UTC - 5 PM UTC
        'New York': (13, 22), # 1 PM UTC - 10 PM UTC
    }
    
    print(f"🌍 Current UTC Time: {current_utc.strftime('%Y.%m.%d %H:%M:%S')}")
    print(f"📅 Current Day: {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][current_day]}")
    print(f"🕐 Current UTC Hour: {current_hour_utc}")
    
    active_sessions = []
    for session, (start, end) in market_sessions.items():
        if start <= end:  # Normal range
            if start <= current_hour_utc <= end:
                active_sessions.append(session)
        else:  # Overnight range (like Sydney)
            if current_hour_utc >= start or current_hour_utc <= end:
                active_sessions.append(session)
    
    if current_day < 5:  # Monday to Friday
        if active_sessions:
            print(f"📈 Active Sessions: {', '.join(active_sessions)}")
            print(f"   ✅ Market is OPEN")
        else:
            print(f"📈 Active Sessions: None")
            print(f"   ⚠️  Market is CLOSED")
    else:
        print(f"📈 Weekend: Market CLOSED")
    
    # 4. Common broker timezones
    print(f"\n🏦 COMMON BROKER TIMEZONES")
    print("-" * 40)
    
    broker_timezones = {
        'GMT+0': 'London Winter, UTC',
        'GMT+1': 'London Summer, CET',
        'GMT+2': 'Cyprus Winter, EET', 
        'GMT+3': 'Cyprus Summer, Moscow',
        'GMT-5': 'New York Winter, EST',
        'GMT-4': 'New York Summer, EDT',
    }
    
    for tz_name, description in broker_timezones.items():
        print(f"   {tz_name:6s}: {description}")
    
    # 5. Recommendations
    print(f"\n💡 TROUBLESHOOTING RECOMMENDATIONS")
    print("-" * 40)
    
    print("📋 To fix time synchronization issues:")
    print()
    print("1. 🔍 CHECK METATRADER SERVER TIME:")
    print("   • Open MetaTrader → Market Watch")
    print("   • Right-click → Symbols → Select your pair")
    print("   • Note the server time shown")
    print("   • Check if it shows GMT+X offset")
    print()
    print("2. 📊 VERIFY TRAINING DATA TIMEZONE:")
    print("   • Re-export H1 data with correct timezone")
    print("   • Ensure training data matches broker timezone")
    print("   • Consider using UTC for consistency")
    print()
    print("3. 🔧 DAEMON ADJUSTMENTS:")
    print("   • Set system timezone to match broker")
    print("   • Or add timezone conversion in daemon")
    print("   • Verify feature calculations use same timeframe")
    print()
    print("4. 🧪 TEST PREDICTION ACCURACY:")
    print("   • Note current price and prediction")
    print("   • Wait 1 hour and compare actual vs predicted")
    print("   • Good predictions should be within 0.1-0.3%")
    
    # 6. Generate test configuration
    print(f"\n🧪 TESTING CONFIGURATION")
    print("-" * 40)
    
    test_timestamp = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    print(f"Current Local Time: {test_timestamp}")
    print(f"Current UTC Time: {utc_time.strftime('%Y.%m.%d %H:%M:%S')}")
    print()
    print("Add this to your EA OnInit() for testing:")
    print('Print("=== TIME SYNC TEST ===");')
    print('Print("EA Time: ", TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES|TIME_SECONDS));')
    print('Print("Server Time: ", TimeToString(TimeTradeServer(), TIME_DATE|TIME_MINUTES|TIME_SECONDS));')
    print('Print("Local Time: ", TimeToString(TimeLocal(), TIME_DATE|TIME_MINUTES|TIME_SECONDS));')
    print('Print("Current Bar: ", TimeToString(iTime(Symbol(), PERIOD_CURRENT, 0), TIME_DATE|TIME_MINUTES));')
    
    return {
        'system_time': system_time,
        'utc_time': utc_time,
        'active_sessions': active_sessions,
        'market_open': len(active_sessions) > 0 and current_day < 5,
        'timezone_offset': timezone_offset
    }

def check_daemon_config():
    """Check daemon configuration files"""
    print(f"\n🔧 DAEMON CONFIGURATION CHECK")
    print("-" * 40)
    
    # Check if daemon files exist
    daemon_files = ["daemon.py", "enhanced_daemon.py"]
    for file in daemon_files:
        if os.path.exists(file):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
    
    # Check model files
    model_dir = "models"
    if os.path.exists(model_dir):
        print(f"✅ Models directory exists")
        model_files = os.listdir(model_dir)
        for file in model_files:
            file_size = os.path.getsize(os.path.join(model_dir, file))
            print(f"   📁 {file} ({file_size:,} bytes)")
    else:
        print(f"❌ Models directory missing")

def main():
    """Main function"""
    print("🚀 LSTM Trading Time Synchronization Checker v1.0")
    print("🎯 Diagnosing time alignment issues for accurate predictions")
    print()
    
    # Run checks
    result = check_time_alignment()
    check_daemon_config()
    
    print(f"\n" + "=" * 60)
    print("🎉 TIME SYNC CHECK COMPLETE!")
    print("=" * 60)
    
    # Summary
    if result['market_open']:
        print("✅ Market is currently OPEN")
    else:
        print("⚠️  Market is currently CLOSED")
    
    print(f"⏰ Your timezone offset: {result['timezone_offset']}")
    
    print(f"\n📋 NEXT STEPS:")
    print("1. Compare times shown above with your MetaTrader")
    print("2. Run enhanced daemon.py with time logging enabled")
    print("3. Test EA with EnableDetailedLogging = true")
    print("4. Monitor prediction accuracy over several hours")
    print("5. Retrain model if timezone mismatch is found")

if __name__ == "__main__":
    main()