//+------------------------------------------------------------------+
//|                                                          GGTH.mq5 |
//|                                         Copyright 2025, Jason Rusk |
//|  Dedicated to my late wife who always supported my trading passion |
//|                                                                    |
//+------------------------------------------------------------------+

#property copyright "Jason.W.Rusk@gmail.com 2025"
#property version   "1.86" // Fixed Multi-Step Accuracy Tracking

#include <Trade/Trade.mqh>
#include <Files/File.mqh>
#include <stdlib.mqh>
#include <Math/Stat/Math.mqh>

// --- TYPE DEFINITIONS ---
enum ENUM_TRADING_MODE { MODE_TRADING_DISABLED, MODE_REGRESSION_ONLY, MODE_COMBINED };
enum ENUM_STOP_LOSS_MODE { SL_ATR_BASED, SL_STATIC_PIPS };
enum ENUM_TAKE_PROFIT_MODE { TP_REGRESSION_TARGET, TP_ATR_MULTIPLE, TP_STATIC_PIPS };
enum ENUM_TARGET_BAR { H_PLUS_1=0, H_PLUS_2, H_PLUS_3, H_PLUS_4, H_PLUS_5 };

// --- INPUT PARAMETERS ---
input group    "Main Settings"
input ENUM_TRADING_MODE TradingLogicMode = MODE_COMBINED;
input bool              EnablePricePredictionDisplay = true;
input ENUM_TARGET_BAR   TakeProfitTargetBar = H_PLUS_5;

input group    "Risk & Position Management"
input ENUM_STOP_LOSS_MODE   StopLossMode = SL_ATR_BASED;
input ENUM_TAKE_PROFIT_MODE TakeProfitMode = TP_REGRESSION_TARGET;
input bool   UseMarketOrderForTP = false;
input double RiskPercent = 3.0;
input double MinimumRiskRewardRatio = 1.5;
input int    StaticStopLossPips = 300;
input int    StaticTakeProfitPips = 400;
input int    ATR_Period = 14;
input double ATR_SL_Multiplier = 1.5;
input double ATR_TP_Multiplier = 2.0;
input double MinProfitPips = 10.0;
input bool   EnableTimeBasedExit = true;
input int    MaxPositionHoldBars = 12;
input int    InpExitBarMinute = 58;
input bool   EnableTrailingStop = true;
input double TrailingStartPips = 12.0;
input double TrailingStopPips = 3.0;

input group    "Confidence & Filters"
input double MinimumModelConfidence = 0.40;
input double MinimumSignalConfidence = 0.80;
input double ClassificationSignalThreshold = 0.60;
input int    RequiredConsistentSteps = 4;
input bool   EnableADXFilter = true;
input int    ADX_Period = 14;
input int    ADX_Threshold = 25;

input group    "Model & Data Settings"
input int    AccuracyLookaheadBars = 5;
input int    AccuracyLookbackOnInit = 60;
input string Symbol_EURJPY = "EURJPY", Symbol_USDJPY = "USDJPY", Symbol_GBPUSD = "GBPUSD";
input string Symbol_EURGBP = "EURGBP", Symbol_USDCAD = "USDCAD", Symbol_USDCHF = "USDCHF";
input int    RequestTimeout = 5000;

// --- Constants ---
#define PREDICTION_STEPS 5
#define SEQ_LEN 20
#define FEATURE_COUNT 15
#define DATA_FOLDER "LSTM_Trading\\data"
#define GUI_PREFIX "GGTHGUI_"
#define BACKTEST_PREDICTIONS_FILE "backtest_predictions.csv"

// --- Global Handles & Variables ---
int atr_handle, macd_handle, rsi_handle, stoch_handle, cci_handle, adx_handle, bb_handle;
CTrade trade;
enum ENUM_PREDICTION_DIRECTION { DIR_BULLISH, DIR_BEARISH, DIR_NEUTRAL };

// --- IMPROVED PREDICTION STRUCTURE ---
struct StepPrediction 
{
    double target_price;
    datetime prediction_bar_time;   // When the prediction was made
    datetime target_bar_time;       // When this step should be evaluated
    ENUM_PREDICTION_DIRECTION direction;
    int step;                      // 0=1-bar ahead, 1=2-bar ahead, etc.
    bool evaluated;                // Whether this prediction has been checked
};

StepPrediction g_step_predictions[];
double g_last_predictions[PREDICTION_STEPS];
double g_last_confidence_score = 0.0;
double g_accuracy_pct[PREDICTION_STEPS];
int    g_total_hits[PREDICTION_STEPS], g_total_predictions[PREDICTION_STEPS];
double g_active_trade_target_price = 0;

struct BacktestPrediction { datetime timestamp; double buy_prob, sell_prob, hold_prob, confidence_score; double predicted_prices[PREDICTION_STEPS]; };
BacktestPrediction g_backtest_predictions[];
int g_backtest_prediction_idx = 0;

struct DaemonResponse { double prices[PREDICTION_STEPS]; double confidence_score; double buy_prob; double sell_prob; };

double g_RiskPercent, g_MinimumRiskRewardRatio, g_ATR_SL_Multiplier, g_ATR_TP_Multiplier, g_MinProfitPips, g_TrailingStartPips, g_TrailingStopPips, g_MinimumModelConfidence, g_MinimumSignalConfidence, g_ClassificationSignalThreshold;
int    g_RequiredConsistentSteps, g_StaticStopLossPips, g_StaticTakeProfitPips, g_ATR_Period, g_MaxPositionHoldBars, g_ExitBarMinute, g_ADX_Period, g_ADX_Threshold, g_AccuracyLookbackOnInit;
bool   g_EnableTimeBasedExit, g_EnableTrailingStop, g_EnableADXFilter;

//+------------------------------------------------------------------+
//| --- GUI PANEL FUNCTIONS ---
//+------------------------------------------------------------------+
void CreateDisplayPanel()
{
   if(!EnablePricePredictionDisplay) return;
   string bg_name = GUI_PREFIX + "background";
   ObjectCreate(0, bg_name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, bg_name, OBJPROP_XDISTANCE, 5);
   ObjectSetInteger(0, bg_name, OBJPROP_YDISTANCE, 20);
   ObjectSetInteger(0, bg_name, OBJPROP_XSIZE, 500); // Much wider panel for horizontal spacing
   ObjectSetInteger(0, bg_name, OBJPROP_YSIZE, 80 + (PREDICTION_STEPS * 80) + 30);
   ObjectSetInteger(0, bg_name, OBJPROP_BGCOLOR, C'20,20,40');
   ObjectSetInteger(0, bg_name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, bg_name, OBJPROP_BACK, true);
   string title_name = GUI_PREFIX + "title";
   ObjectCreate(0, title_name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, title_name, OBJPROP_XDISTANCE, 20);
   ObjectSetInteger(0, title_name, OBJPROP_YDISTANCE, 40);
   ObjectSetString(0, title_name, OBJPROP_TEXT, "GGTH LSTM Prediction (" + _Symbol + " H1)");
   ObjectSetInteger(0, title_name, OBJPROP_COLOR, clrWhite);
   ObjectSetInteger(0, title_name, OBJPROP_FONTSIZE, 9);
   int y_pos = 90;
   for(int i = 0; i < PREDICTION_STEPS; i++)
   {
      string hour_label_name = GUI_PREFIX + "hour_" + (string)i;
      ObjectCreate(0, hour_label_name, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, hour_label_name, OBJPROP_XDISTANCE, 20); // More left margin
      ObjectSetInteger(0, hour_label_name, OBJPROP_YDISTANCE, y_pos);
      ObjectSetString(0, hour_label_name, OBJPROP_TEXT, StringFormat("H+%d:", i + 1));
      ObjectSetInteger(0, hour_label_name, OBJPROP_COLOR, clrSilver);
      ObjectSetInteger(0, hour_label_name, OBJPROP_FONTSIZE, 8);
      
      string price_label_name = GUI_PREFIX + "price_" + (string)i;
      ObjectCreate(0, price_label_name, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, price_label_name, OBJPROP_XDISTANCE, 120); // Much more spacing from hour label
      ObjectSetInteger(0, price_label_name, OBJPROP_YDISTANCE, y_pos);
      ObjectSetString(0, price_label_name, OBJPROP_TEXT, "Initializing...");
      ObjectSetInteger(0, price_label_name, OBJPROP_COLOR, clrWhite);
      ObjectSetInteger(0, price_label_name, OBJPROP_FONTSIZE, 8);
      
      string acc_label_name = GUI_PREFIX + "acc_" + (string)i;
      ObjectCreate(0, acc_label_name, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, acc_label_name, OBJPROP_XDISTANCE, 300); // Much more spacing from price
      ObjectSetInteger(0, acc_label_name, OBJPROP_YDISTANCE, y_pos);
      ObjectSetString(0, acc_label_name, OBJPROP_TEXT, "Acc: Init...");
      ObjectSetInteger(0, acc_label_name, OBJPROP_COLOR, clrGray);
      ObjectSetInteger(0, acc_label_name, OBJPROP_FONTSIZE, 8);
      y_pos += 80;
   }
   ChartRedraw();
}

void UpdateDisplayPanel()
{
    if(!EnablePricePredictionDisplay) return;
    
    string title_name = GUI_PREFIX + "title";
    string title_text = StringFormat("GGTH LSTM (%s H1) | Confidence: %.2f", _Symbol, g_last_confidence_score);
    ObjectSetString(0, title_name, OBJPROP_TEXT, title_text);
    
    static int debug_updates = 0;
    
    for(int i = 0; i < PREDICTION_STEPS; i++)
    {
        string price_label_name = GUI_PREFIX + "price_" + (string)i;
        string price_text = (g_last_predictions[i] == 0) ? "Calculating..." : 
                           DoubleToString(g_last_predictions[i], _Digits);
        ObjectSetString(0, price_label_name, OBJPROP_TEXT, price_text);
        
        string acc_label_name = GUI_PREFIX + "acc_" + (string)i;
        string acc_text = "Acc: N/A";
        
        if(g_total_predictions[i] > 0)
        {
            g_accuracy_pct[i] = ((double)g_total_hits[i] / (double)g_total_predictions[i]) * 100.0;
            acc_text = StringFormat("%.1f%% (%d/%d)", 
                                   g_accuracy_pct[i], 
                                   g_total_hits[i], 
                                   g_total_predictions[i]);
        }
        else
        {
            acc_text = "Acc: 0 preds";
        }
        
        ObjectSetString(0, acc_label_name, OBJPROP_TEXT, acc_text);
        
        // Color-code accuracy
        color acc_color = clrGray;
        if(g_total_predictions[i] > 5)
        {
            if(g_accuracy_pct[i] >= 60.0) acc_color = clrLimeGreen;
            else if(g_accuracy_pct[i] >= 45.0) acc_color = clrYellow;
            else acc_color = clrTomato;
        }
        ObjectSetInteger(0, acc_label_name, OBJPROP_COLOR, acc_color);
        
        if(debug_updates < 5 || (g_total_predictions[i] > 0 && (g_total_predictions[i] % 10 == 0)))
        {
            PrintFormat("Display Step %d: Hits=%d, Total=%d, Accuracy=%.1f%%, Text='%s'", 
                       i+1, g_total_hits[i], g_total_predictions[i], g_accuracy_pct[i], acc_text);
        }
    }
    
    if(debug_updates < 5)
    {
        PrintFormat("UpdateDisplayPanel #%d: Tester=%s, PredArray=%d, DisplayEnabled=%s", 
                   debug_updates + 1, 
                   MQLInfoInteger(MQL_TESTER) ? "YES" : "NO",
                   ArraySize(g_step_predictions),
                   EnablePricePredictionDisplay ? "YES" : "NO");
        debug_updates++;
    }
    
    ChartRedraw();
}

void DeleteDisplayPanel() 
{ 
   ObjectsDeleteAll(0, GUI_PREFIX); 
   ChartRedraw(); 
}

//+------------------------------------------------------------------+
//| --- IMPROVED ACCURACY TRACKING FUNCTIONS ---
//+------------------------------------------------------------------+
void AddStepPredictions(const double &predicted_prices[], datetime current_bar_time)
{
    if(!EnablePricePredictionDisplay) return;
    
    double prediction_bar_close = iClose(_Symbol, PERIOD_H1, 1);
    if(prediction_bar_close <= 0) return;
    
    static int predictions_added = 0;
    
    for(int step = 0; step < PREDICTION_STEPS; step++)
    {
        datetime target_time = current_bar_time + ((step + 1) * PeriodSeconds(PERIOD_H1));
        
        ENUM_PREDICTION_DIRECTION step_direction = DIR_NEUTRAL;
        double price_diff = predicted_prices[step] - prediction_bar_close;
        double threshold = 0.0001;
        
        if(price_diff > threshold)
            step_direction = DIR_BULLISH;
        else if(price_diff < -threshold)
            step_direction = DIR_BEARISH;
        
        StepPrediction pred;
        pred.target_price = predicted_prices[step];
        pred.prediction_bar_time = current_bar_time;
        pred.target_bar_time = target_time;
        pred.direction = step_direction;
        pred.step = step;
        pred.evaluated = false;
        
        int size = ArraySize(g_step_predictions);
        ArrayResize(g_step_predictions, size + 1);
        g_step_predictions[size] = pred;
        
        predictions_added++;
        if(predictions_added <= 10)
        {
            PrintFormat("Prediction #%d - Step %d: Base=%.5f, Target=%.5f, Diff=%.5f, Dir=%s", 
                       predictions_added, step+1, prediction_bar_close, predicted_prices[step], 
                       price_diff, step_direction == DIR_BULLISH ? "BULL" : step_direction == DIR_BEARISH ? "BEAR" : "NEUTRAL");
        }
    }
}

void CheckStepPredictionAccuracy()
{
    if(!EnablePricePredictionDisplay || ArraySize(g_step_predictions) == 0) return;
    
    datetime current_bar_time = iTime(_Symbol, PERIOD_H1, 0);
    static int evaluations_done = 0;
    
    for(int i = ArraySize(g_step_predictions) - 1; i >= 0; i--)
    {
        if(g_step_predictions[i].evaluated || current_bar_time < g_step_predictions[i].target_bar_time) 
            continue;
            
        int target_bar_index = iBarShift(_Symbol, PERIOD_H1, g_step_predictions[i].target_bar_time);
        if(target_bar_index < 0) continue;
        
        double target_open = iOpen(_Symbol, PERIOD_H1, target_bar_index);
        double target_high = iHigh(_Symbol, PERIOD_H1, target_bar_index);
        double target_low = iLow(_Symbol, PERIOD_H1, target_bar_index);
        double target_close = iClose(_Symbol, PERIOD_H1, target_bar_index);
        
        int prediction_bar_index = iBarShift(_Symbol, PERIOD_H1, g_step_predictions[i].prediction_bar_time);
        double base_price = iClose(_Symbol, PERIOD_H1, prediction_bar_index + 1);
        
        bool is_hit = false;
        string hit_reason = "";
        
        if(g_step_predictions[i].direction == DIR_BULLISH)
        {
            if(g_step_predictions[i].target_price <= target_high)
            {
                is_hit = true;
                hit_reason = "Target reached in high";
            }
            else if(target_close > base_price)
            {
                is_hit = true;
                hit_reason = "Close higher than base";
            }
        }
        else if(g_step_predictions[i].direction == DIR_BEARISH)
        {
            if(g_step_predictions[i].target_price >= target_low)
            {
                is_hit = true;
                hit_reason = "Target reached in low";
            }
            else if(target_close < base_price)
            {
                is_hit = true;
                hit_reason = "Close lower than base";
            }
        }
        else
        {
            double price_change = MathAbs(target_close - base_price);
            double threshold = 0.0005;
            if(price_change <= threshold)
            {
                is_hit = true;
                hit_reason = "Stayed neutral";
            }
        }
        
        g_total_predictions[g_step_predictions[i].step]++;
        if(is_hit) g_total_hits[g_step_predictions[i].step]++;
        
        g_step_predictions[i].evaluated = true;
        evaluations_done++;
        
        if(evaluations_done <= 15)
        {
            PrintFormat("Eval #%d - Step %d (%s): %s | Base=%.5f Target=%.5f | Bar: O=%.5f H=%.5f L=%.5f C=%.5f | %s", 
                       evaluations_done,
                       g_step_predictions[i].step + 1,
                       g_step_predictions[i].direction == DIR_BULLISH ? "BULL" : g_step_predictions[i].direction == DIR_BEARISH ? "BEAR" : "NEUT",
                       is_hit ? "HIT" : "MISS",
                       base_price,
                       g_step_predictions[i].target_price,
                       target_open, target_high, target_low, target_close,
                       hit_reason);
        }
    }
    
    if(ArraySize(g_step_predictions) > 100)
    {
        int evaluated_count = 0;
        for(int i = 0; i < ArraySize(g_step_predictions); i++)
        {
            if(g_step_predictions[i].evaluated) evaluated_count++;
        }
        
        if(evaluated_count > 80)
        {
            datetime current_bar_time_local = iTime(_Symbol, PERIOD_H1, 0);
            for(int i = ArraySize(g_step_predictions) - 1; i >= 0; i--)
            {
                if(g_step_predictions[i].evaluated && 
                   current_bar_time_local - g_step_predictions[i].target_bar_time > 24 * PeriodSeconds(PERIOD_H1))
                {
                    ArrayRemove(g_step_predictions, i, 1);
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| --- CORE HELPER FUNCTIONS ---
//+------------------------------------------------------------------+
void InitializeParameters()
{
   g_RiskPercent = RiskPercent; 
   g_MinimumRiskRewardRatio = MinimumRiskRewardRatio;
   g_RequiredConsistentSteps = RequiredConsistentSteps; 
   g_StaticStopLossPips = StaticStopLossPips;
   g_StaticTakeProfitPips = StaticTakeProfitPips; 
   g_ATR_Period = ATR_Period;
   g_ATR_SL_Multiplier = ATR_SL_Multiplier; 
   g_ATR_TP_Multiplier = ATR_TP_Multiplier;
   g_MinProfitPips = MinProfitPips; 
   g_EnableTimeBasedExit = EnableTimeBasedExit;
   g_MaxPositionHoldBars = MaxPositionHoldBars; 
   g_ExitBarMinute = InpExitBarMinute;
   g_EnableTrailingStop = EnableTrailingStop; 
   g_TrailingStartPips = TrailingStartPips;
   g_TrailingStopPips = TrailingStopPips; 
   g_EnableADXFilter = EnableADXFilter;
   g_ADX_Period = ADX_Period; 
   g_ADX_Threshold = ADX_Threshold;
   g_MinimumModelConfidence = MinimumModelConfidence;
   g_MinimumSignalConfidence = MinimumSignalConfidence;
   g_ClassificationSignalThreshold = ClassificationSignalThreshold;
   g_AccuracyLookbackOnInit = AccuracyLookbackOnInit;
}

bool LoadBacktestPredictions()
{
   ArrayFree(g_backtest_predictions); 
   g_backtest_prediction_idx = 0;
   
   if(!FileIsExist(BACKTEST_PREDICTIONS_FILE, FILE_COMMON)) 
   { 
      PrintFormat("FATAL: Backtest file not found in MQL5\\Common\\Files folder: %s", BACKTEST_PREDICTIONS_FILE); 
      return false; 
   }
     
   int file_handle = FileOpen(BACKTEST_PREDICTIONS_FILE, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON); 
   if(file_handle == INVALID_HANDLE) 
   { 
      PrintFormat("FATAL: Could not open backtest file. Code: %d", GetLastError()); 
      return false; 
   }
     
   if(!FileIsEnding(file_handle)) FileReadString(file_handle);

   int count = 0;
   int expected_columns = 5 + PREDICTION_STEPS;

   while(!FileIsEnding(file_handle))
   {
      string line = FileReadString(file_handle);
      string fields[];
      int splits = StringSplit(line, ';', fields);
      
      if(splits != expected_columns)
      {
         if(StringLen(line) > 5) Print("Warning: Skipping corrupted or incomplete line in backtest file. Found ", splits, " of ", expected_columns, " columns.");
         continue;
      }
      
      datetime ts = StringToTime(fields[0]);
      if(ts == 0) continue;
      
      ArrayResize(g_backtest_predictions, count + 1);
      g_backtest_predictions[count].timestamp = ts;
      g_backtest_predictions[count].buy_prob = StringToDouble(fields[1]);
      g_backtest_predictions[count].sell_prob = StringToDouble(fields[2]);
      g_backtest_predictions[count].hold_prob = StringToDouble(fields[3]);
      g_backtest_predictions[count].confidence_score = StringToDouble(fields[4]);
      
      for(int i=0; i<PREDICTION_STEPS; i++) 
      { 
         g_backtest_predictions[count].predicted_prices[i] = StringToDouble(fields[5+i]);
      }
      count++;
   }
   FileClose(file_handle);
   PrintFormat("Loaded %d valid pre-computed records.", count);
   return(count > 0);
}

bool FindPredictionForBar(datetime bar_time, BacktestPrediction &found_pred, bool reset_search_index=false)
{
   if(reset_search_index)
   {
      g_backtest_prediction_idx = 0;
   }
   for(int i = g_backtest_prediction_idx; i < ArraySize(g_backtest_predictions); i++)
   {
      if(g_backtest_predictions[i].timestamp == bar_time) 
      { 
         found_pred = g_backtest_predictions[i]; 
         g_backtest_prediction_idx = i;
         return true; 
      }
      if(g_backtest_predictions[i].timestamp > bar_time) 
      {
         return false;
      }
   }
   return false;
}

string GenerateRequestID() 
{ 
   MathSrand((int)GetTickCount()); 
   string id = (string)TimeLocal() + "_" + IntegerToString(MathRand()); 
   StringReplace(id, ":", "-"); 
   StringReplace(id, " ", "_"); 
   return id; 
}

bool JsonGetValue(const string &json_string, const string &key, double &out_value)
{
   string search_key = "\"" + key + "\"";
   int key_pos = StringFind(json_string, search_key); 
   if(key_pos < 0) return false;
   int colon_pos = StringFind(json_string, ":", key_pos); 
   if(colon_pos < 0) return false;
   int next_comma_pos = StringFind(json_string, ",", colon_pos);
   int next_brace_pos = StringFind(json_string, "}", colon_pos);
   int end_pos = (next_comma_pos > 0 && (next_brace_pos < 0 || next_comma_pos < next_brace_pos)) ? next_comma_pos : next_brace_pos;
   if(end_pos < 0) end_pos = StringLen(json_string);
   string value_str = StringSubstr(json_string, colon_pos + 1, end_pos - (colon_pos + 1));
   StringTrimLeft(value_str); 
   StringTrimRight(value_str);
   out_value = StringToDouble(value_str);
   return true;
}

bool SendToDaemon(const double &features[], double current_price, double atr_val, DaemonResponse &response)
{
   string request_id = GenerateRequestID();
   string filename = "request_" + request_id + ".json";
   string response_file = "response_" + request_id + ".json";
   string json = StringFormat("{\r\n  \"request_id\": \"%s\",\r\n  \"action\": \"predict_combined\",\r\n  \"current_price\": %.5f,\r\n  \"atr\": %.5f,\r\n  \"features\": [",
                              request_id, current_price, atr_val);
   for(int i = 0; i < ArraySize(features); i++) 
      json += DoubleToString(features[i], 8) + (i < ArraySize(features) - 1 ? ", " : "");
   json += "]\r\n}";
   
   int file_handle = FileOpen(DATA_FOLDER + "\\" + filename, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(file_handle == INVALID_HANDLE) 
   { 
      PrintFormat("Error writing request. Code: %d", GetLastError()); 
      return false; 
   }
   FileWriteString(file_handle, json); 
   FileClose(file_handle);
   
   long start_time = GetTickCount();
   while(GetTickCount() - start_time < RequestTimeout)
   {
      Sleep(100);
      if(FileIsExist(DATA_FOLDER + "\\" + response_file))
      {
         Sleep(50);
         int rfile = FileOpen(DATA_FOLDER + "\\" + response_file, FILE_READ | FILE_TXT | FILE_ANSI);
         if(rfile == INVALID_HANDLE) continue;
         string content = FileReadString(rfile);
         FileClose(rfile); 
         FileDelete(DATA_FOLDER + "\\" + response_file);
         
         double temp_prices[PREDICTION_STEPS];
         double temp_confidence, temp_buy, temp_sell;
         
         int prices_pos = StringFind(content, "\"predicted_prices\""); 
         if(prices_pos < 0) continue;
         int start_bracket = StringFind(content, "[", prices_pos); 
         int end_bracket = StringFind(content, "]", start_bracket);
         if(start_bracket < 0 || end_bracket < 0) continue;
         string prices_str = StringSubstr(content, start_bracket + 1, end_bracket - start_bracket - 1);
         string price_values[];
         if(StringSplit(prices_str, ',', price_values) == PREDICTION_STEPS)
         {
            for(int i = 0; i < PREDICTION_STEPS; i++) 
            { 
               StringTrimLeft(price_values[i]); 
               StringTrimRight(price_values[i]); 
               temp_prices[i] = StringToDouble(price_values[i]); 
            }
            if(JsonGetValue(content, "confidence_score", temp_confidence) && 
               JsonGetValue(content, "buy_probability", temp_buy) &&
               JsonGetValue(content, "sell_probability", temp_sell))
            {
               ArrayCopy(response.prices, temp_prices, 0, 0, PREDICTION_STEPS);
               response.confidence_score = temp_confidence;
               response.buy_prob = temp_buy;
               response.sell_prob = temp_sell;
               return true;
            }
         }
      }
   }
   return false;
}

double CalculateLotSize(double stopLossPrice, double entryPrice)
{
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE); 
   if(accountBalance <= 0) return 0.0;
   double riskAmount = accountBalance * g_RiskPercent / 100.0;
   double loss_for_one_lot = 0;
   ENUM_ORDER_TYPE orderType = (entryPrice > stopLossPrice) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   if(!OrderCalcProfit(orderType, _Symbol, 1.0, entryPrice, stopLossPrice, loss_for_one_lot)) return 0.0;
   double loss_for_one_lot_abs = MathAbs(loss_for_one_lot); 
   if(loss_for_one_lot_abs <= 0) return 0.0;
   double lotSize = riskAmount / loss_for_one_lot_abs;
   double minVolume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxVolume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double volStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   lotSize = MathFloor(lotSize / volStep) * volStep;
   return(NormalizeDouble(fmin(maxVolume, fmax(minVolume, lotSize)), 2));
}

void EnsureDataFolderExists() 
{ 
   if(!FolderCreate(DATA_FOLDER)) 
   { 
      PrintFormat("Warning: Could not create folder '%s'.", DATA_FOLDER); 
   } 
}

void ManageTrailingStop()
{
   if(!g_EnableTrailingStop || !PositionSelect(_Symbol)) return;
   double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
   double currentSL = PositionGetDouble(POSITION_SL);
   long positionType = PositionGetInteger(POSITION_TYPE);
   MqlTick tick; 
   if(!SymbolInfoTick(_Symbol, tick)) return;
   double pips_to_points = _Point * pow(10, _Digits % 2);
   if(positionType == POSITION_TYPE_BUY)
   {
      if((tick.bid - entryPrice) > (g_TrailingStartPips * pips_to_points))
      {
         double newSL = tick.bid - (g_TrailingStopPips * pips_to_points);
         if(newSL > currentSL || currentSL == 0) 
            trade.PositionModify(_Symbol, newSL, PositionGetDouble(POSITION_TP));
      }
   }
   else if(positionType == POSITION_TYPE_SELL)
   {
      if((entryPrice - tick.ask) > (g_TrailingStartPips * pips_to_points))
      {
         double newSL = tick.ask + (g_TrailingStopPips * pips_to_points);
         if(newSL < currentSL || currentSL == 0) 
            trade.PositionModify(_Symbol, newSL, PositionGetDouble(POSITION_TP));
      }
   }
}

void CalculateImprovedInitialAccuracy()
{
    if(g_AccuracyLookbackOnInit <= 0 || !EnablePricePredictionDisplay) return;
    if(ArraySize(g_backtest_predictions) == 0) 
    { 
        Print("Cannot calculate initial accuracy: backtest predictions not loaded."); 
        return; 
    }
    
    PrintFormat("Calculating improved initial accuracy over past %d bars...", g_AccuracyLookbackOnInit);
    
    MqlRates price_data[];
    int bars_needed = g_AccuracyLookbackOnInit + PREDICTION_STEPS + 5;
    
    if(CopyRates(_Symbol, PERIOD_H1, 0, bars_needed, price_data) < bars_needed)
    {
        Print("Not enough historical data available.");
        return;
    }
    
    ArraySetAsSeries(price_data, true);
    bool first_search = true;
    int processed_bars = 0;
    
    for(int i = g_AccuracyLookbackOnInit; i >= PREDICTION_STEPS; i--)
    {
        if(i >= ArraySize(price_data)) continue;
        
        datetime prediction_time = price_data[i].time;
        BacktestPrediction pred_data;
        
        if(!FindPredictionForBar(prediction_time, pred_data, first_search)) continue;
        first_search = false;
        processed_bars++;
        
        double base_price = price_data[i].close;
        
        for(int step = 0; step < PREDICTION_STEPS; step++)
        {
            int target_bar_index = i - (step + 1);
            if(target_bar_index < 0) continue;
            
            double predicted_price = pred_data.predicted_prices[step];
            double target_high = price_data[target_bar_index].high;
            double target_low = price_data[target_bar_index].low;
            double target_close = price_data[target_bar_index].close;
            
            ENUM_PREDICTION_DIRECTION direction = DIR_NEUTRAL;
            double price_diff = predicted_price - base_price;
            double threshold = 0.0001;
            
            if(price_diff > threshold)
                direction = DIR_BULLISH;
            else if(price_diff < -threshold)
                direction = DIR_BEARISH;
            
            bool was_hit = false;
            
            if(direction == DIR_BULLISH)
            {
                if(predicted_price <= target_high || target_close > base_price)
                    was_hit = true;
            }
            else if(direction == DIR_BEARISH)
            {
                if(predicted_price >= target_low || target_close < base_price)
                    was_hit = true;
            }
            else
            {
                double price_change = MathAbs(target_close - base_price);
                if(price_change <= 0.0005)
                    was_hit = true;
            }
            
            g_total_predictions[step]++;
            if(was_hit) g_total_hits[step]++;
        }
    }
    
    PrintFormat("Initial accuracy calculation complete. Processed %d bars.", processed_bars);
    for(int i = 0; i < PREDICTION_STEPS; i++)
    {
        if(g_total_predictions[i] > 0)
        {
            double acc = ((double)g_total_hits[i] / (double)g_total_predictions[i]) * 100.0;
            PrintFormat("Step %d: %d hits / %d predictions = %.1f%%", 
                       i+1, g_total_hits[i], g_total_predictions[i], acc);
        }
        else
        {
            PrintFormat("Step %d: No predictions", i+1);
        }
    }
    g_backtest_prediction_idx = 0;
}

//+------------------------------------------------------------------+
//| MQL5 Main Event Handlers
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=== Siobhan EA v1.86 Initializing ===");
   InitializeParameters();
   
   if(MQLInfoInteger(MQL_TESTER) || g_AccuracyLookbackOnInit > 0)
   { 
      if(!LoadBacktestPredictions()) 
      { 
         if(MQLInfoInteger(MQL_TESTER)) 
         {
            Print("FATAL: In tester mode, cannot continue without backtest file. Check location and name.");
            return(INIT_FAILED); 
         }
      } 
   }
   EnsureDataFolderExists();
   
   string symbols[]={Symbol_EURJPY,Symbol_USDJPY,Symbol_GBPUSD,Symbol_EURGBP,Symbol_USDCAD,Symbol_USDCHF};
   for(int i=0;i<ArraySize(symbols);i++) SymbolSelect(symbols[i],true);
   
   atr_handle=iATR(_Symbol,PERIOD_H1,g_ATR_Period);
   macd_handle=iMACD(_Symbol,PERIOD_H1,12,26,9,PRICE_CLOSE);
   rsi_handle=iRSI(_Symbol,PERIOD_H1,14,PRICE_CLOSE);
   stoch_handle=iStochastic(_Symbol,PERIOD_H1,14,3,3,MODE_SMA,STO_LOWHIGH);
   cci_handle=iCCI(_Symbol,PERIOD_H1,20,PRICE_TYPICAL);
   adx_handle=iADX(_Symbol,PERIOD_H1,g_ADX_Period);
   bb_handle = iBands(_Symbol, PERIOD_H1, 20, 0, 2, PRICE_CLOSE);
   
   if(atr_handle==INVALID_HANDLE||macd_handle==INVALID_HANDLE||rsi_handle==INVALID_HANDLE||stoch_handle==INVALID_HANDLE||cci_handle==INVALID_HANDLE||adx_handle==INVALID_HANDLE||bb_handle==INVALID_HANDLE)
   { 
      Print("FATAL: Failed to create one or more indicator handles."); 
      return(INIT_FAILED); 
   }
     
   ArrayInitialize(g_last_predictions,0.0); 
   ArrayInitialize(g_accuracy_pct,0.0);
   ArrayInitialize(g_total_hits,0); 
   ArrayInitialize(g_total_predictions,0);
   ArrayFree(g_step_predictions);
   
   CreateDisplayPanel();
   if(MQLInfoInteger(MQL_TESTER))
   {
      CalculateImprovedInitialAccuracy();
   }
   UpdateDisplayPanel();
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   IndicatorRelease(atr_handle); 
   IndicatorRelease(macd_handle); 
   IndicatorRelease(rsi_handle);
   IndicatorRelease(stoch_handle); 
   IndicatorRelease(cci_handle); 
   IndicatorRelease(adx_handle);
   IndicatorRelease(bb_handle);
   DeleteDisplayPanel(); 
   Comment("");
}

void OnTick()
{
   if(PositionsTotal()>0&&PositionSelect(_Symbol))
   {
      MqlTick tick; 
      if(SymbolInfoTick(_Symbol,tick))
      {
         if(UseMarketOrderForTP&&g_active_trade_target_price>0)
         {
            if((PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY&&tick.bid>=g_active_trade_target_price)||(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_SELL&&tick.ask<=g_active_trade_target_price))
            { 
               PrintFormat("Closing by market TP hit: #%d",(long)PositionGetInteger(POSITION_TICKET)); 
               trade.PositionClose(_Symbol); 
               g_active_trade_target_price=0; 
               return; 
            }
         }
         if(g_EnableTimeBasedExit)
         {
            datetime deadline=(datetime)PositionGetInteger(POSITION_TIME)+(g_MaxPositionHoldBars*PeriodSeconds(PERIOD_H1));
            MqlDateTime now; 
            TimeToStruct(TimeCurrent(),now);
            if(TimeCurrent()>=deadline&&now.min>=g_ExitBarMinute)
            { 
               PrintFormat("Closing by time limit: #%d",(long)PositionGetInteger(POSITION_TICKET)); 
               trade.PositionClose(_Symbol); 
               g_active_trade_target_price=0; 
               return; 
            }
         }
      }
      ManageTrailingStop();
      return;
   }

   static datetime last_bar_time=0;
   datetime current_bar_time=iTime(_Symbol,PERIOD_H1,0);
   if(current_bar_time==last_bar_time) return;
   last_bar_time=current_bar_time;
   
   g_active_trade_target_price=0;
   g_last_confidence_score = 0.0;
   
   CheckStepPredictionAccuracy();
   
   if(TradingLogicMode==MODE_TRADING_DISABLED) return;
   
   DaemonResponse response;
   bool got_prediction=false;
   
   if(MQLInfoInteger(MQL_TESTER))
   {
      BacktestPrediction current_pred;
      if(FindPredictionForBar(iTime(_Symbol,PERIOD_H1,1),current_pred, false))
      {
         ArrayCopy(response.prices, current_pred.predicted_prices, 0, 0, PREDICTION_STEPS);
         response.confidence_score = current_pred.confidence_score;
         response.buy_prob = current_pred.buy_prob;
         response.sell_prob = current_pred.sell_prob;
         got_prediction=true;
      }
   }
   else
   {
      double features[FEATURE_COUNT * SEQ_LEN];
      int data_needed = SEQ_LEN + 30;
      MqlRates rates[]; 
      if(CopyRates(_Symbol, PERIOD_H1, 1, data_needed, rates) < data_needed) return;
      double macd[],rsi[],stoch_k[],cci[],upper_bb[],lower_bb[],atr[],ej_c[],uj_c[],gu_c[],eg_c[],uc_c[],uchf_c[];
      if(CopyBuffer(macd_handle,0,1,data_needed,macd)<data_needed||CopyBuffer(rsi_handle,0,1,data_needed,rsi)<data_needed||
         CopyBuffer(stoch_handle,0,1,data_needed,stoch_k)<data_needed||CopyBuffer(cci_handle,0,1,data_needed,cci)<data_needed||
         CopyBuffer(bb_handle,1,1,data_needed,upper_bb)<data_needed||CopyBuffer(bb_handle,2,1,data_needed,lower_bb)<data_needed||
         CopyBuffer(atr_handle,0,1,data_needed,atr)<data_needed) return;
      if(CopyClose(Symbol_EURJPY,PERIOD_H1,1,data_needed,ej_c)<data_needed||CopyClose(Symbol_USDJPY,PERIOD_H1,1,data_needed,uj_c)<data_needed||
         CopyClose(Symbol_GBPUSD,PERIOD_H1,1,data_needed,gu_c)<data_needed||CopyClose(Symbol_EURGBP,PERIOD_H1,1,data_needed,eg_c)<data_needed||
         CopyClose(Symbol_USDCAD,PERIOD_H1,1,data_needed,uc_c)<data_needed||CopyClose(Symbol_USDCHF,PERIOD_H1,1,data_needed,uchf_c)<data_needed) return;
         
      int feature_index=0;
      for(int i=SEQ_LEN-1; i>=0; i--)
      {
         double eurusd_ret=(rates[i].close/rates[i+1].close)-1.0;
         features[feature_index++]=eurusd_ret; 
         features[feature_index++]=(double)rates[i].tick_volume;
         features[feature_index++]=atr[i]; 
         features[feature_index++]=macd[i]; 
         features[feature_index++]=rsi[i];
         features[feature_index++]=stoch_k[i]; 
         features[feature_index++]=cci[i];
         MqlDateTime dt;
         TimeToStruct(rates[i].time,dt);
         features[feature_index++]=(double)dt.hour; 
         features[feature_index++]=(double)dt.day_of_week;
         double eurjpy_ret=(ej_c[i]/ej_c[i+1])-1.0, usdjpy_ret=(uj_c[i]/uj_c[i+1])-1.0, gbpusd_ret=(gu_c[i]/gu_c[i+1])-1.0;
         double eurgbp_ret=(eg_c[i]/eg_c[i+1])-1.0, usdcad_ret=(uc_c[i]/uc_c[i+1])-1.0, usdchf_ret=(uchf_c[i]/uchf_c[i+1])-1.0;
         features[feature_index++]=(usdjpy_ret+usdcad_ret+usdchf_ret)-(eurusd_ret+gbpusd_ret);
         features[feature_index++]=eurusd_ret+eurjpy_ret+eurgbp_ret;
         features[feature_index++]=-(eurjpy_ret+usdjpy_ret);
         features[feature_index++]=(upper_bb[i]-lower_bb[i])/(rates[i].close+1e-10);
         features[feature_index++]=(double)rates[i].tick_volume-(double)rates[i+5].tick_volume;
         double body=MathAbs(rates[i].close-rates[i].open); 
         double range=rates[i].high-rates[i].low; 
         double bar_type=0;
         if(range>0&&(body/range)<0.1)bar_type=1.0;
         if(rates[i].close>rates[i].open&&rates[i].open<rates[i+1].open&&rates[i].close>rates[i+1].close)bar_type=2.0;
         if(rates[i].close<rates[i].open&&rates[i].open>rates[i+1].open&&rates[i].close<rates[i+1].close)bar_type=-2.0;
         bar_type+=(rates[i].open-rates[i+1].close)/(atr[i]+1e-10);
         features[feature_index++]=bar_type;
      }
        
      MqlTick tick; 
      if(!SymbolInfoTick(_Symbol, tick)) return;
      if(SendToDaemon(features, tick.ask, atr[0], response))
      { 
         got_prediction = true; 
      }
   }
     
   if(got_prediction)
   {
        ArrayCopy(g_last_predictions, response.prices, 0, 0, PREDICTION_STEPS);
        g_last_confidence_score = response.confidence_score;
        AddStepPredictions(response.prices, current_bar_time);
   }

   UpdateDisplayPanel();
   if(!got_prediction)return;

   if(g_EnableADXFilter)
   {
      double adx[];
      if(CopyBuffer(adx_handle,0,1,1,adx)<1||adx[0]<g_ADX_Threshold)return;
   }
   MqlTick latest_tick;
   if(!SymbolInfoTick(_Symbol,latest_tick))return;
   double atr_val[];
   if(CopyBuffer(atr_handle,0,1,1,atr_val)<1)return;
   double pips_to_points=_Point*pow(10,_Digits%2);
   
   if(g_last_confidence_score < g_MinimumModelConfidence)
   {
      Print("Skipping trade. Model confidence ", DoubleToString(g_last_confidence_score,2), " < threshold ", DoubleToString(g_MinimumModelConfidence,2));
      return;
   }
   
   double target_price=response.prices[TakeProfitTargetBar];
   double spread_points=SymbolInfoInteger(_Symbol,SYMBOL_SPREAD)*_Point;
   double min_profit_points=g_MinProfitPips*pips_to_points;
   int bullish_steps=0,bearish_steps=0;
   for(int i=0;i<PREDICTION_STEPS;i++)
   {
      if(response.prices[i]>latest_tick.ask)bullish_steps++;
      if(response.prices[i]<latest_tick.bid)bearish_steps++;
   }
   double bullish_confidence=(double)bullish_steps/PREDICTION_STEPS;
   double bearish_confidence=(double)bearish_steps/PREDICTION_STEPS;
   
   bool is_regression_buy = bullish_steps>=g_RequiredConsistentSteps && bullish_confidence>=g_MinimumSignalConfidence;
   bool is_regression_sell = bearish_steps>=g_RequiredConsistentSteps && bearish_confidence>=g_MinimumSignalConfidence;
   bool is_classification_buy = response.buy_prob > g_ClassificationSignalThreshold;
   bool is_classification_sell = response.sell_prob > g_ClassificationSignalThreshold;

   bool buy_signal = false;
   bool sell_signal = false;

   if(TradingLogicMode == MODE_REGRESSION_ONLY)
   {
       buy_signal = is_regression_buy;
       sell_signal = is_regression_sell;
   }
   else if(TradingLogicMode == MODE_COMBINED)
   {
       buy_signal = is_regression_buy && is_classification_buy;
       sell_signal = is_regression_sell && is_classification_sell;
   }

   if(buy_signal && (target_price-latest_tick.ask) > (min_profit_points+spread_points))
   {
      double sl=(StopLossMode==SL_STATIC_PIPS)?latest_tick.ask-(g_StaticStopLossPips*pips_to_points):latest_tick.ask-(atr_val[0]*g_ATR_SL_Multiplier);
      if(latest_tick.ask-sl>0&&(target_price-latest_tick.ask)/(latest_tick.ask-sl)>=g_MinimumRiskRewardRatio)
      {
         double tp;
         if(TakeProfitMode==TP_REGRESSION_TARGET)tp=target_price;
         else if(TakeProfitMode==TP_STATIC_PIPS)tp=latest_tick.ask+(g_StaticTakeProfitPips*pips_to_points);
         else tp=latest_tick.ask+(atr_val[0]*g_ATR_TP_Multiplier);
         if(UseMarketOrderForTP)
         {
            tp=0;
            g_active_trade_target_price=(TakeProfitMode==TP_REGRESSION_TARGET)?target_price:tp;
         }
         double lots=CalculateLotSize(sl,latest_tick.ask);
         if(lots>0&&trade.Buy(lots,_Symbol,latest_tick.ask,sl,tp,"Siobhan LSTM Buy"))return;
      }
   }
   else if(sell_signal && (latest_tick.bid-target_price) > (min_profit_points+spread_points))
   {
      double sl=(StopLossMode==SL_STATIC_PIPS)?latest_tick.bid+(g_StaticStopLossPips*pips_to_points):latest_tick.bid+(atr_val[0]*g_ATR_SL_Multiplier);
      if(sl-latest_tick.bid>0&&(latest_tick.bid-target_price)/(sl-latest_tick.bid)>=g_MinimumRiskRewardRatio)
      {
         double tp;
         if(TakeProfitMode==TP_REGRESSION_TARGET)tp=target_price;
         else if(TakeProfitMode==TP_STATIC_PIPS)tp=latest_tick.bid-(g_StaticTakeProfitPips*pips_to_points);
         else tp=latest_tick.bid-(atr_val[0]*g_ATR_TP_Multiplier);
         if(UseMarketOrderForTP)
         {
            tp=0;
            g_active_trade_target_price=(TakeProfitMode==TP_REGRESSION_TARGET)?target_price:tp;
         }
         double lots=CalculateLotSize(sl,latest_tick.bid);
         if(lots>0&&trade.Sell(lots,_Symbol,latest_tick.bid,sl,tp,"Siobhan LSTM Sell"))return;
      }
   }
}

double OnTester()
{
   double history_profits[]; 
   HistorySelect(0, TimeCurrent());
   int deals = HistoryDealsTotal(), profit_count = 0; 
   if(deals <= 1) return 0.0;
   ArrayResize(history_profits, deals);
   for(int i = 0; i < deals; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket > 0 && HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT) 
         history_profits[profit_count++] = HistoryDealGetDouble(ticket, DEAL_PROFIT);
   }
   if(profit_count <= 1) return 0.0; 
   ArrayResize(history_profits, profit_count);
   double mean_profit = MathMean(history_profits);
   double std_dev_profit = MathStandardDeviation(history_profits);
   if(std_dev_profit < 0.0001) return 0.0;
   double sharpe_ratio = mean_profit / std_dev_profit;
   double custom_criterion = sharpe_ratio * MathSqrt(profit_count);
   PrintFormat("OnTester Pass Complete: Trades=%d, Mean Profit=%.2f, StdDev=%.2f, Sharpe=%.3f, Custom Criterion=%.3f", 
              profit_count, mean_profit, std_dev_profit, sharpe_ratio, custom_criterion);
   return custom_criterion;
}
//+------------------------------------------------------------------+