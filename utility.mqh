//+------------------------------------------------------------------+
//|                                                      utility.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property strict

#import "kernel32.dll"
  bool CopyFileW(string lpExistingFileName, string lpNewFileName, bool bFailIfExists);
  bool DeleteFileW(string lpFileName);
#import

string FILE_NAME = "FOREX.csv";
string NEW_DATA_NAME = "flag_go";
double slPoints = 1940;
double tpPoints = 640;
double slPointsScalp = 220;
double tpPointsScalp = 250;
int MAGIC = 10000;

bool isNewCandle(string symbol)
  {
   static datetime timeLastCandle=0;
   if(iTime(Symbol(),0,0)==timeLastCandle)
      return (false);
   timeLastCandle=iTime(Symbol(),0,0);
   return(true);
  }

void writeData(string &symbols[], int size){
     
     string toWrite = TimeToString(iTime(Symbol(), 0, 1));
     // Iterate for each Symbol
     for( int i = 0; i< size; i++){
     
          double close = iClose(symbols[i], 0, 1);
          double open = iOpen(symbols[i], 0, 1);
          double high = iHigh(symbols[i], 0, 1);
          double low = iLow(symbols[i], 0, 1);
          double body = 0;
          double highPips = 0;
          double lowInPips = 0;
          // Green candle
          if( close >= open) {
              body = close - open;
              lowInPips = open - low;
              highPips = high - close;
          }
          else{
               body = close - open;
               lowInPips = close - low;
               highPips = high - open;
          }
          double highBand = high - iBands(symbols[i], 0, 50, 2.55, 0, PRICE_HIGH, 1, 1);
          double lowBand = low - iBands(symbols[i], 0, 50, 2.55, 0, PRICE_LOW, 2, 1);
          double distFromMVAVG100 = close - iMA(symbols[i], 0, 100, 0, 1, PRICE_CLOSE, 1);
          double distFromMVAVG200 = close - iMA(symbols[i], 0, 150, 0, 1, PRICE_CLOSE, 1);
          double distFromMVAVG25 = close - iMA(symbols[i], 0, 50, 0, 1, PRICE_CLOSE, 1);
          double avg25 = iMA(symbols[i], 0, 20, 0, 1, PRICE_CLOSE, 1);
          double avg50 = iMA(symbols[i], 0, 50, 0, 1, PRICE_CLOSE, 1);
          //toWrite = StringConcatenate(toWrite, ",", DoubleToStr(close, 5), ",", DoubleToStr(body, 5), ",", DoubleToStr(lowInPips, 5), ",", DoubleToStr(highPips, 5), 
          //
          // {"_CLOSE", "_BODY", "_LOWINPIPS", "_HIGHINPIPS", "_HIGHBAND50", "_LOWBAND50", "_DISTAVG200", "_DISTAVG100" };
          toWrite = StringConcatenate(toWrite, ",", DoubleToStr(close, 5), ",", DoubleToStr(body, 5), ",", DoubleToStr(lowInPips, 5), ",", DoubleToStr(highPips, 5), "," , DoubleToStr(highBand, 5), ",", DoubleToStr(lowBand, 5), ",", DoubleToStr(distFromMVAVG200), ",", DoubleToStr(distFromMVAVG100,5), ",", DoubleToStr(distFromMVAVG25, 5), "," + DoubleToStr(open, 5) , "," , DoubleToStr(avg25, 5) , "," , DoubleToStr(avg50, 5));   
         
        
     }
      
      ResetLastError();
      int file_handle=FileOpen(FILE_NAME, FILE_READ|FILE_WRITE);
      if( file_handle != INVALID_HANDLE)
      { 
          FileSeek(file_handle, 0, SEEK_END);
          PrintFormat("%s file is available for writing","FILE");
           
          FileWrite(file_handle, toWrite);
            //--- close the file
          FileClose(file_handle);
       
       }
       else
          PrintFormat("Failed to open %s file, Error code = %d","FILE",GetLastError());
      
      

      //ShellExecuteW (NULL, "open", "cmd", "/c copy /Y " + FILE_NAME +  " "+ "C:\\Users\\ed7\\Desktop", NULL, NULL);
      string src_path = "C:\\Users\\ed7\\AppData\\Roaming\\MetaQuotes\\Terminal\\1DAFD9A7C67DC84FE37EAA1FC1E5CF75\\MQL4\\Files\\" + FILE_NAME;
      string dst_path = "E:\\o.csv";
      CopyFileW(src_path, dst_path, false);

      
      file_handle=FileOpen(NEW_DATA_NAME, FILE_READ|FILE_WRITE);
      if( file_handle != INVALID_HANDLE)
      { 
          FileSeek(file_handle, 0, SEEK_END);
          PrintFormat("%s file is available for writing",NEW_DATA_NAME);
         

            //--- write the time and values of signals to the file
           
          FileWrite(file_handle, "GO");
            //--- close the file
          FileClose(file_handle);
       
       }
       else
          PrintFormat("Failed to open %s file, Error code = %d","FILE",GetLastError());
          
          
      src_path = "C:\\Users\\ed7\\AppData\\Roaming\\MetaQuotes\\Terminal\\1DAFD9A7C67DC84FE37EAA1FC1E5CF75\\MQL4\\Files\\" + NEW_DATA_NAME;
      dst_path = "E:\\flag_go";
      int res = CopyFileW(src_path, dst_path, false);
      while(res == false) Print("ERROR FLAG GO");
      FileDelete(src_path);
     

}

string readAction(){

      string dst_path = "C:\\Users\\ed7\\AppData\\Roaming\\MetaQuotes\\Terminal\\1DAFD9A7C67DC84FE37EAA1FC1E5CF75\\MQL4\\Files\\" + "strategy_done";
      string src_path = "E:\\strategy_done";
      bool res = CopyFileW(src_path, dst_path, false);
      
      while(res != true){
         Sleep(1000 * 10);
         res = CopyFileW(src_path, dst_path, false);
      }
      res = DeleteFileW(src_path);
    
      while(res == False) Print("error strategy done");
      
      // Read Action from strategy -- if here strategy has done 
      
      dst_path = "C:\\Users\\ed7\\AppData\\Roaming\\MetaQuotes\\Terminal\\1DAFD9A7C67DC84FE37EAA1FC1E5CF75\\MQL4\\Files\\" + "ACTION";
      src_path = "E:\\ACTION";
      res = CopyFileW(src_path, dst_path, false);
     
      
      while(res == False) Print("error");
      
      string str = "";
      ResetLastError();
      int file_handle=FileOpen("ACTION", FILE_READ|FILE_WRITE);
      if( file_handle != INVALID_HANDLE)
      { 
   
          PrintFormat("%s file is available for reading","Action");
         
         int str_size;
         ;
         //--- read data from the file
          while(!FileIsEnding(file_handle))
         {
            //--- find out how many symbols are used for writing the time
            str_size=FileReadInteger(file_handle,INT_VALUE);
            //--- read the string
            str=FileReadString(file_handle,str_size);
            //--- print the string
            PrintFormat(str);
        }
      //--- close the file
      FileClose(file_handle);
     
       }
       else
          PrintFormat("Failed to open %s file, Error code = %d","FILE",GetLastError());
      
       return str;  
      

}

void process_action(string action){
   
      string toWrite = "ID,WHAT,LOT";
      
      if(StringCompare("OUT", action) == 0){
         Print("action is OUT");
          
      }
      if(StringFind(action, "CLOSE") != -1){
         string result[];
         StringSplit(action, StringGetCharacter(",", 0), result);
         double lots = StringToDouble(result[2]);
         int ticket = StringToInteger(result[1]);
         
        
         if( OrderSelect(ticket, SELECT_BY_TICKET) == true){
            int order_type = OrderType();
            double to_close = 0;
            Print(order_type);
            if(order_type == OP_BUY) to_close = Bid;
            if(order_type == OP_SELL) to_close = Ask;
            while(to_close == 0) Print(order_type);
            bool res = OrderClose(ticket, lots, to_close, 2000, clrAntiqueWhite);
            if(res == False){
               Print("Unable to Close", GetLastError());
            }
         }
         else Print("ERRRROR", GetLastError());
         
      
      }
      else if(StringFind(action, "BUY") != -1){
         string result[];
         StringSplit(action, StringGetCharacter(",", 0), result);
         double lots = StringToDouble(result[1]);
         Print("Going To Buy ");
         string scalp = result[2];
         double tp = StrToDouble(result[3]);
         double sl = StrToDouble(result[4]);
         Print(result[1]);
         int ticket = buy(lots, scalp, tp, sl);
      }
      else if(StringFind(action, "SELL") != -1){
         string result[];
         StringSplit(action, StringGetCharacter(",", 0), result);
         double lots = StringToDouble(result[1]);
         double tp = StrToDouble(result[3]);
         double sl = StrToDouble(result[4]);
         
         string scalp = result[2];
         Print(result[1]);
         int ticket = sell(lots, scalp, tp, sl);
      }
  
      ResetLastError();
      FileDelete("orders");
      int file_handle=FileOpen("orders", FILE_WRITE|FILE_CSV, ",");
      if( file_handle != INVALID_HANDLE)
      { 
          
          
           
          FileWrite(file_handle, "ID", "OPEN_AT", "TIME", "SYMBOL", "LOTS", "PROFIT");
          
          int total = OrdersTotal();
          for( int pos = 0; pos < total; pos ++){
             if(OrderSelect(pos, SELECT_BY_POS) == false) continue;
             FileWrite(file_handle, OrderTicket(), OrderOpenPrice(), OrderOpenTime(),
             OrderSymbol(), OrderLots(), OrderProfit());
          
          }
            //--- close the file
          FileClose(file_handle);
       
       }
       else
          PrintFormat("Failed to open %s file, Error code = %d","FILE",GetLastError());
      
      //ShellExecuteW (NULL, "open", "cmd", "/c copy /Y " + FILE_NAME +  " "+ "C:\\Users\\ed7\\Desktop", NULL, NULL);
      string src_path = "C:\\Users\\ed7\\AppData\\Roaming\\MetaQuotes\\Terminal\\1DAFD9A7C67DC84FE37EAA1FC1E5CF75\\MQL4\\Files\\" + "orders";
      string dst_path = "E:\\orders";
      CopyFileW(src_path, dst_path, false);
}

int buy(double lot, string scalp, double tpU, double slU){
   double sl = NormalizeDouble(Bid - slPoints * Point, Digits);
   double tp = NormalizeDouble(Bid + tpPoints * Point, Digits);
   if( StringCompare(scalp, "scalp") == 0) {
       sl = NormalizeDouble(Bid - slPointsScalp * Point, Digits);
       tp = NormalizeDouble(Bid + tpPointsScalp * Point, Digits);
   }
   
   int ticket = OrderSend(Symbol(), OP_BUY, lot, Ask, 4000, slU, tpU, "Buy Order", MAGIC, 0, clrGreen); 
   if( ticket < 0){
      Print("OrderSend failed error:", GetLastError());
   }
   
   return ticket;
}

int sell(double lot, string scalp, double tpU, double slU){
   double sl = NormalizeDouble(Ask + slPoints * Point, Digits);
   double tp = NormalizeDouble(Ask - tpPoints * Point, Digits);
   if( StringCompare(scalp, "scalp") == 0) {
       sl = NormalizeDouble(Ask + slPointsScalp * Point, Digits);
       tp = NormalizeDouble(Ask - tpPointsScalp * Point, Digits);
   }
   int ticket = OrderSend(Symbol(), OP_SELL, lot, Bid, 4000, slU, tpU, "sell ORder", MAGIC, 0,clrRed); 
   if( ticket < 0){
      Print("OrderSend failed error:", GetLastError());
   }

   return ticket;
}

void move_take_profit(){
  return;
   for(int i = 1; i <= OrdersTotal(); i++){
      if( OrderSelect(i-1, SELECT_BY_POS) == true){
         double open_at = OrderOpenPrice();
         double current_price = (Ask+Bid)/2.0;
         
         if( OrderType() == OP_BUY)
         {  double dist = current_price - open_at;
           
            if( current_price - open_at >= (Point * 650)){
               double new_sl = current_price - (Point * 250);
               new_sl = NormalizeDouble(new_sl, Digits);
                Print("NEW SL");
               Print(DoubleToStr(new_sl));
               bool res = OrderModify(OrderTicket(), OrderOpenPrice(), new_sl, OrderTakeProfit(),0, clrGreen);
              
            }   
         }
         if( OrderType() == OP_SELL)
         {
            if( open_at - current_price >= (Point * 650)){
               double new_sl = current_price + (Point * 250);
                 new_sl = NormalizeDouble(new_sl, Digits);
               Print("NEW SL");
               Print(DoubleToStr(new_sl));
               bool res = OrderModify(OrderTicket(), OrderOpenPrice(), new_sl, OrderTakeProfit(),0, clrGreen);
   
            }  
         }
      }
      
      if( OrderSelect(i-1, SELECT_BY_POS) == true){
         double open_at = OrderOpenPrice();
         double current_price = (Ask+Bid)/2.0;
         
         if( OrderType() == OP_BUY)
         {  double dist = current_price - open_at;
           
            if( current_price - open_at <= -(Point * 450)){
               double new_tp = current_price + (Point * 250);
               new_tp = NormalizeDouble(new_tp, Digits);
                
               bool res = OrderModify(OrderTicket(), OrderOpenPrice(), OrderStopLoss(), new_tp,0, clrGreen);
              
            }   
         }
         if( OrderType() == OP_SELL)
         {
            if( open_at - current_price <=- (Point * 450)){
               double new_tp = current_price - (Point * 250);
                 new_tp = NormalizeDouble(new_tp, Digits);
               Print("NEW SL");
               Print(DoubleToStr(new_tp));
               bool res = OrderModify(OrderTicket(), OrderOpenPrice(), OrderStopLoss(), new_tp,0, clrGreen);
   
            }  
         }
      }
   }

}


void writeDataBack(string &symbols[], int size, int j){
     
     string toWrite = TimeToString(iTime(Symbol(), 0, j));
     // Iterate for each Symbol
     for( int i = 0; i< size; i++){
     
          double close = iClose(symbols[i], 0, j);
          double open = iOpen(symbols[i], 0, j);
          double high = iHigh(symbols[i], 0, j);
          double low = iLow(symbols[i], 0, j);
          double body = 0;
          double highPips = 0;
          double lowInPips = 0;
          // Green candle
          if( close >= open) {
              body = close - open;
              lowInPips = open - low;
              highPips = high - close;
          }
          else{
               body = close - open;
               lowInPips = close - low;
               highPips = high - open;
          }
          double highBand = high - iBands(symbols[i], 0, 50, 2.55, 0, PRICE_HIGH, 1, j);
          double lowBand = low - iBands(symbols[i], 0, 50, 2.55, 0, PRICE_LOW, 2, j);
          double distFromMVAVG100 = close - iMA(symbols[i], 0, 100, 0, 1, PRICE_CLOSE, j);
          double distFromMVAVG200 = close - iMA(symbols[i], 0, 150, 0, 1, PRICE_CLOSE, j);
          double distFromMVAVG25 = close - iMA(symbols[i], 0, 50, 0, 1, PRICE_CLOSE, j);
          double avg25 = iMA(symbols[i], 0, 20, 0, 1, PRICE_CLOSE, j);
          double avg50 = iMA(symbols[i], 0, 50, 0, 1, PRICE_CLOSE, j);
          //toWrite = StringConcatenate(toWrite, ",", DoubleToStr(close, 5), ",", DoubleToStr(body, 5), ",", DoubleToStr(lowInPips, 5), ",", DoubleToStr(highPips, 5), 
          //
          // {"_CLOSE", "_BODY", "_LOWINPIPS", "_HIGHINPIPS", "_HIGHBAND50", "_LOWBAND50", "_DISTAVG200", "_DISTAVG100" };
          toWrite = StringConcatenate(toWrite, ",", DoubleToStr(close, 5), ",", DoubleToStr(body, 5), ",", DoubleToStr(lowInPips, 5), ",", DoubleToStr(highPips, 5), "," , DoubleToStr(highBand, 5), ",", DoubleToStr(lowBand, 5), ",", DoubleToStr(distFromMVAVG200), ",", DoubleToStr(distFromMVAVG100,5), ",", DoubleToStr(distFromMVAVG25, 5), "," + DoubleToStr(open, 5) , "," , DoubleToStr(avg25, 5) , "," , DoubleToStr(avg50, 5));   
         
        
     }
      
      ResetLastError();
      int file_handle=FileOpen(FILE_NAME, FILE_READ|FILE_WRITE);
      if( file_handle != INVALID_HANDLE)
      { 
          FileSeek(file_handle, 0, SEEK_END);
          PrintFormat("%s file is available for writing","FILE");
           
          FileWrite(file_handle, toWrite);
            //--- close the file
          FileClose(file_handle);
       
       }
       else
          PrintFormat("Failed to open %s file, Error code = %d","FILE",GetLastError());
      
      
     

}
