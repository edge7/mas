//+------------------------------------------------------------------+
//|                                                     FxIsCool.mq4 |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict
#include "../Include/utility.mqh"
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+

string symbols [] = {"uer"};
string indicators [] = {"_CLOSE", "_BODY", "_LOWINPIPS", "_HIGHINPIPS", "_HIGHBAND50", "_LOWBAND50", "_DISTAVG200", "_DISTAVG100", "_DISTAVG25", "_OPEN", "_AVG25", "_AVG50"};

//string indicators [] = {"_CLOSE", "_HIGHBAND50", "_LOWBAND50", "_DISTAVG100", "_DISTAVG50" };
int NUM_SYMBOLS = ArraySize(symbols);
int NUM_INDICATORS = ArraySize( indicators);

int OnInit()
  {
//---
   symbols[0] = Symbol();
   FileDelete(FILE_NAME);
   string toWrite = "TIME";
   for( int i = 0; i< NUM_SYMBOLS; i++)
   {  for( int j = 0; j < NUM_INDICATORS; j++)
      {
           toWrite = StringConcatenate(toWrite, ",", symbols[i], indicators[j]);
      }

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

   for (int j = 100; j>=2;j--)
      writeDataBack(symbols, NUM_SYMBOLS, j );

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
      bool newCandle = isNewCandle(Symbol());

      if( ! newCandle){
          move_take_profit();
          return;
      }

      writeData(symbols, NUM_SYMBOLS);
      string action = readAction();
      //writeSup();
      process_action(action);

      Print(action);

  }
//+------------------------------------------------------------------+

//---