import pandas as pd
import pandas_ta as ta
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime as date
import time
import curses
import requests

res2 = requests.get("https://marketdata.tradermade.com/api/v1/timeseries?currency=GBPUSD&api_key=WZBi_G50S5w8nj6RmMeX&start_date=2024-03-01-00:00&end_date=2024-03-29-8:00&format=records&interval=hourly")
data = res2.json()

print(data)
result = data["quotes"]

#print(result)
lst = []
for i in result:
  #open = res2["quotes"][i].close
  lst.append(i)


df = pd.DataFrame(lst)
#data = pd.read_csv("usd15m.csv")
print(df.columns)
#df = pd.DataFrame(data)

df = df[["open", "high", "low", "close"]]

close = df["close"]
open = df["open"]
high = df["high"]
low = df["low"]

df["RSI"] = ta.rsi(close, length=7)
df["SMA"] = ta.sma(close, length=200)

bands = ta.bbands(close, length=14, std=2.5)
df = df.join(bands)

df.dropna(inplace=True)
df.reset_index(inplace=True)

df = df[["open", "high", "low", "close", "RSI", "SMA", "BBU_14_2.5", "BBL_14_2.5"]]

#trend check
def trend_checker(df, candles):
  trend = [0] * len(df)
  for i in range(len(df)):
    count = 0
    for j in range(i-candles, i):
      if df.iloc[j]["close"] > df.iloc[j]["SMA"]:
        count = count + 1
        if count == candles:
          trend[i] = 1
      elif df.iloc[i]["close"] < df.iloc[i]["SMA"]:
        count = count + 1
        if count == candles:
          trend[i] = 2
  return trend

t = trend_checker(df, 7)

df["Trend"] = t


def signals(df):
  sigs = [0] * len(df)
  for i in range(len(df)):
    if df.iloc[i]["Trend"] == 1 and df.iloc[i]["close"] > df.iloc[i]["BBU_14_2.5"] and df.iloc[i]["RSI"] > 75:
      sigs[i] = 2
    elif df.iloc[i]["Trend"] == 2 and df.iloc[i]["close"] < df.iloc[i]["BBL_14_2.5"] and df.iloc[i]["RSI"] < 25:
      sigs[i] = 1
  return sigs

s = signals(df)
df["Signals"] = s

m_df = df[["high","low","SMA","RSI","BBU_14_2.5","BBL_14_2.5","Trend","Signals"]].copy()

x = m_df.values

x = torch.FloatTensor(x)

#print(x)



#loading fat model
class Fat(nn.Module):
    def __init__(self, in_feature=8, h1=16, h2=32, h3=16, h4=8, out_feature=3):
        super().__init__()
        self.fc1 = nn.Linear(in_feature, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.out = nn.Linear(h4, out_feature)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.out(x)
        return x

torch.manual_seed(31)
model = Fat()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#model = Fat()

model.load_state_dict(torch.load("fat.pt"))

model.eval()
#creating the display area
def main(stdscr):
  curses.curs_set(0)
  curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
  curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
  red = curses.color_pair(1)
  green = curses.color_pair(2)
  sh, sw = stdscr.getmaxyx()
  bw = sw//5
  stdscr.addstr(2,1, "CURENCY PAIRS:", green)
  stdscr.addstr(2, 16, "EURUSD", curses.A_STANDOUT)
  stdscr.addstr(15,1, "INDICATORS", curses.A_STANDOUT)
  head1 = curses.newwin(3,bw, 5, 1)
  head1.box()
  head2 = curses.newwin(3,bw, 5, 1+bw)
  head2.box()
  head3 = curses.newwin(3,bw, 5, 1+bw*2)
  head3.box()
  head4 = curses.newwin(3,bw, 5, 1+bw*3)
  head4.box()
  head5 = curses.newwin(3,bw, 5, 1+bw*4)
  head5.box()
  body1 = curses.newwin(6, bw, 8,1)
  body1.box()
  body2 = curses.newwin(6, bw, 8,1+bw)
  body2.box()
  body3 = curses.newwin(6, bw, 8,1+bw*2)
  body3.box()
  body4 = curses.newwin(6, bw, 8,1+bw*3)
  body4.box()
  body5 = curses.newwin(6, bw, 8,1+bw*4)
  body5.box()
  pad1 = curses.newpad(len(df),len(df))
  pad2 = curses.newpad(len(df),len(df))
  pad3 = curses.newpad(len(df),len(df))
  pad4 = curses.newpad(len(df),len(df))
  pad5 = curses.newpad(len(df),len(df))
  head6 = curses.newwin(3,bw, 16, 1)
  head6.box()
  head7 = curses.newwin(3,bw, 16, 1+bw)
  head7.box()
  head8 = curses.newwin(3,bw, 16, 1+bw*2)
  head8.box()
  head9 = curses.newwin(3,bw, 16, 1+bw*3)
  head9.box()
  head10 = curses.newwin(3,bw, 16, 1+bw*4)
  head10.box()
  body6 = curses.newwin(6, bw, 19,1)
  body6.box()
  body7 = curses.newwin(6, bw, 19,1+bw)
  body7.box()
  body8 = curses.newwin(6, bw, 19,1+bw*2)
  body8.box()
  body9 = curses.newwin(6, bw, 19,1+bw*3)
  body9.box()
  body10 = curses.newwin(6, bw, 19,1+bw*4)
  body10.box()
  pad6 = curses.newpad(len(df),len(df))
  pad7 = curses.newpad(len(df),len(df))
  pad8 = curses.newpad(len(df),len(df))
  pad9 = curses.newpad(len(df),len(df))
  pad10 = curses.newpad(len(df),len(df))
  stdscr.addstr(26,1, "BOT AND ALGORITHYM SIGNALS", curses.A_STANDOUT)
  bwin_h = curses.newwin(3, bw, 27, 1)
  bwin_h.box()
  bwin_b = curses.newwin(6, bw, 29, 1)
  bwin_b.box()
  bwin_p = curses.newpad(len(df),len(df))
  swin_h = curses.newwin(3, bw, 27, 1+bw+2)
  swin_h.box()
  swin_b = curses.newwin(6, bw, 29, 1+bw+2)
  swin_b.box()
  swin_p = curses.newpad(len(df),len(df))
  loop = True
  while loop:
    ch = stdscr.getch()
    if ch == ord("q"):
      loop = False
    for i in range(len(df)):
      pad1.addstr(i,1, str(df.iloc[i]["open"]))
      pad2.addstr(i,1, str(df.iloc[i]["high"]))
      pad3.addstr(i,1, str(df.iloc[i]["low"]))
      pad4.addstr(i,1, str(df.iloc[i]["close"]))
      if df.iloc[i]["close"] > df.iloc[i]["open"]:
        pad5.addstr(i,1, "↑", green)
      else:
        pad5.addstr(i,1, "↓", red)
      pad6.addstr(i,1, str(df.iloc[i]["RSI"]))
      pad7.addstr(i,1, str(df.iloc[i]["SMA"]))
      pad8.addstr(i,1, str(df.iloc[i]["BBU_14_2.5"]))
      pad9.addstr(i,1, str(df.iloc[i]["BBL_14_2.5"]))
      if df.iloc[i]["Trend"] == 1:
        pad10.addstr(i,1, "↑", green)
      elif df.iloc[i]["Trend"] == 2:
        pad10.addstr(i,1, "↓", red)
      else:
        pad10.addstr(i,1, "—")
      pred = model.forward(x[i])
      predv = pred.argmax().item()
      if predv == 1:
        bwin_p.addstr(i,1, "BUY", green)
      elif predv == 2:
        bwin_p.addstr(i,1, "SELL", red)
      else:
        bwin_p.addstr(i,1, "WAIT")
      if df.iloc[i]["Signals"] == 1:
        swin_p.addstr(i,1, "BUY", green)
      elif df.iloc[i]["Signals"] == 2:
        swin_p.addstr(i,1, "SELL", red)
      else:
        swin_p.addstr(i,1, "WAIT")



    head1.addstr(1,2, "open")
    head2.addstr(1,2, "high")
    head3.addstr(1,2, "low")
    head4.addstr(1,2, "close")
    head5.addstr(1,2, "TYPE")
    head6.addstr(1,2, "RSI")
    head7.addstr(1,2, "SMA_200")
    head8.addstr(1,2, "BB_UP")
    head9.addstr(1,2, "BB_DN")
    head10.addstr(1,2, "TREND")
    bwin_h.addstr(1,2, "BOT_PRED")
    swin_h.addstr(1,2, "ALGO_PRED")

    head1.refresh()
    head2.refresh()
    head3.refresh()
    head4.refresh()
    head5.refresh()
    body1.refresh()
    body2.refresh()
    body3.refresh()
    body4.refresh()
    body5.refresh()
    head6.refresh()
    head7.refresh()
    head8.refresh()
    head9.refresh()
    head10.refresh()
    bwin_h.refresh()
    bwin_b.refresh()
    swin_h.refresh()
    swin_b.refresh()
    body6.refresh()
    body7.refresh()
    body8.refresh()
    body9.refresh()
    body10.refresh()
    for i in range(len(df)):
      pad1.refresh(i,0, 9,2, 12,bw-1)
      pad2.refresh(i,0, 9,bw+2, 12,bw*2-2)
      pad3.refresh(i,0, 9,bw*2+2, 12,bw*3-2)
      pad4.refresh(i,0, 9,bw*3+2, 12,bw*4-2)
      pad5.refresh(i,0, 9,bw*4+2, 12,bw*5-2)
      pad6.refresh(i,0, 20,2, 23,bw-1)
      pad7.refresh(i,0, 20,bw+2, 23,bw*2-2)
      pad8.refresh(i,0, 20,bw*2+2, 23,bw*3-2)
      pad9.refresh(i,0, 20,bw*3+2, 23,bw*4-2)
      pad10.refresh(i,0, 20,bw*4+2, 23,bw*5-2)
      bwin_p.refresh(i,0, 30,2, 33,bw-2)
      swin_p.refresh(i,0, 30,bw+4, 33,bw*2-2)
      stdscr.addstr(4,1, str(time.asctime()), green)
      stdscr.refresh()
      time.sleep(1)

curses.wrapper(main)


