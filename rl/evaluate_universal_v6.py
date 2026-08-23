"""
UNIVERSAL PPO V6 EVALUATOR
==========================
Evaluates the V4 universal policy on validation and unseen test periods
for every stock that can be downloaded.

It reports:
- PPO return
- Buy & Hold
- Cash
- Long-only
- Short-only
- Sharpe
- max drawdown
- trades
- win rate
- profit factor
- action exposure
- transition counts

Run:
    python rl/evaluate_universal_v6.py
"""
import os,sys,json,warnings
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO

PROJECT_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path: sys.path.insert(0,PROJECT_ROOT)
from indicators import IndicatorEngine

TICKERS=["RELIANCE","TCS","INFY","ICICIBANK","SBIN","LT","ITC"]
HISTORY_PERIOD="5y"; MIN_USABLE_ROWS=300
INITIAL_BALANCE=100000.0; TRANSACTION_COST=0.0005
TRAIN_RATIO=0.70; VALIDATION_RATIO=0.15
MODEL_ROOT=os.path.join(PROJECT_ROOT,"models","universal_v6")
MODEL_PATH=os.path.join(MODEL_ROOT,"universal_ppo_v6.zip")
RESULT_CSV=os.path.join(MODEL_ROOT,"universal_v6_evaluation.csv")
RESULT_JSON=os.path.join(MODEL_ROOT,"universal_v6_evaluation.json")
ACTION_DIR=os.path.join(MODEL_ROOT,"actions")
os.makedirs(ACTION_DIR,exist_ok=True)
warnings.filterwarnings("ignore")

BASE_FEATURES=["Open","High","Low","Close","Volume","EMA20","EMA50","EMA200","RSI","MACD","MACD_SIGNAL","MACD_HISTOGRAM","BB_UPPER","BB_MIDDLE","BB_LOWER","VWAP","AVG_VOLUME","ATR","ADX","OBV","STOCH_RSI"]
def clean(df):
    if df is None or df.empty:return None
    df=df.copy()
    if isinstance(df.columns,pd.MultiIndex):
        cols=[]
        for c in df.columns:
            parts=[str(x) for x in c if str(x) not in ("","None")]
            cols.append(next((x for x in parts if x in ["Open","High","Low","Close","Volume"]),parts[-1]))
        df.columns=cols;df=df.loc[:,~df.columns.duplicated()]
    df=df[[c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]].copy()
    if len(df.columns)!=5:return None
    for c in df.columns:df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.replace([np.inf,-np.inf],np.nan).dropna().sort_index().loc[lambda x:x["Close"]>0]

def add(df):
    df=df.copy(); c=df["Close"]
    df["RET_1"]=c.pct_change(1);df["RET_5"]=c.pct_change(5);df["RET_20"]=c.pct_change(20)
    df["VOL_20"]=c.pct_change().rolling(20).std()
    df["EMA20_SLOPE"]=df["EMA20"].pct_change(5);df["EMA50_SLOPE"]=df["EMA50"].pct_change(10)
    df["ATR_PCT"]=df["ATR"]/c.replace(0,np.nan)
    df["TREND_SCORE"]=((c>df["EMA20"]).astype(float)+(df["EMA20"]>df["EMA50"]).astype(float)+(df["EMA50"]>df["EMA200"]).astype(float))/3
    return df.replace([np.inf,-np.inf],np.nan).dropna()

class EvalEnv:
    def __init__(self,df):
        self.df=df;self.pos=0
    def obs(self,row):
        close=max(float(row["Close"]),1e-8);v=[
            0,np.clip(float(row["High"])/close-1,-5,5),np.clip(float(row["Low"])/close-1,-5,5),0,
            np.clip(np.log1p(max(float(row["Volume"]),1)/max(float(row["AVG_VOLUME"]),1)),-10,10)]
        for c in ["EMA20","EMA50","EMA200"]:v.append(np.clip(float(row[c])/close-1,-5,5))
        v.append(np.clip((float(row["RSI"])-50)/50,-1,1))
        for c in ["MACD","MACD_SIGNAL","MACD_HISTOGRAM"]:v.append(np.clip(float(row[c])/close,-1,1))
        for c in ["BB_UPPER","BB_MIDDLE","BB_LOWER"]:v.append(np.clip(float(row[c])/close-1,-5,5))
        v += [np.clip(float(row["VWAP"])/close-1,-5,5),0,np.clip(float(row["ATR"])/close,0,1),np.clip(float(row["ADX"])/100,0,1),np.sign(float(row["OBV"])),np.clip(float(row["STOCH_RSI"])*2-1,-1,1)]
        v += [np.clip(float(row["RET_1"]),-1,1),np.clip(float(row["RET_5"]),-1,1),np.clip(float(row["RET_20"]),-1,1),np.clip(float(row["VOL_20"]),0,1),np.clip(float(row["EMA20_SLOPE"]),-1,1),np.clip(float(row["EMA50_SLOPE"]),-1,1),np.clip(float(row["ATR_PCT"]),0,1),np.clip(float(row["TREND_SCORE"])*2-1,-1,1),float(self.pos)]
        return np.clip(np.nan_to_num(np.asarray(v,dtype=np.float32),nan=0,posinf=10,neginf=-10),-10,10).astype(np.float32)

def load(ticker):
    errs=[]
    for sym in [f"{ticker}.NS",f"{ticker}.BO",ticker]:
        try:
            raw=clean(yf.Ticker(sym).history(period=HISTORY_PERIOD,interval="1d",auto_adjust=True,actions=False,timeout=60))
            if raw is None or len(raw)<MIN_USABLE_ROWS: errs.append(sym+": empty/too-small");continue
            df=add(IndicatorEngine(raw).calculate_all())
            if len(df)>=MIN_USABLE_ROWS:return df,sym
        except Exception as e:errs.append(sym+": "+str(e))
    raise RuntimeError(" | ".join(errs))

def split(df):
    a=int(len(df)*TRAIN_RATIO);b=int(len(df)*(TRAIN_RATIO+VALIDATION_RATIO))
    return df.iloc[:a],df.iloc[a:b],df.iloc[b:]

def sharpe(returns):
    r=np.asarray(returns,float)
    if len(r)<2 or np.std(r,ddof=1)==0:return 0.0
    return float(np.sqrt(252)*np.mean(r)/np.std(r,ddof=1))

def max_dd(equity):
    e=np.asarray(equity,float);peak=np.maximum.accumulate(e)
    return float(np.min(e/np.maximum(peak,1e-12)-1))

def evaluate_policy(model,df):
    env=EvalEnv(df); eq=INITIAL_BALANCE; equity=[eq]; rets=[]; actions=[]; positions=[]; trades=0; costs=0; wins=0;closed=[]
    transitions={}
    obs=env.obs(df.iloc[0])
    for i in range(len(df)-1):
        action,_=model.predict(obs,deterministic=True)
        action=int(np.asarray(action).item()); target={0:-1.0,1:-0.5,2:0.0,3:0.5,4:1.0}[action]
        old=env.pos; change=abs(target-old); cost=change*TRANSACTION_COST
        if change:trades+=1
        transitions[f"{old}->{target}"]=transitions.get(f"{old}->{target}",0)+1
        env.pos=target
        p=float(df.iloc[i]["Close"]);n=float(df.iloc[i+1]["Close"]);mr=n/p-1
        net=target*mr-cost
        eq*=1+float(np.clip(net,-.99,10))
        costs+=cost*eq; rets.append(net);equity.append(eq);actions.append(action);positions.append(target)
        if old!=0 and target!=old: pass
        obs=env.obs(df.iloc[i+1])
    # Trade statistics from each contiguous non-flat position.
    current=None;trade_ret=0
    for pos,r in zip(positions,rets):
        if pos!=current:
            if current not in (None,0):
                closed.append(trade_ret)
            current=pos;trade_ret=0
        if current not in (None,0): trade_ret=(1+trade_ret)*(1+r)-1
    if current not in (None,0):closed.append(trade_ret)
    wins=sum(x>0 for x in closed)
    losses=[x for x in closed if x<0]
    gains=sum(x for x in closed if x>0);loss=sum(abs(x) for x in losses)
    return {
        "initial_equity":INITIAL_BALANCE,"final_equity":eq,
        "return_pct":(eq/INITIAL_BALANCE-1)*100,
        "sharpe":sharpe(rets),"max_drawdown_pct":max_dd(equity)*100,
        "trades":trades,"closed_trades":len(closed),
        "win_rate_pct":(wins/len(closed)*100 if closed else 0),
        "profit_factor":(gains/loss if loss>0 else (float("inf") if gains>0 else 0)),
        "avg_trade_return_pct":(np.mean(closed)*100 if closed else 0),
        "transaction_costs":costs,
        "long_exposure_pct":sum(p>0 for p in positions)/len(positions)*100,
        "short_exposure_pct":sum(p<0 for p in positions)/len(positions)*100,
        "flat_exposure_pct":positions.count(0)/len(positions)*100,
        "average_position":float(np.mean(positions)) if positions else 0.0,
        "transitions":transitions
    }

def benchmark(df,mode):
    r=df["Close"].pct_change().fillna(0).to_numpy()[1:]
    if mode=="cash": rr=np.zeros_like(r)
    elif mode=="long": rr=r
    else: rr=-r
    eq=INITIAL_BALANCE;es=[eq]
    for x in rr:eq*=1+x;es.append(eq)
    return {"return_pct":(eq/INITIAL_BALANCE-1)*100,"sharpe":sharpe(rr),"max_drawdown_pct":max_dd(es)*100}

def main():
    print("="*80);print("AI TRADING ASSISTANT — UNIVERSAL PPO V6 EVALUATION");print("="*80)
    if not os.path.exists(MODEL_PATH):raise FileNotFoundError(MODEL_PATH)
    model=PPO.load(MODEL_PATH);rows=[];all_actions={}
    for ticker in TICKERS:
        print("\n"+"#"*72);print("EVALUATING:",ticker)
        try:
            df,sym=load(ticker);tr,val,test=split(df)
            for name,part in [("validation",val),("test",test)]:
                result=evaluate_policy(model,part)
                bh=benchmark(part,"long");cash=benchmark(part,"cash");short=benchmark(part,"short")
                result.update({"ticker":ticker,"symbol":sym,"period":name,
                                "buy_hold_return_pct":bh["return_pct"],"cash_return_pct":cash["return_pct"],
                                "short_only_return_pct":short["return_pct"],
                                "excess_vs_bh_pct":result["return_pct"]-bh["return_pct"]})
                rows.append(result)
                print(f"{name.upper()}: PPO {result['return_pct']:.2f}% | B&H {bh['return_pct']:.2f}% | Sharpe {result['sharpe']:.2f} | DD {result['max_drawdown_pct']:.2f}% | trades {result['trades']} | L/F/S {result['long_exposure_pct']:.1f}/{result['flat_exposure_pct']:.1f}/{result['short_exposure_pct']:.1f}%")
                if name=="test":
                    all_actions[ticker]=result["transitions"]
        except Exception as e:
            print("FAILED:",ticker,e)
            rows.append({"ticker":ticker,"period":"ERROR","error":str(e)})
    dfout=pd.DataFrame(rows)
    dfout.to_csv(RESULT_CSV,index=False)
    serial=[]
    for x in rows:
        y=dict(x)
        if "transitions" in y:y["transitions"]={str(k):int(v) for k,v in y["transitions"].items()}
        serial.append(y)
    summary={"model":MODEL_PATH,"results":serial,"successful_stocks":len(set(x["ticker"] for x in rows if x.get("period")=="test"))}
    tests=[x for x in rows if x.get("period")=="test"]
    if tests:
        summary["test_aggregate"]={
            "mean_return_pct":float(np.mean([x["return_pct"] for x in tests])),
            "median_return_pct":float(np.median([x["return_pct"] for x in tests])),
            "mean_excess_vs_bh_pct":float(np.mean([x["excess_vs_bh_pct"] for x in tests])),
            "stocks_positive":int(sum(x["return_pct"]>0 for x in tests)),
            "stocks_beating_bh":int(sum(x["excess_vs_bh_pct"]>0 for x in tests)),
            "mean_sharpe":float(np.mean([x["sharpe"] for x in tests])),
            "worst_drawdown_pct":float(min(x["max_drawdown_pct"] for x in tests))
        }
    with open(RESULT_JSON,"w",encoding="utf-8") as f:json.dump(summary,f,indent=2,default=str)
    print("\n"+"="*80);print("V6 EVALUATION COMPLETE");print("CSV:",RESULT_CSV);print("JSON:",RESULT_JSON)
    if tests:
        a=summary["test_aggregate"];print(f"Test mean return: {a['mean_return_pct']:.2f}%");print(f"Mean excess vs B&H: {a['mean_excess_vs_bh_pct']:.2f}%");print(f"Positive: {a['stocks_positive']}/{len(tests)}");print(f"Beat B&H: {a['stocks_beating_bh']}/{len(tests)}");print(f"Mean Sharpe: {a['mean_sharpe']:.2f}");print(f"Worst DD: {a['worst_drawdown_pct']:.2f}%")

if __name__=="__main__":main()
