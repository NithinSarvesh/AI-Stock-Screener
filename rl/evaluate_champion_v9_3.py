from __future__ import annotations
import inspect, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from indicators import IndicatorEngine
from rl.v6_inference import PPOV6Inference
from rl.trading_env_v9_2 import StockTradingEnvV92

MODEL_PATH = PROJECT_ROOT / 'models' / 'champion_search_v9_3' / 'A_baseline.zip'
OUT_DIR = PROJECT_ROOT / 'models' / 'champion_search_v9_3' / 'validation'
HISTORY = '5y'
EPISODE_LENGTH = 252
STOCKS = {
    'RELIANCE':'RELIANCE.NS','TCS':'TCS.NS','INFY':'INFY.NS',
    'HDFCBANK':'HDFCBANK.NS','ICICIBANK':'ICICIBANK.NS','SBIN':'SBIN.NS',
    'ITC':'ITC.NS','LT':'LT.NS','BHARTIARTL':'BHARTIARTL.NS','AXISBANK':'AXISBANK.NS'
}
ENV_KWARGS = {
    'initial_balance':100000.0,'transaction_cost':0.0005,'episode_length':EPISODE_LENGTH,
    'random_start':False,'drawdown_penalty':0.05,'downside_penalty':0.02,
    'directional_weight':0.003,'turnover_penalty':0.005,
}
ACTION_NAMES = ['SHORT','HALF_SHORT','FLAT','HALF_LONG','LONG']

def flatten(df):
    if hasattr(df.columns,'nlevels') and df.columns.nlevels > 1:
        df=df.copy(); df.columns=df.columns.get_level_values(0)
    return df

def prepare(ticker):
    print(f'Downloading: {ticker}')
    df=flatten(yf.download(ticker,period=HISTORY,interval='1d',auto_adjust=False,progress=False,threads=False))
    if df.empty:
        print('  REJECT: empty data'); return None
    try:
        df=IndicatorEngine(df).calculate_all()
        df=PPOV6Inference.add_context_features(df)
    except Exception as e:
        print(f'  REJECT: feature preparation failed: {e}'); return None
    df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=['Close']).copy()
    if len(df)<300:
        print(f'  REJECT: only {len(df)} usable rows'); return None
    print(f'  ACCEPT: {len(df)} rows')
    return df

def make_env(df):
    accepted=set(inspect.signature(StockTradingEnvV92.__init__).parameters)
    kwargs={k:v for k,v in ENV_KWARGS.items() if k in accepted}
    return StockTradingEnvV92(df,**kwargs)

def evaluate(model,name,df):
    n=len(df); train_end=int(n*.60); val_end=int(n*.80)
    test=df.iloc[val_end:].copy()
    context_start=max(0,val_end-250)
    eval_df=df.iloc[context_start:].copy()
    env=make_env(eval_df)
    obs,info=env.reset(seed=42)
    actions=[]; rewards=[]; equity=float(info.get('equity',100000.0)); trades=0
    for _ in range(EPISODE_LENGTH):
        action,_=model.predict(obs,deterministic=True)
        action=int(np.asarray(action).reshape(-1)[0]); actions.append(action)
        obs,reward,terminated,truncated,step_info=env.step(action)
        rewards.append(float(reward)); equity=float(step_info.get('equity',equity)); trades=int(step_info.get('trade_count',trades))
        if terminated or truncated: break
    start=float(info.get('equity',100000.0))
    strategy_return=equity/start-1 if start else 0.0
    bh=float(test['Close'].iloc[-1]/test['Close'].iloc[0]-1)
    r=pd.Series(rewards,dtype=float)
    sharpe=float(r.mean()/r.std(ddof=1)*np.sqrt(252)) if len(r)>1 and r.std(ddof=1)>0 else 0.0
    wealth=(1+r).cumprod() if len(r) else pd.Series([1.0]); dd=wealth/wealth.cummax()-1
    counts=np.bincount(actions,minlength=5); max_pct=float(counts.max()/max(1,len(actions)))
    return {'symbol':name,'train_rows':train_end,'validation_rows':val_end-train_end,'test_rows':len(test),
            'test_start':str(test.index[0]),'test_end':str(test.index[-1]),'strategy_return':strategy_return,
            'buy_hold_return':bh,'excess_vs_buy_hold':strategy_return-bh,'max_drawdown':float(dd.min()),
            'sharpe':sharpe,'trades':trades,'max_action_pct':max_pct,
            **{f'action_{ACTION_NAMES[i].lower()}_pct':float(counts[i]/max(1,len(actions))) for i in range(5)}}

def main():
    print('='*80); print('V9.3 SHORTLIST CHAMPION VALIDATION'); print('='*80)
    print(f'Model: {MODEL_PATH}'); print(f'History: {HISTORY}'); print('Split: 60% context / 20% validation / 20% unseen test\n')
    if not MODEL_PATH.exists(): raise FileNotFoundError(f'Champion model not found:\n{MODEL_PATH}')
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    model=PPO.load(str(MODEL_PATH)); results=[]
    for i,(name,ticker) in enumerate(STOCKS.items(),1):
        print('='*80); print(f'[{i}/{len(STOCKS)}] {name}'); print('='*80)
        df=prepare(ticker)
        if df is None: continue
        try:
            x=evaluate(model,name,df); results.append(x)
            print(f"Strategy return : {x['strategy_return']:+.2%}\nBuy & Hold      : {x['buy_hold_return']:+.2%}\nExcess          : {x['excess_vs_buy_hold']:+.2%}\nMax drawdown    : {x['max_drawdown']:+.2%}\nSharpe          : {x['sharpe']:+.3f}\nTrades          : {x['trades']}\nMax action pct  : {x['max_action_pct']:.2%}")
        except Exception as e: print(f'REJECT: evaluation failed: {e}')
    if not results: raise RuntimeError('No stocks were successfully evaluated.')
    out=pd.DataFrame(results)
    summary={'model':str(MODEL_PATH),'stocks_evaluated':len(out),'average_return':float(out.strategy_return.mean()),
             'median_return':float(out.strategy_return.median()),'average_buy_hold':float(out.buy_hold_return.mean()),
             'average_excess':float(out.excess_vs_buy_hold.mean()),'average_max_drawdown':float(out.max_drawdown.mean()),
             'worst_max_drawdown':float(out.max_drawdown.min()),'average_sharpe':float(out.sharpe.mean()),
             'average_trades':float(out.trades.mean()),'max_action_concentration':float(out.max_action_pct.max())}
    rp=OUT_DIR/'champion_test_results.csv'; sp=OUT_DIR/'champion_test_summary.json'; out.to_csv(rp,index=False); sp.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('\n'+'='*80); print('V9.3 CHAMPION VALIDATION COMPLETE'); print('='*80)
    for k,v in summary.items(): print(f'{k:28s}: {v:+.2%}' if isinstance(v,float) and abs(v)<2 else f'{k:28s}: {v}')
    print('\nSaved:'); print(rp); print(sp)

if __name__=='__main__': main()
