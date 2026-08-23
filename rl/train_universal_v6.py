"""
UNIVERSAL PPO V6 TRAINER
========================
A deliberately different universal RL formulation designed to address
Universal PPO V3's degenerate "buy once and hold" behavior.

Key changes from V3:
- 30-dimensional observation: existing 22 signals + 8 regime/context features.
- Risk-aware reward using net log return, downside penalty, drawdown penalty,
  turnover penalty, and a very small missed-opportunity penalty.
- Reward uses incremental drawdown pressure instead of only raw return.
- Same 7-stock basket, no ticker identity in observations.
- Same chronological 70/15/15 split; validation/test never enter training.
- Strong action/position diagnostics during training.

Run from project root:
    python rl/train_universal_v6.py
"""
import os, sys, json, time, random, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from indicators import IndicatorEngine

TICKERS = ["RELIANCE","TCS","INFY","ICICIBANK","SBIN","LT","ITC"]
HISTORY_PERIOD = "5y"
MIN_RAW_ROWS = 400
MIN_USABLE_ROWS = 300
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

INITIAL_BALANCE = 100000.0
TRANSACTION_COST = 0.0005
EPISODE_LENGTH = 252

TOTAL_TIMESTEPS = 200_000

LEARNING_RATE = 0.0006
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
SEED = 42

# V4 reward weights. These are intentionally modest; they must not
# overwhelm actual portfolio return.
DRAWDOWN_WEIGHT = 0.10
DOWNSIDE_WEIGHT = 0.35
TURNOVER_WEIGHT = 0.002
MISSED_MOVE_WEIGHT = 0.002
MISSED_MOVE_ATR = 1.25
DIRECTIONAL_ERROR_WEIGHT = 0.002

MODEL_ROOT = os.path.join(PROJECT_ROOT, "models", "universal_v6")
MODEL_DIR = MODEL_ROOT
LOG_DIR = os.path.join(MODEL_ROOT, "logs")
META_DIR = os.path.join(MODEL_ROOT, "metadata")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_ROOT, "universal_ppo_v6")
META_PATH = os.path.join(META_DIR, "universal_ppo_v6_metadata.json")

REQUIRED = ["Open","High","Low","Close","Volume"]

class TrainingCallback(BaseCallback):
    def __init__(self, every=10_000, verbose=0):
        super().__init__(verbose)
        self.every = every
        self.last = 0
    def _on_step(self):
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            print(f"\n[UNIVERSAL PPO V6] {self.num_timesteps:,}/{TOTAL_TIMESTEPS:,} timesteps")
        return True

def clean_ohlcv(df):
    if df is None or df.empty:
        return None
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        cols=[]
        for c in df.columns:
            parts=[str(x) for x in c if str(x) not in ("","None")]
            pick=next((x for x in parts if x in REQUIRED), parts[-1])
            cols.append(pick)
        df.columns=cols
        df=df.loc[:,~df.columns.duplicated()]
    rename={}
    for c in df.columns:
        for r in REQUIRED:
            if str(c).strip().lower()==r.lower():
                rename[c]=r
    df=df.rename(columns=rename)
    missing=[x for x in REQUIRED if x not in df.columns]
    if missing:
        raise ValueError("Missing OHLCV columns: "+", ".join(missing))
    df=df[REQUIRED].copy()
    for c in REQUIRED:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.replace([np.inf,-np.inf],np.nan).dropna().sort_index()
    df=df.loc[~df.index.duplicated(keep="last")]
    return df[df["Close"]>0]

def load_stock(ticker):
    candidates=[f"{ticker}.NS", f"{ticker}.BO", ticker]
    errors=[]
    for symbol in candidates:
        print(f"Trying: {symbol}")
        try:
            raw=yf.Ticker(symbol).history(period=HISTORY_PERIOD, interval="1d",
                                          auto_adjust=True, actions=False, timeout=60)
            raw=clean_ohlcv(raw)
            if raw is not None and len(raw)>=MIN_RAW_ROWS:
                print(f"Resolved: {symbol} | raw rows: {len(raw):,}")
                features=IndicatorEngine(raw).calculate_all()
                features=add_v4_features(features)
                features=features.replace([np.inf,-np.inf],np.nan).dropna().copy()
                if len(features)>=MIN_USABLE_ROWS:
                    print(f"Usable rows: {len(features):,}")
                    return features, symbol
                errors.append(f"{symbol}: only {len(features)} usable rows")
            else:
                errors.append(f"{symbol}: empty/too-small")
        except Exception as e:
            errors.append(f"{symbol}: {e}")
    raise RuntimeError(f"Could not load {ticker}. " + " | ".join(errors))

def add_v4_features(df):
    df=df.copy()
    close=df["Close"].astype(float)
    # Context features. These are calculated from information available
    # at the current bar; no future prices are used.
    df["RET_1"] = close.pct_change(1)
    df["RET_5"] = close.pct_change(5)
    df["RET_20"] = close.pct_change(20)
    df["VOL_20"] = close.pct_change().rolling(20).std()
    df["EMA20_SLOPE"] = df["EMA20"].pct_change(5)
    df["EMA50_SLOPE"] = df["EMA50"].pct_change(10)
    df["ATR_PCT"] = df["ATR"] / close.replace(0,np.nan)
    df["TREND_SCORE"] = (
        (close > df["EMA20"]).astype(float)
        + (df["EMA20"] > df["EMA50"]).astype(float)
        + (df["EMA50"] > df["EMA200"]).astype(float)
    ) / 3.0
    return df

class UniversalV4Env(gym.Env):
    """
    One universal environment. A new episode selects one stock uniformly.
    The ticker is never exposed to the policy.
    """
    metadata={"render_modes":[]}
    BASE_FEATURES=[
        "Open","High","Low","Close","Volume","EMA20","EMA50","EMA200",
        "RSI","MACD","MACD_SIGNAL","MACD_HISTOGRAM","BB_UPPER","BB_MIDDLE",
        "BB_LOWER","VWAP","AVG_VOLUME","ATR","ADX","OBV","STOCH_RSI"
    ]
    CONTEXT_FEATURES=[
        "RET_1","RET_5","RET_20","VOL_20","EMA20_SLOPE",
        "EMA50_SLOPE","ATR_PCT","TREND_SCORE"
    ]
    def __init__(self, data, seed=42):
        super().__init__()
        self.data=data
        self.tickers=list(data)
        self.rng=np.random.default_rng(seed)
        self.observation_space=spaces.Box(low=-10,high=10,shape=(30,),dtype=np.float32)
        self.action_space=spaces.Discrete(5)
        self.env_df=None
        self.current_ticker=None
        self.current_step=0
        self.end_step=0
        self.position=0
        self.equity=INITIAL_BALANCE
        self.peak=INITIAL_BALANCE
        self.max_dd=0.0
        self.trades=0
        self.tx_cost=0.0
        self.long_steps=0
        self.short_steps=0
        self.flat_steps=0
        self.total_reward=0.0
        self.transitions=[]

    def _obs(self,row):
        close=max(float(row["Close"]),1e-8)
        vals=[
            0.0,
            np.clip(float(row["High"])/close-1,-5,5),
            np.clip(float(row["Low"])/close-1,-5,5),
            0.0,
            np.clip(np.log1p(max(float(row["Volume"]),1)/max(float(row["AVG_VOLUME"]),1)),-10,10),
        ]
        for c in ["EMA20","EMA50","EMA200"]:
            vals.append(np.clip(float(row[c])/close-1,-5,5))
        vals.append(np.clip((float(row["RSI"])-50)/50,-1,1))
        for c in ["MACD","MACD_SIGNAL","MACD_HISTOGRAM"]:
            vals.append(np.clip(float(row[c])/close,-1,1))
        for c in ["BB_UPPER","BB_MIDDLE","BB_LOWER"]:
            vals.append(np.clip(float(row[c])/close-1,-5,5))
        vals.append(np.clip(float(row["VWAP"])/close-1,-5,5))
        vals.append(0.0)
        vals.append(np.clip(float(row["ATR"])/close,0,1))
        vals.append(np.clip(float(row["ADX"])/100,0,1))
        vals.append(np.sign(float(row["OBV"])))
        vals.append(np.clip(float(row["STOCH_RSI"])*2-1,-1,1))
        # V4 context
        vals.extend([
            np.clip(float(row["RET_1"]),-1,1),
            np.clip(float(row["RET_5"]),-1,1),
            np.clip(float(row["RET_20"]),-1,1),
            np.clip(float(row["VOL_20"]),0,1),
            np.clip(float(row["EMA20_SLOPE"]),-1,1),
            np.clip(float(row["EMA50_SLOPE"]),-1,1),
            np.clip(float(row["ATR_PCT"]),0,1),
            np.clip(float(row["TREND_SCORE"])*2-1,-1,1),
            float(self.position),
        ])
        x=np.nan_to_num(np.asarray(vals,dtype=np.float32),nan=0,posinf=10,neginf=-10)
        return np.clip(x,-10,10).astype(np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        idx=int(self.rng.integers(0,len(self.tickers)))
        self.current_ticker=self.tickers[idx]
        self.env_df=self.data[self.current_ticker]
        max_start=max(0,len(self.env_df)-EPISODE_LENGTH-2)
        self.current_step=int(self.rng.integers(0,max_start+1)) if max_start>0 else 0
        self.end_step=min(len(self.env_df)-2,self.current_step+EPISODE_LENGTH)
        self.position=0
        self.equity=INITIAL_BALANCE
        self.peak=INITIAL_BALANCE
        self.max_dd=0.0
        self.trades=0
        self.tx_cost=0.0
        self.long_steps=self.short_steps=self.flat_steps=0
        self.total_reward=0.0
        self.transitions=[]
        row=self.env_df.iloc[self.current_step]
        return self._obs(row),{"training_stock":self.current_ticker}

    def step(self,action):
        action=int(np.asarray(action).item())
        target={0:-1.0,1:-0.5,2:0.0,3:0.5,4:1.0}[action]
        row=self.env_df.iloc[self.current_step]
        nxt=self.env_df.iloc[self.current_step+1]
        price=max(float(row["Close"]),1e-8)
        next_price=max(float(nxt["Close"]),1e-8)
        market_return=next_price/price-1
        old=self.position
        change=abs(target-old)
        cost=change*TRANSACTION_COST
        if change: self.trades+=1
        self.position=target
        strategy_return=self.position*market_return
        net_return=float(np.clip(strategy_return-cost,-0.99,10))
        prev_equity=self.equity
        self.equity*=1+net_return
        self.peak=max(self.peak,self.equity)
        dd=self.equity/self.peak-1
        self.max_dd=min(self.max_dd,dd)
        self.tx_cost += cost*prev_equity
        log_ret=float(np.log(max(self.equity,1e-8)/max(prev_equity,1e-8)))
        downside=max(0.0,-net_return)
        drawdown_pressure=abs(min(dd,0.0))
        turnover_penalty=TURNOVER_WEIGHT*change
        missed=0.0
        directional_error=0.0
        atr_pct=max(float(row["ATR"])/price,1e-8)
        move_strength=abs(market_return)/atr_pct
        if abs(self.position) < 1e-8 and move_strength>MISSED_MOVE_ATR:
            missed=MISSED_MOVE_WEIGHT*min(move_strength-MISSED_MOVE_ATR,2.0)
        # Penalize being on the wrong side of a sufficiently large move.
        # This is symmetric: long is penalized for strong negative moves,
        # short is penalized for strong positive moves. It does not force
        # short trades; it only makes persistent one-sided exposure costly.
        if move_strength > MISSED_MOVE_ATR and market_return * self.position < 0:
            directional_error = DIRECTIONAL_ERROR_WEIGHT * min(move_strength-MISSED_MOVE_ATR, 2.0) * abs(self.position)
        reward=log_ret - DOWNSIDE_WEIGHT*(downside**2) - DRAWDOWN_WEIGHT*drawdown_pressure - turnover_penalty - missed - directional_error
        reward=float(np.clip(reward,-10,10))
        self.total_reward+=reward
        if self.position==1:self.long_steps+=1
        elif self.position==-1:self.short_steps+=1
        else:self.flat_steps+=1
        self.current_step+=1
        terminated=self.current_step>=self.end_step
        obs=self._obs(self.env_df.iloc[min(self.current_step,len(self.env_df)-1)])
        info={
            "training_stock":self.current_ticker,"action":action,
            "action_name":{0:"SHORT",1:"HALF_SHORT",2:"FLAT",3:"HALF_LONG",4:"LONG"}[action],
            "position":self.position,"previous_position":old,
            "current_price":price,"next_price":next_price,
            "market_return":float(market_return),
            "strategy_return":float(strategy_return),
            "net_return":float(net_return),"reward":reward,
            "equity":float(self.equity),"drawdown":float(dd),
            "max_drawdown":float(self.max_dd),"total_trades":self.trades,
            "transaction_cost":float(cost),"total_transaction_cost":float(self.tx_cost),
            "missed_opportunity_penalty":float(missed),"directional_error_penalty":float(directional_error),
            "current_step":self.current_step
        }
        return obs,reward,terminated,False,info

    def close(self): pass

def split_data(df):
    n=len(df)
    a=int(n*TRAIN_RATIO)
    b=int(n*(TRAIN_RATIO+VALIDATION_RATIO))
    return df.iloc[:a].copy(),df.iloc[a:b].copy(),df.iloc[b:].copy()

def main():
    print("="*80)
    print("AI TRADING ASSISTANT — UNIVERSAL PPO V6 TRAINING")
    print("="*80)
    print("Stocks:",", ".join(TICKERS))
    print("Timesteps:",f"{TOTAL_TIMESTEPS:,}")
    print("V6 observation size: 30")
    print("V6 reward: risk-aware return + downside + drawdown + turnover + small missed-move penalty")

    training_data={}
    metadata={}
    for ticker in TICKERS:
        print("\n"+"#"*72)
        print("STOCK:",ticker)
        try:
            full,resolved=load_stock(ticker)
            train,val,test=split_data(full)
            if len(train)<MIN_USABLE_ROWS:
                raise ValueError(f"Training rows too small: {len(train)}")
            training_data[ticker]=train
            metadata[ticker]={
                "resolved_symbol":resolved,
                "full_rows":len(full),
                "train_rows":len(train),
                "validation_rows":len(val),
                "test_rows":len(test),
                "train_start":str(train.index[0]),
                "train_end":str(train.index[-1]),
                "validation_start":str(val.index[0]),
                "validation_end":str(val.index[-1]),
                "test_start":str(test.index[0]),
                "test_end":str(test.index[-1]),
            }
            print(f"✓ {ticker}: train={len(train)}, validation={len(val)}, test={len(test)}")
        except Exception as e:
            print(f"✗ {ticker}: {e}")

    if len(training_data)<5:
        raise RuntimeError(f"Only {len(training_data)} stocks available. Need at least 5.")

    env=UniversalV4Env(training_data,seed=SEED)
    print("\n"+"="*80)
    print("ENVIRONMENT CHECK")
    print("="*80)
    obs,info=env.reset(seed=SEED)
    print("Observation:",obs.shape)
    print("Observation space:",env.observation_space)
    print("Action space:",env.action_space)
    assert obs.shape==(30,) and np.isfinite(obs).all()
    for a in [0,1,2,3,4]:
        env.reset(seed=SEED)
        o,r,term,trunc,inf=env.step(a)
        assert o.shape==(30,) and np.isfinite(o).all() and np.isfinite(r)
        print(f"Action {a}: OK | reward={r:.6f}")
    env.close()

    train_env=Monitor(UniversalV4Env(training_data,seed=SEED))
    model=PPO(
        "MlpPolicy",train_env,
        learning_rate=LEARNING_RATE,n_steps=N_STEPS,batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,gamma=GAMMA,gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,ent_coef=ENT_COEF,vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,seed=SEED,verbose=1,
        tensorboard_log=None,device="auto"
    )
    print("\n"+"="*80)
    print("STARTING PPO V4 TRAINING")
    print("="*80)
    model.learn(total_timesteps=TOTAL_TIMESTEPS,callback=TrainingCallback(),progress_bar=True)
    model.save(MODEL_PATH)
    meta={
        "version":"universal_v6","algorithm":"PPO","policy":"MlpPolicy",
        "timesteps":TOTAL_TIMESTEPS,"observation_size":30,
        "actions":{"0":"SHORT","1":"HALF_SHORT","2":"FLAT","3":"HALF_LONG","4":"LONG"},
        "tickers_trained":list(training_data),
        "config":{
            "transaction_cost":TRANSACTION_COST,"episode_length":EPISODE_LENGTH,
            "drawdown_weight":DRAWDOWN_WEIGHT,"downside_weight":DOWNSIDE_WEIGHT,
            "turnover_weight":TURNOVER_WEIGHT,"missed_move_weight":MISSED_MOVE_WEIGHT,
            "missed_move_atr":MISSED_MOVE_ATR,
            "learning_rate":LEARNING_RATE,"n_steps":N_STEPS,"batch_size":BATCH_SIZE,
            "n_epochs":N_EPOCHS,"gamma":GAMMA,"gae_lambda":GAE_LAMBDA,
            "ent_coef":ENT_COEF
        },
        "data":metadata,
        "warning":"Training completion does not imply profitability. Evaluate on validation/test data."
    }
    with open(META_PATH,"w",encoding="utf-8") as f: json.dump(meta,f,indent=2,default=str)
    print("\nMODEL SAVED:",MODEL_PATH+".zip")
    print("METADATA SAVED:",META_PATH)
    print("\nUNIVERSAL PPO V6 TRAINING COMPLETE")
    print("Next step: run rl/evaluate_universal_v6.py")

if __name__=="__main__":
    main()
