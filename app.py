from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# ==========================================
# 核心引擎：纯 Pandas 计算技术指标
# (无需 numba, 无需 pandas_ta，兼容所有 Python 版本)
# ==========================================
def calculate_indicators(df):
    # 1. 计算均线 (SMA)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_150'] = df['Close'].rolling(window=150).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # 2. 计算成交量均线
    df['Vol_SMA'] = df['Volume'].rolling(window=20).mean()

    # 3. 计算 MACD (12, 26, 9)
    # EMA 快速与慢速
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    
    df['MACD_Line'] = ema12 - ema26
    df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Line'] - df['Signal_Line'] # 能量柱
    
    # 4. 计算 RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# ==========================================
# 策略逻辑：筛选最强股票
# ==========================================
def analyze_stock(ticker):
    try:
        # 下载数据 (修复 yfinance 新版格式问题)
        df = yf.download(ticker, period="1y", interval="1d", progress=False, multi_level_index=False)
        
        # 过滤刚上市或数据不足的股票
        if df.empty or len(df) < 200: 
            return None
            
        # 计算所有指标
        df = calculate_indicators(df)
        
        # 获取最新一行(curr) 和 前一天(prev)
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(curr['Close'])
        
        # --- 筛选条件 START ---
        
        # 1. 趋势过滤 (Trend Template)
        # 股价在 50日线上，且 50 > 150 > 200 (最强多头排列)
        trend_strong = (price > curr['SMA_50'] > curr['SMA_150'] > curr['SMA_200'])
        
        # 2. MACD 动能
        # 必须在零轴上方 (多头市场)
        macd_above_zero = curr['MACD_Line'] > 0
        # 触发信号：刚刚金叉 OR 能量柱向上放大(加速)
        macd_trigger = (prev['MACD_Hist'] < 0 and curr['MACD_Hist'] > 0) or \
                       (curr['MACD_Hist'] > 0 and curr['MACD_Hist'] > prev['MACD_Hist'])
        
        # 3. RSI 强度
        # 大于50说明强，小于85防止山顶站岗
        rsi_check = 50 < curr['RSI'] < 85
        
        # 4. 成交量 (Volume)
        # 今日成交量 > 20日均量 * 1.2倍
        vol_ratio = float(curr['Volume']) / float(curr['Vol_SMA']) if curr['Vol_SMA'] > 0 else 0
        vol_check = vol_ratio > 1.2
        
        # --- 筛选条件 END ---
        
        # 只有当所有条件都满足时，才返回数据
        if trend_strong and macd_above_zero and macd_trigger and rsi_check and vol_check:
            return {
                'symbol': ticker,
                'price': round(price, 2),
                'change': round(((price - float(prev['Close'])) / float(prev['Close'])) * 100, 2),
                'vol_ratio': round(vol_ratio, 2),
                'rsi': round(float(curr['RSI']), 2),
                'score': round(float(curr['RSI']) + vol_ratio * 10, 0) # 综合评分
            }
            
    except Exception:
        return None
    return None

# ==========================================
# 路由设置
# ==========================================
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/scan', methods=['POST'])
def run_scan():
    # 这里定义你想扫描的股票池。
    # 想要全市场？你需要把 6000 个代码放进这个列表 (建议分批扫)
    # 下面是精选的 "纳斯达克100 + 热门动能股" 列表，作为演示
    tickers = [
        "NVDA", "AMD", "TSLA", "MSFT", "AAPL", "META", "AMZN", "GOOGL", "NFLX", 
        "PLTR", "COIN", "MARA", "MSTR", "SMCI", "ARM", "AVGO", "TSM", "ORCL",
        "PANW", "CRWD", "UBER", "ABNB", "SHOP", "SQ", "DKNG", "HOOD", "AFRM",
        "UPST", "CVNA", "RIVN", "LCID", "SOFI", "AI", "PATH", "U", "NET"
    ]
    
    results = []
    print(f"正在启动多线程扫描 {len(tickers)} 只股票...")
    
    # 开启 10 个线程并行处理
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_stock, t): t for t in tickers}
        for future in futures:
            res = future.result()
            if res:
                results.append(res)
    
    # 按评分从高到低排序
    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=8888)