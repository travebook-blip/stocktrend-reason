"""
分析引擎 (stock_analyzer.py)
--------------------------------
讀取觀察清單 (watchlist.txt) -> 從資料來源取得每檔股票歷史 ->
計算所有技術指標與籌碼控盤研判 -> 輸出 data.json 給前端 index.html 使用。

技術指標 (台股慣例)
  - 三線多排    : MA5 > MA10 > MA20 (多頭排列)
  - KDJ        : 9 日；判斷 K>D、3 日內 KD 金叉/死叉
  - MACD       : 12/26/9；紅綠柱、柱狀擴大或縮小、3 日內 DIF/MACD 金叉/死叉
  - 月線扣抵    : 20MA 即將被扣掉的價 vs 目前均價 -> 扣低(易漲) / 扣高(易跌)

籌碼 (法人)
  - 外資 / 投信 今日、近 5 日、近 20 日淨買賣張數與連續買賣天數
  - 控盤研判    : 用「法人淨買超」與「當日漲跌」的關係，推論
                 (a) 是外資還是投信主導股價
                 (b) 上漲是誰買上去、下跌是誰賣下來

使用方式
  python stock_analyzer.py                 # 模擬資料 (免設定，先看成品)
  python stock_analyzer.py --live          # 真實資料 (需 FinMind token)
      --token XXXX  或設環境變數 FINMIND_TOKEN
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import datetime as dt
import numpy as np
import pandas as pd

from data_sources import get_source

LOT = 1000  # 1 張 = 1000 股


# ===========================================================================
# 技術指標
# ===========================================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    df["ma5"] = c.rolling(5).mean()
    df["ma10"] = c.rolling(10).mean()
    df["ma20"] = c.rolling(20).mean()
    df["ma60"] = c.rolling(60).mean()

    # --- KDJ (9 日) ---
    n = 9
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = (c - low_n) / (high_n - low_n) * 100
    rsv = rsv.replace([np.inf, -np.inf], np.nan).fillna(50)
    k_vals, d_vals, k_prev, d_prev = [], [], 50.0, 50.0
    for r in rsv:
        k_prev = k_prev * 2 / 3 + r / 3
        d_prev = d_prev * 2 / 3 + k_prev / 3
        k_vals.append(k_prev)
        d_vals.append(d_prev)
    df["K"] = k_vals
    df["D"] = d_vals
    df["J"] = 3 * df["K"] - 2 * df["D"]

    # --- MACD (12/26/9) ---
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()   # 訊號線 (MACD)
    df["HIST"] = df["DIF"] - df["DEA"]                        # 柱狀體 (OSC)
    return df


def _recent_cross(fast: pd.Series, slow: pd.Series, within: int = 3):
    """
    偵測 fast 線是否在最近 within 個交易日內穿越 slow 線。
    回傳 (kind, days_ago)：kind 為 'golden'(金叉)/'death'(死叉)/None；
    days_ago=0 表示今天發生。取最近一次。
    """
    diff = (fast - slow).dropna()
    if len(diff) < 2:
        return None, None
    sign = np.sign(diff.values)
    for ago in range(0, within):
        i = len(sign) - 1 - ago
        if i - 1 < 0:
            break
        prev, cur = sign[i - 1], sign[i]
        if prev < 0 and cur > 0:
            return "golden", ago
        if prev > 0 and cur < 0:
            return "death", ago
    return None, None


def ma20_deduction(df: pd.DataFrame, look_ahead: int = 5) -> dict:
    """
    月線(20MA)扣抵分析。
    明日的 20MA = 丟掉「20 天前的收盤(扣抵值)」再加上明日收盤。
    若扣抵值 < 目前 20MA -> 丟掉它會把均價墊高 -> 扣低 (月線易上揚)。
    同時看未來 look_ahead 天的扣抵值，估計月線走向。
    """
    closes = df["close"].dropna().values
    if len(closes) < 21:
        return {"available": False}
    ma20 = float(np.mean(closes[-20:]))
    deduct_today = float(closes[-20])          # 下一個交易日要被扣掉的價
    is_low = deduct_today < ma20

    upcoming = closes[-20: -20 + look_ahead]    # 未來數日的扣抵值
    low_cnt = int(np.sum(upcoming < ma20))
    if low_cnt >= look_ahead - 1:
        trend = "月線扣低，未來數日易上揚"
    elif low_cnt <= 1:
        trend = "月線扣高，未來數日有下彎壓力"
    else:
        trend = "扣抵值高低交錯，月線方向待確認"

    return {
        "available": True,
        "ma20": round(ma20, 2),
        "deduct_value": round(deduct_today, 2),
        "type": "扣低" if is_low else "扣高",
        "bullish": bool(is_low),
        "implication": "月線易上揚" if is_low else "月線易下彎",
        "trend": trend,
    }


def technical_block(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    ma_bullish = bool(last["ma5"] > last["ma10"] > last["ma20"]) \
        if not pd.isna(last["ma20"]) else False

    kd_kind, kd_ago = _recent_cross(df["K"], df["D"], within=3)
    macd_kind, macd_ago = _recent_cross(df["DIF"], df["DEA"], within=3)

    hist = df["HIST"].dropna().values
    hist_now = float(hist[-1]) if len(hist) else 0.0
    hist_prev = float(hist[-2]) if len(hist) > 1 else 0.0
    color = "red" if hist_now >= 0 else "green"          # 紅柱(多) / 綠柱(空)
    if color == "red":
        bar_trend = "紅柱擴大" if hist_now > hist_prev else "紅柱縮短"
    else:
        bar_trend = "綠柱擴大" if hist_now < hist_prev else "綠柱縮短"

    return {
        "ma_bullish": ma_bullish,
        "ma": {k: (round(float(last[k]), 2) if not pd.isna(last[k]) else None)
               for k in ["ma5", "ma10", "ma20", "ma60"]},
        "kd": {
            "k": round(float(last["K"]), 1),
            "d": round(float(last["D"]), 1),
            "j": round(float(last["J"]), 1),
            "k_gt_d": bool(last["K"] > last["D"]),
            "cross": kd_kind,
            "cross_days_ago": kd_ago,
        },
        "macd": {
            "dif": round(float(last["DIF"]), 3),
            "dea": round(float(last["DEA"]), 3),
            "hist": round(hist_now, 3),
            "color": color,
            "bar_trend": bar_trend,
            "cross": macd_kind,
            "cross_days_ago": macd_ago,
        },
        "ma20_deduction": ma20_deduction(df),
    }


# ===========================================================================
# 籌碼 / 控盤研判
# ===========================================================================
def _consecutive(series: pd.Series) -> int:
    """從最後一天往回算連續同方向(買或賣)的天數，正數=連買，負數=連賣。"""
    vals = series.values
    if len(vals) == 0 or vals[-1] == 0:
        return 0
    sign = 1 if vals[-1] > 0 else -1
    cnt = 0
    for v in vals[::-1]:
        if (v > 0 and sign > 0) or (v < 0 and sign < 0):
            cnt += 1
        else:
            break
    return cnt * sign


def chips_block(df: pd.DataFrame) -> dict:
    # 股 -> 張
    foreign = (df["foreign_net"] / LOT).round().astype(int)
    trust = (df["trust_net"] / LOT).round().astype(int)
    ret = df["close"].pct_change().fillna(0)

    def summary(s):
        return {
            "today": int(s.iloc[-1]),
            "d5": int(s.tail(5).sum()),
            "d20": int(s.tail(20).sum()),
            "consecutive": _consecutive(s),
        }

    # ---- 控盤研判 (取近 20 個交易日) ----
    win = 20
    f = foreign.tail(win).reset_index(drop=True).astype(float)
    t = trust.tail(win).reset_index(drop=True).astype(float)
    r = ret.tail(win).reset_index(drop=True)

    def safe_corr(a, b):
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    f_corr = safe_corr(f, r)   # 外資買賣 與 漲跌 的相關
    t_corr = safe_corr(t, r)   # 投信買賣 與 漲跌 的相關

    up = r > 0
    down = r < 0
    f_up, t_up = float(f[up].sum()), float(t[up].sum())       # 上漲日法人合計
    f_down, t_down = float(f[down].sum()), float(t[down].sum())  # 下跌日法人合計

    period_ret = float((df["close"].iloc[-1] / df["close"].iloc[-win] - 1) * 100) \
        if len(df) > win else 0.0

    up_driver = "外資" if f_up >= t_up else "投信"
    down_driver = "外資" if f_down <= t_down else "投信"  # 賣超越多(越負)者主導下跌

    # 控盤研判：以「近20日累積淨額且與股價趨勢同向」為主，相關性為信心加權
    f_net20, t_net20 = float(f.sum()), float(t.sum())
    trend_sign = 1.0 if period_ret >= 0 else -1.0
    f_pos = max(f_net20 * trend_sign, 0.0) * (0.5 + abs(f_corr))
    t_pos = max(t_net20 * trend_sign, 0.0) * (0.5 + abs(t_corr))
    total_force = f_pos + t_pos
    if total_force <= 1e-9:
        controller, balance = "中性", 0.0
    else:
        balance = round((t_pos - f_pos) / total_force, 2)
        controller = "中性" if abs(balance) < 0.15 else ("外資" if f_pos > t_pos else "投信")

    # 產生一句白話研判
    if controller == "中性":
        verdict = "近 20 日法人influence不明顯，股價較偏籌碼以外因素或散戶。"
    else:
        main = controller
        if period_ret >= 0:
            verdict = (f"近 20 日股價漲約 {period_ret:.1f}%，"
                       f"主要由{up_driver}在上漲日加碼推升；研判由「{main}」主導控盤。")
        else:
            verdict = (f"近 20 日股價跌約 {period_ret:.1f}%，"
                       f"主要由{down_driver}在下跌日調節；研判由「{main}」主導控盤。")

    return {
        "foreign": summary(foreign),
        "trust": summary(trust),
        "control": {
            "controller": controller,
            "balance": balance,
            "up_driver": up_driver,
            "down_driver": down_driver,
            "foreign_corr": round(f_corr, 2),
            "trust_corr": round(t_corr, 2),
            "period_return": round(period_ret, 1),
            "verdict": verdict,
        },
    }


# ===========================================================================
# 主流程
# ===========================================================================
def analyze_stock(code: str, source) -> dict | None:
    try:
        df = source.get_history(code)
    except Exception as e:
        print(f"  ! {code} 取得資料失敗：{e}", file=sys.stderr)
        return None
    if df is None or len(df) < 30:
        print(f"  ! {code} 資料不足，略過", file=sys.stderr)
        return None

    df = add_indicators(df)
    last = df.iloc[-1]
    prev_close = df["close"].iloc[-2]
    change = float(last["close"] - prev_close)

    return {
        "code": code,
        "name": source.get_name(code),
        "date": str(last["date"]),
        "price": {
            "close": round(float(last["close"]), 2),
            "change": round(change, 2),
            "change_pct": round(change / prev_close * 100, 2),
            "volume_lots": int(last["volume"] / LOT),
        },
        "technical": technical_block(df),
        "chips": chips_block(df),
    }


def load_watchlist(path: str) -> list[str]:
    if not os.path.exists(path):
        # 預設清單 (對應 DemoSource 的劇情)
        return ["2330", "2454", "2603", "2317", "3008", "2412"]
    codes = []
    for line in open(path, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if line:
            codes.append(line.split()[0])
    return codes


def main():
    ap = argparse.ArgumentParser(description="台股技術+籌碼分析，輸出 data.json")
    ap.add_argument("--live", action="store_true", help="使用 FinMind 真實資料")
    ap.add_argument("--token", default=os.environ.get("FINMIND_TOKEN", ""),
                    help="FinMind API token (或設環境變數 FINMIND_TOKEN)")
    ap.add_argument("--watchlist", default="watchlist.txt")
    ap.add_argument("--out", default="data.json")
    args = ap.parse_args()

    source = get_source(args.live, args.token)
    codes = load_watchlist(args.watchlist)
    mode = "真實資料 (FinMind)" if args.live else "模擬資料 (Demo)"
    print(f"模式：{mode}　|　共 {len(codes)} 檔：{', '.join(codes)}")

    results = []
    for code in codes:
        print(f"分析 {code} ...")
        r = analyze_stock(code, source)
        if r:
            results.append(r)

    out = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mode": "demo" if not args.live else "live",
        "stocks": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # 同時輸出 data.js，讓 index.html 直接雙擊開啟也能讀到資料 (免架伺服器)
    js_path = os.path.splitext(args.out)[0] + ".js"
    with open(js_path, "w", encoding="utf-8") as f:
        f.write("window.__STOCK_DATA__ = ")
        json.dump(out, f, ensure_ascii=False)
        f.write(";")
    print(f"完成 -> {args.out} / {js_path} ({len(results)} 檔)")


if __name__ == "__main__":
    main()
