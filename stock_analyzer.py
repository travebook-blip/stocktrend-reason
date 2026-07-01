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


def _safe_corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _hit_rate(x: np.ndarray, r: np.ndarray):
    """
    方向命中率：在該法人「有買或有賣」的那些天裡，
    股價漲跌方向跟他一致的比例。跟成交量大小完全無關，
    只問「這個法人一動作，股價通常跟不跟」。
    回傳 (rate, n_valid)；資料不足回傳 (None, 0)。
    """
    x_sign = np.sign(x)
    r_sign = np.sign(r)
    mask = (x_sign != 0) & (r_sign != 0)
    n = int(mask.sum())
    if n == 0:
        return None, 0
    hits = int(((x_sign == r_sign) & mask).sum())
    return hits / n, n


def _compute_control(df: pd.DataFrame, foreign: pd.Series, trust: pd.Series,
                      ret: pd.Series, win: int) -> dict:
    """
    對指定天數(win)的區間，計算法人控盤研判。
    核心邏輯改為「方向命中率」而非「買賣張數」：誰進出時股價比較會跟，
    誰就是控盤者——即使他買賣的量比較小。
    另外特別抓出「雙方方向相反」的那些天，股價實際跟誰，作為最直接的證據。
    """
    MIN_N = 3          # 命中率至少要幾個有效樣本才採信
    WIN_MARGIN = 0.15  # 命中率差距、或偏離50%多少才判定有主導方（不是純巧合）

    n = min(win, len(df) - 1)  # -1 因為 period_return 需要 win 天前的收盤
    if n < 5:
        return {"available": False, "window": win}

    f = foreign.tail(n).reset_index(drop=True).astype(float).values
    t = trust.tail(n).reset_index(drop=True).astype(float).values
    r = ret.tail(n).reset_index(drop=True).values

    f_corr = _safe_corr(pd.Series(f), pd.Series(r))
    t_corr = _safe_corr(pd.Series(t), pd.Series(r))

    up, down = r > 0, r < 0
    f_up, t_up = float(f[up].sum()), float(t[up].sum())
    f_down, t_down = float(f[down].sum()), float(t[down].sum())
    up_driver = "外資" if f_up >= t_up else "投信"
    down_driver = "外資" if f_down <= t_down else "投信"

    period_ret = float((df["close"].iloc[-1] / df["close"].iloc[-n] - 1) * 100)

    # ---- 方向命中率（核心：跟買賣量無關）----
    f_rate, f_n = _hit_rate(f, r)
    t_rate, t_n = _hit_rate(t, r)
    f_ok, t_ok = f_n >= MIN_N, t_n >= MIN_N

    # ---- 雙方方向相反的那些天，股價實際跟誰（最直接的證據）----
    f_sign, t_sign, r_sign = np.sign(f), np.sign(t), np.sign(r)
    conflict = (f_sign != 0) & (t_sign != 0) & (f_sign != t_sign) & (r_sign != 0)
    conflict_n = int(conflict.sum())
    trust_win_conflict = int(((r_sign == t_sign) & conflict).sum())
    foreign_win_conflict = int(((r_sign == f_sign) & conflict).sum())
    conflict_trust_rate = round(trust_win_conflict / conflict_n, 2) if conflict_n > 0 else None

    method = "insufficient"
    if f_ok and t_ok:
        balance = round(t_rate - f_rate, 2)
        method = "both"
    elif t_ok and not f_ok:
        balance = round((t_rate - 0.5) * 2, 2) if t_rate >= 0.5 else 0.0
        method = "trust_only"
    elif f_ok and not t_ok:
        balance = round((0.5 - f_rate) * 2, 2) if f_rate >= 0.5 else 0.0
        method = "foreign_only"
    else:
        balance = 0.0

    controller = "中性" if abs(balance) < WIN_MARGIN else ("外資" if balance < 0 else "投信")

    # ---- 白話研判：優先用「方向衝突日」講清楚（最貼近使用者想問的事）----
    conflict_note = ""
    if conflict_n >= 2 and conflict_trust_rate is not None:
        winner = "投信" if conflict_trust_rate > 0.5 else ("外資" if conflict_trust_rate < 0.5 else None)
        if winner:
            pct = round((conflict_trust_rate if winner == "投信" else 1 - conflict_trust_rate) * 100)
            conflict_note = (f"其中有 {conflict_n} 天外資與投信方向相反，"
                              f"股價有 {pct}% 的天數跟隨「{winner}」（即使另一方同時反向操作）。")

    if method == "both":
        base = f"近 {n} 日外資方向命中率 {round(f_rate*100)}%（{f_n}天有動作），投信方向命中率 {round(t_rate*100)}%（{t_n}天有動作）。"
    elif method == "trust_only":
        base = f"近 {n} 日投信雖僅 {t_n} 天有動作，但方向命中率達 {round(t_rate*100)}%；外資動作天數不足以判斷。"
    elif method == "foreign_only":
        base = f"近 {n} 日外資方向命中率 {round(f_rate*100)}%（{f_n}天有動作）；投信動作天數不足以判斷。"
    else:
        base = f"近 {n} 日兩方買賣動作都太少，無法判斷方向命中率。"

    if controller == "中性":
        verdict = base + "　雙方對股價方向的影響力接近，較偏籌碼以外因素或散戶。" + conflict_note
    else:
        verdict = base + f"　研判由「{controller}」主導股價方向（不代表買賣量較大）。" + conflict_note

    return {
        "available": True, "window": n,
        "controller": controller, "balance": balance,
        "up_driver": up_driver, "down_driver": down_driver,
        "foreign_corr": round(f_corr, 2), "trust_corr": round(t_corr, 2),
        "foreign_hit_rate": round(f_rate, 2) if f_rate is not None else None,
        "foreign_hit_n": f_n,
        "trust_hit_rate": round(t_rate, 2) if t_rate is not None else None,
        "trust_hit_n": t_n,
        "conflict_days": conflict_n,
        "conflict_trust_rate": conflict_trust_rate,
        "method": method,
        "period_return": round(period_ret, 1), "verdict": verdict,
    }


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
            "d60": int(s.tail(60).sum()),
            "consecutive": _consecutive(s),
        }

    # 短、中兩種區間各自研判，不同時間尺度可能給出不同結論（例如短線中性、波段仍是某方主導）
    windows = {"20": _compute_control(df, foreign, trust, ret, 20),
               "60": _compute_control(df, foreign, trust, ret, 60)}

    return {
        "foreign": summary(foreign),
        "trust": summary(trust),
        "windows": windows,
        "control": windows["20"],   # 向下相容：預設仍以近20日為主要欄位
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
