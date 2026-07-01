"""
資料來源層 (data_sources.py)
--------------------------------
提供兩種資料來源：
  1. FinMindSource  - 從 FinMind API 抓取真實台股「日線價格」與「三大法人買賣超」。
  2. DemoSource     - 產生模擬資料，讓你不必設定 token 就能先看到整個網站長相。

兩者都回傳統一格式：對每一檔股票回傳一個 pandas.DataFrame，欄位為
    date, open, high, low, close, volume,
    foreign_net (外資淨買超, 單位: 股), trust_net (投信淨買超, 單位: 股)
依日期由舊到新排序。上層的 stock_analyzer.py 只認這個格式，
所以日後要換資料來源，只要再寫一個有 .get_history(code) / .get_name(code) 的類別即可。
"""

from __future__ import annotations
import time
import datetime as dt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. FinMind 真實資料
# ---------------------------------------------------------------------------
class FinMindSource:
    """
    從 FinMind 抓真實資料。
    需要免費 token: https://finmindtrade.com/  註冊後在會員中心取得 api token。
    """
    BASE = "https://api.finmindtrade.com/api/v4/data"

    def __init__(self, token: str = "", lookback_days: int = 180):
        self.token = token
        self.lookback_days = lookback_days
        self._name_map: dict[str, str] | None = None
        try:
            import requests  # 延遲匯入，DemoSource 不需要
            self._requests = requests
        except ImportError as e:
            raise ImportError("使用真實資料需要 requests 套件：pip install requests") from e

    def _get(self, dataset: str, code: str, start: str) -> pd.DataFrame:
        params = {"dataset": dataset, "data_id": code, "start_date": start}
        if self.token:
            params["token"] = self.token
        r = self._requests.get(self.BASE, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        data = payload.get("data", [])
        return pd.DataFrame(data)

    def get_name(self, code: str) -> str:
        if self._name_map is None:
            try:
                df = self._get("TaiwanStockInfo", "", "")
                self._name_map = dict(zip(df["stock_id"].astype(str),
                                          df["stock_name"]))
            except Exception:
                self._name_map = {}
        return self._name_map.get(str(code), code)

    def get_history(self, code: str) -> pd.DataFrame:
        start = (dt.date.today() -
                 dt.timedelta(days=self.lookback_days)).isoformat()

        price = self._get("TaiwanStockPrice", code, start)
        if price.empty:
            raise ValueError(f"FinMind 查無 {code} 的價格資料")
        price = price.rename(columns={
            "date": "date", "open": "open", "max": "high",
            "min": "low", "close": "close", "Trading_Volume": "volume",
        })[["date", "open", "high", "low", "close", "volume"]]

        # 三大法人買賣超：欄位含 name(外資/投信/自營商) 與 buy / sell (單位: 股)
        chips = self._get("TaiwanStockInstitutionalInvestorsBuySell", code, start)
        foreign = pd.Series(0, index=price.index, dtype=float)
        trust = pd.Series(0, index=price.index, dtype=float)
        if not chips.empty:
            chips["net"] = chips["buy"] - chips["sell"]
            # FinMind 的外資名稱可能為 'Foreign_Investor' 或 'Foreign_Dealer_Self'
            f = chips[chips["name"].str.contains("Foreign", case=False, na=False)]
            t = chips[chips["name"].str.contains("Investment_Trust", case=False, na=False)]
            f = f.groupby("date")["net"].sum()
            t = t.groupby("date")["net"].sum()
            price = price.set_index("date")
            price["foreign_net"] = f.reindex(price.index).fillna(0)
            price["trust_net"] = t.reindex(price.index).fillna(0)
            price = price.reset_index()
        else:
            price["foreign_net"] = 0.0
            price["trust_net"] = 0.0

        price = price.sort_values("date").reset_index(drop=True)
        for c in ["open", "high", "low", "close", "volume",
                  "foreign_net", "trust_net"]:
            price[c] = pd.to_numeric(price[c], errors="coerce")
        time.sleep(0.3)  # 對 API 友善，避免觸發速率限制
        return price.dropna(subset=["close"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. 模擬資料 (Demo)
# ---------------------------------------------------------------------------
class DemoSource:
    """
    產生 ~150 個交易日的模擬資料。
    重點：價格的「漲跌」會跟法人買賣超相關，這樣控盤分析才會跑出有意義的結論。
    每檔股票會被指定一種「劇情」(外資控盤上攻 / 投信控盤上攻 / 外資調節下殺 ...)，
    方便你檢視儀表板在各種情況下的呈現。
    """
    SCENARIOS = {
        "2330": ("台積電", "foreign_up"),     # 外資買上去
        "2454": ("聯發科", "trust_up"),       # 投信買上去
        "2603": ("長榮",   "foreign_down"),   # 外資賣下來
        "2317": ("鴻海",   "trust_down"),     # 投信賣下來
        "3008": ("大立光", "both_up"),        # 外資投信同買
        "2412": ("中華電", "choppy"),         # 盤整、法人influence小
    }

    def __init__(self, lookback_days: int = 150, seed: int = 7):
        self.n = lookback_days
        self.rng = np.random.default_rng(seed)

    def get_name(self, code: str) -> str:
        return self.SCENARIOS.get(code, (code, ""))[0]

    def get_history(self, code: str) -> pd.DataFrame:
        name, scenario = self.SCENARIOS.get(code, (code, "choppy"))
        n = self.n
        rng = self.rng

        # 法人每日淨買超 (張) — 依劇情塑造趨勢
        f = rng.normal(0, 600, n)   # 外資
        t = rng.normal(0, 350, n)   # 投信
        ramp = np.linspace(0, 1, n)
        if scenario == "foreign_up":
            f += 700 * ramp + 350
            t += rng.normal(0, 180, n)
        elif scenario == "trust_up":
            t += 600 * ramp + 300
            f += rng.normal(0, 250, n)
        elif scenario == "foreign_down":
            f -= 800 * ramp + 300
            t += rng.normal(0, 180, n)
        elif scenario == "trust_down":
            t -= 550 * ramp + 250
            f += rng.normal(0, 250, n)
        elif scenario == "both_up":
            f += 450 * ramp + 200
            t += 380 * ramp + 180
        # choppy: 維持隨機

        # 價格報酬率 = 法人買盤帶動 + 雜訊 (法人買 -> 漲)；係數調溫和，日波動約 ±1%
        base_ret = (f / 250000.0) + (t / 180000.0) + rng.normal(0, 0.010, n)
        base_ret = np.clip(base_ret, -0.085, 0.085)   # 模擬漲跌停限制
        start_price = float(rng.integers(40, 350))
        close = start_price * np.cumprod(1 + base_ret)

        # 由收盤反推開高低量
        prev = np.concatenate([[start_price], close[:-1]])
        intraday = np.abs(rng.normal(0, 0.012, n)) + 0.004
        high = np.maximum(close, prev) * (1 + intraday)
        low = np.minimum(close, prev) * (1 - intraday)
        open_ = prev * (1 + rng.normal(0, 0.005, n))
        open_ = np.clip(open_, low, high)
        volume = (np.abs(f) + np.abs(t) + rng.uniform(3000, 9000, n)) * 1000

        # 交易日 (跳過週末)
        dates, d = [], dt.date.today() - dt.timedelta(days=int(n * 1.5))
        while len(dates) < n:
            if d.weekday() < 5:
                dates.append(d.isoformat())
            d += dt.timedelta(days=1)
        dates = dates[:n]

        return pd.DataFrame({
            "date": dates,
            "open": np.round(open_, 2),
            "high": np.round(high, 2),
            "low": np.round(low, 2),
            "close": np.round(close, 2),
            "volume": np.round(volume).astype("int64"),
            "foreign_net": np.round(f * 1000),   # 張 -> 股，跟 FinMind 單位一致
            "trust_net": np.round(t * 1000),
        })


def get_source(live: bool, token: str = ""):
    return FinMindSource(token=token) if live else DemoSource()
