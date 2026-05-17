"""
strategy_manager.py
══════════════════════════════════════════════════════════════
5 стратегии: от агресивен скалп до безопасна балансирана.
Всяка включва:
  • Индикаторни параметри
  • Time-based filtering (часови прозорци)
  • Ръчно редактируеми настройки
══════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, time as dtime
import pytz

import pandas as pd
import pandas_ta as ta
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  TIME FILTER
# ─────────────────────────────────────────────────────────────

@dataclass
class TimeFilter:
    """
    Разрешава търговия само в определени часови прозорци.
    Всички часове са в UTC.
    """
    enabled: bool = True
    sessions: list = field(default_factory=lambda: [
        {"name": "London",    "start": "07:00", "end": "16:00"},
        {"name": "New York",  "start": "13:00", "end": "21:00"},
    ])
    # Ръчно зададен прозорец (ако не се ползват сесиите)
    custom_enabled: bool = False
    custom_start: str = "08:00"   # HH:MM UTC
    custom_end: str   = "20:00"

    def is_allowed(self, dt: Optional[datetime] = None) -> tuple[bool, str]:
        """
        Проверява дали е разрешена търговия в момента.
        Връща (allowed: bool, reason: str)
        """
        if not self.enabled:
            return True, "Филтърът е изключен"

        now = dt or datetime.utcnow()
        current = now.strftime("%H:%M")

        if self.custom_enabled:
            allowed = self.custom_start <= current <= self.custom_end
            reason  = f"Ръчен прозорец {self.custom_start}–{self.custom_end} UTC"
            return allowed, reason

        for session in self.sessions:
            if session["start"] <= current <= session["end"]:
                return True, f"Сесия {session['name']} активна"

        active = [s["name"] for s in self.sessions]
        return False, f"Извън сесиите: {', '.join(active)}"

    def add_session(self, name: str, start: str, end: str) -> None:
        self.sessions.append({"name": name, "start": start, "end": end})

    def remove_session(self, name: str) -> None:
        self.sessions = [s for s in self.sessions if s["name"] != name]


# ─────────────────────────────────────────────────────────────
#  STRATEGY BASE
# ─────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    """Шаблон за стратегия — редактируем от потребителя."""
    name: str = "Unnamed"
    description: str = ""
    style: str = "balanced"   # aggressive | scalp | trend | balanced | safe
    timeframes: list = field(default_factory=lambda: ["15m", "1h"])
    symbol: str = "BTC/USDT"

    # RSI
    rsi_period: int = 14
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0

    # EMA
    ema_fast: int = 9
    ema_medium: int = 21
    ema_slow: int = 50

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # ATR
    atr_period: int = 14

    # Stochastic
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_smooth: int = 3

    # Volume
    vol_ma_period: int = 20
    vol_multiplier: float = 1.3   # Мин. обем = avg × multiplier

    # Confidence threshold (0–100)
    min_confidence: float = 70.0

    # Time filter
    time_filter: TimeFilter = field(default_factory=TimeFilter)

    # Custom indicator code (paste-able от UI)
    custom_indicator_code: str = ""


# ─────────────────────────────────────────────────────────────
#  5 СТРАТЕГИИ
# ─────────────────────────────────────────────────────────────

def _tf_london_ny() -> TimeFilter:
    return TimeFilter(enabled=True, sessions=[
        {"name": "London",   "start": "07:00", "end": "12:00"},
        {"name": "New York", "start": "13:00", "end": "21:00"},
    ])

def _tf_ny_only() -> TimeFilter:
    return TimeFilter(enabled=True, sessions=[
        {"name": "New York", "start": "14:30", "end": "21:00"},
    ])

def _tf_asian() -> TimeFilter:
    return TimeFilter(enabled=True, sessions=[
        {"name": "Asian",    "start": "00:00", "end": "08:00"},
        {"name": "London",   "start": "07:00", "end": "16:00"},
    ])

def _tf_all_day() -> TimeFilter:
    return TimeFilter(enabled=False)


STRATEGY_TEMPLATES: dict[str, StrategyConfig] = {

    # ── 1. QUANTUM SCALPER ──────────────────────────────────
    "Quantum Scalper": StrategyConfig(
        name        = "Quantum Scalper",
        description = "Агресивен скалп. Кратко задържане, много сделки, малки TP/SL.",
        style       = "scalp",
        timeframes  = ["1m", "5m"],
        rsi_period      = 7,
        rsi_oversold    = 30.0,
        rsi_overbought  = 70.0,
        ema_fast        = 5,
        ema_medium      = 13,
        ema_slow        = 34,
        macd_fast       = 8,
        macd_slow       = 17,
        macd_signal     = 9,
        bb_period       = 14,
        bb_std          = 1.8,
        atr_period      = 7,
        stoch_k         = 5,
        stoch_d         = 3,
        stoch_smooth    = 3,
        vol_multiplier  = 1.5,
        min_confidence  = 65.0,
        time_filter     = _tf_ny_only(),
    ),

    # ── 2. NEXUS SMC ELITE ──────────────────────────────────
    "NEXUS SMC Elite": StrategyConfig(
        name        = "NEXUS SMC Elite",
        description = "Smart Money Concepts: order blocks, liquidity, BOS/CHoCH.",
        style       = "aggressive",
        timeframes  = ["15m", "1h"],
        rsi_period      = 14,
        rsi_oversold    = 35.0,
        rsi_overbought  = 65.0,
        ema_fast        = 9,
        ema_medium      = 21,
        ema_slow        = 50,
        macd_fast       = 12,
        macd_slow       = 26,
        macd_signal     = 9,
        bb_period       = 20,
        bb_std          = 2.0,
        atr_period      = 14,
        stoch_k         = 14,
        stoch_d         = 3,
        stoch_smooth    = 3,
        vol_multiplier  = 1.4,
        min_confidence  = 72.0,
        time_filter     = _tf_london_ny(),
    ),

    # ── 3. TITAN TREND ──────────────────────────────────────
    "Titan Trend": StrategyConfig(
        name        = "Titan Trend",
        description = "Следва основния тренд. Triple EMA + MACD + Volume.",
        style       = "trend",
        timeframes  = ["1h", "4h"],
        rsi_period      = 14,
        rsi_oversold    = 40.0,
        rsi_overbought  = 60.0,
        ema_fast        = 21,
        ema_medium      = 50,
        ema_slow        = 200,
        macd_fast       = 12,
        macd_slow       = 26,
        macd_signal     = 9,
        bb_period       = 20,
        bb_std          = 2.0,
        atr_period      = 14,
        stoch_k         = 14,
        stoch_d         = 3,
        stoch_smooth    = 3,
        vol_multiplier  = 1.2,
        min_confidence  = 75.0,
        time_filter     = _tf_london_ny(),
    ),

    # ── 4. PHOENIX REVERSAL ─────────────────────────────────
    "Phoenix Reversal": StrategyConfig(
        name        = "Phoenix Reversal",
        description = "Улавя обрати при изчерпване. RSI extreme + BB squeeze.",
        style       = "balanced",
        timeframes  = ["30m", "1h"],
        rsi_period      = 14,
        rsi_oversold    = 25.0,
        rsi_overbought  = 75.0,
        ema_fast        = 9,
        ema_medium      = 21,
        ema_slow        = 55,
        macd_fast       = 12,
        macd_slow       = 26,
        macd_signal     = 9,
        bb_period       = 20,
        bb_std          = 2.5,
        atr_period      = 14,
        stoch_k         = 14,
        stoch_d         = 3,
        stoch_smooth    = 3,
        vol_multiplier  = 1.3,
        min_confidence  = 73.0,
        time_filter     = _tf_asian(),
    ),

    # ── 5. IRON SHIELD ──────────────────────────────────────
    "Iron Shield": StrategyConfig(
        name        = "Iron Shield",
        description = "Балансирана и безопасна. Малко сделки, висока точност.",
        style       = "safe",
        timeframes  = ["4h", "1d"],
        rsi_period      = 21,
        rsi_oversold    = 30.0,
        rsi_overbought  = 70.0,
        ema_fast        = 20,
        ema_medium      = 50,
        ema_slow        = 200,
        macd_fast       = 12,
        macd_slow       = 26,
        macd_signal     = 9,
        bb_period       = 20,
        bb_std          = 2.0,
        atr_period      = 21,
        stoch_k         = 21,
        stoch_d         = 7,
        stoch_smooth    = 7,
        vol_multiplier  = 1.5,
        min_confidence  = 80.0,
        time_filter     = _tf_all_day(),
    ),
}


# ─────────────────────────────────────────────────────────────
#  INDICATOR ENGINE
# ─────────────────────────────────────────────────────────────

class IndicatorEngine:
    """
    Изчислява всички технически индикатори върху OHLCV DataFrame.
    Поддържа и изпълнение на custom_indicator_code от потребителя.
    """

    def compute(self, df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
        """
        Приема df с колони: open, high, low, close, volume
        Добавя индикаторни колони и връща обогатения df.
        """
        if len(df) < 50:
            return df

        c  = df["close"]
        h  = df["high"]
        lo = df["low"]
        v  = df["volume"]

        # RSI
        df["rsi"] = ta.rsi(c, length=cfg.rsi_period)

        # EMA
        df["ema_fast"]   = ta.ema(c, length=cfg.ema_fast)
        df["ema_medium"] = ta.ema(c, length=cfg.ema_medium)
        df["ema_slow"]   = ta.ema(c, length=cfg.ema_slow)

        # MACD
        macd = ta.macd(c, fast=cfg.macd_fast, slow=cfg.macd_slow, signal=cfg.macd_signal)
        if macd is not None:
            df["macd"]        = macd.iloc[:, 0]
            df["macd_signal"] = macd.iloc[:, 2]
            df["macd_hist"]   = macd.iloc[:, 1]

        # Bollinger Bands
        bb = ta.bbands(c, length=cfg.bb_period, std=cfg.bb_std)
        if bb is not None:
            df["bb_upper"] = bb.iloc[:, 0]
            df["bb_mid"]   = bb.iloc[:, 1]
            df["bb_lower"] = bb.iloc[:, 2]
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

        # ATR
        df["atr"] = ta.atr(h, lo, c, length=cfg.atr_period)

        # Stochastic
        stoch = ta.stoch(h, lo, c, k=cfg.stoch_k, d=cfg.stoch_d, smooth_k=cfg.stoch_smooth)
        if stoch is not None:
            df["stoch_k"] = stoch.iloc[:, 0]
            df["stoch_d"] = stoch.iloc[:, 1]

        # Volume MA + ratio
        df["vol_ma"]    = ta.sma(v, length=cfg.vol_ma_period)
        df["vol_ratio"] = v / df["vol_ma"].replace(0, np.nan)

        # Supertrend (за тренд потвърждение)
        try:
            st = ta.supertrend(h, lo, c, length=10, multiplier=3.0)
            if st is not None:
                df["supertrend"]      = st.iloc[:, 0]
                df["supertrend_dir"]  = st.iloc[:, 1]
        except Exception:
            pass

        # ADX (сила на тренда)
        try:
            adx = ta.adx(h, lo, c, length=14)
            if adx is not None:
                df["adx"] = adx.iloc[:, 0]
        except Exception:
            pass

        # Custom indicator code
        if cfg.custom_indicator_code.strip():
            df = self._run_custom(df, cfg.custom_indicator_code)

        return df.dropna(subset=["rsi", "ema_fast"])

    @staticmethod
    def _run_custom(df: pd.DataFrame, code: str) -> pd.DataFrame:
        """
        Изпълнява потребителски Python код за индикатор.
        Кодът получава `df` и трябва да му добави колони.
        """
        try:
            local_ns = {"df": df.copy(), "pd": pd, "ta": ta, "np": np}
            exec(code, local_ns)
            df = local_ns.get("df", df)
        except Exception as e:
            logger.error(f"Custom indicator error: {e}")
        return df


# ─────────────────────────────────────────────────────────────
#  SIGNAL GENERATOR
# ─────────────────────────────────────────────────────────────

class SignalGenerator:
    """
    Генерира сигнали (LONG / SHORT / HOLD) от индикаторния DataFrame.
    Прилага гласуваща система — всеки индикатор дава точки.
    """

    def generate(self, df: pd.DataFrame, cfg: StrategyConfig) -> dict:
        """
        Анализира последния ред на df и връща сигнал.
        """
        if df.empty or len(df) < 2:
            return self._hold("Недостатъчно данни")

        # Проверка на времевия филтър
        allowed, reason = cfg.time_filter.is_allowed()
        if not allowed:
            return self._hold(f"TimeFilter: {reason}")

        last = df.iloc[-1]
        prev = df.iloc[-2]

        votes_long  = 0
        votes_short = 0
        reasons     = []

        # ── RSI ─────────────────────────────────────────────
        if "rsi" in df.columns:
            rsi = last.get("rsi", 50)
            if rsi < cfg.rsi_oversold:
                votes_long  += 2
                reasons.append(f"RSI={rsi:.1f} oversold")
            elif rsi > cfg.rsi_overbought:
                votes_short += 2
                reasons.append(f"RSI={rsi:.1f} overbought")

        # ── EMA Alignment ───────────────────────────────────
        if all(c in df.columns for c in ["ema_fast","ema_medium","ema_slow"]):
            ef  = last.get("ema_fast",  0)
            em  = last.get("ema_medium",0)
            es  = last.get("ema_slow",  0)
            cur = last.get("close",     0)
            if ef > em > es:
                votes_long  += 2
                reasons.append("EMA alignment bullish")
            elif ef < em < es:
                votes_short += 2
                reasons.append("EMA alignment bearish")
            if cur > ef:
                votes_long  += 1
            else:
                votes_short += 1

        # ── MACD ────────────────────────────────────────────
        if all(c in df.columns for c in ["macd","macd_signal"]):
            ml  = last.get("macd",        0)
            ms  = last.get("macd_signal", 0)
            pml = prev.get("macd",        0)
            pms = prev.get("macd_signal", 0)
            if pml < pms and ml > ms:          # crossover up
                votes_long  += 2
                reasons.append("MACD crossover UP")
            elif pml > pms and ml < ms:        # crossover down
                votes_short += 2
                reasons.append("MACD crossover DOWN")
            elif ml > ms:
                votes_long  += 1
            else:
                votes_short += 1

        # ── Bollinger Bands ──────────────────────────────────
        if all(c in df.columns for c in ["bb_upper","bb_lower"]):
            close = last.get("close", 0)
            bbu   = last.get("bb_upper", 0)
            bbl   = last.get("bb_lower", 0)
            if close < bbl:
                votes_long  += 2
                reasons.append("Price below BB lower")
            elif close > bbu:
                votes_short += 2
                reasons.append("Price above BB upper")

        # ── Stochastic ───────────────────────────────────────
        if all(c in df.columns for c in ["stoch_k","stoch_d"]):
            sk  = last.get("stoch_k", 50)
            sd  = last.get("stoch_d", 50)
            psk = prev.get("stoch_k", 50)
            psd = prev.get("stoch_d", 50)
            if sk < 20 and sd < 20:
                votes_long  += 1
                reasons.append(f"Stoch oversold {sk:.0f}")
            elif sk > 80 and sd > 80:
                votes_short += 1
                reasons.append(f"Stoch overbought {sk:.0f}")
            if psk < psd and sk > sd:
                votes_long  += 1
                reasons.append("Stoch crossover UP")
            elif psk > psd and sk < sd:
                votes_short += 1
                reasons.append("Stoch crossover DOWN")

        # ── Volume ───────────────────────────────────────────
        if "vol_ratio" in df.columns:
            vr = last.get("vol_ratio", 1.0)
            if vr >= cfg.vol_multiplier:
                if votes_long >= votes_short:
                    votes_long  += 1
                    reasons.append(f"Vol spike {vr:.1f}x")
                else:
                    votes_short += 1
                    reasons.append(f"Vol spike {vr:.1f}x")

        # ── Supertrend ───────────────────────────────────────
        if "supertrend_dir" in df.columns:
            sd_val = last.get("supertrend_dir", 0)
            if sd_val > 0:
                votes_long  += 1
            elif sd_val < 0:
                votes_short += 1

        # ── ADX (само при тренд стратегии) ───────────────────
        if "adx" in df.columns and cfg.style in ["trend", "aggressive"]:
            adx_val = last.get("adx", 0)
            if adx_val > 25:
                if votes_long >= votes_short:
                    votes_long  += 1
                else:
                    votes_short += 1

        # ── РЕШЕНИЕ ──────────────────────────────────────────
        max_votes   = 12
        threshold   = 4 if cfg.style in ["safe", "balanced"] else 3

        if votes_long >= threshold and votes_long > votes_short:
            conf = min(97, 60 + (votes_long / max_votes) * 35)
            if conf >= cfg.min_confidence:
                return {
                    "signal":      "LONG",
                    "confidence":  round(conf, 1),
                    "votes_long":  votes_long,
                    "votes_short": votes_short,
                    "reasons":     reasons,
                    "atr":         last.get("atr", None),
                    "rsi":         last.get("rsi", None),
                }

        if votes_short >= threshold and votes_short > votes_long:
            conf = min(97, 60 + (votes_short / max_votes) * 35)
            if conf >= cfg.min_confidence:
                return {
                    "signal":      "SHORT",
                    "confidence":  round(conf, 1),
                    "votes_long":  votes_long,
                    "votes_short": votes_short,
                    "reasons":     reasons,
                    "atr":         last.get("atr", None),
                    "rsi":         last.get("rsi", None),
                }

        return self._hold(f"L={votes_long} S={votes_short} < {threshold}")

    @staticmethod
    def _hold(reason: str) -> dict:
        return {
            "signal": "HOLD", "confidence": 50,
            "votes_long": 0, "votes_short": 0,
            "reasons": [reason], "atr": None, "rsi": None
        }


# ─────────────────────────────────────────────────────────────
#  STRATEGY MANAGER (главен клас)
# ─────────────────────────────────────────────────────────────

class StrategyManager:
    """
    Управлява избраната стратегия, прилага индикаторите
    и генерира сигнали.

    Употреба:
    ---------
        sm  = StrategyManager()
        sm.select("NEXUS SMC Elite")
        sm.update_param("rsi_oversold", 30)

        df_with_indicators = sm.compute(df_ohlcv)
        signal = sm.signal(df_with_indicators)
    """

    def __init__(self):
        self._templates   = {k: StrategyConfig(**v.__dict__) for k, v in STRATEGY_TEMPLATES.items()}
        self._active_name = "NEXUS SMC Elite"
        self._indicators  = IndicatorEngine()
        self._signals     = SignalGenerator()

    @property
    def active(self) -> StrategyConfig:
        return self._templates[self._active_name]

    @property
    def names(self) -> list[str]:
        return list(self._templates.keys())

    def select(self, name: str) -> None:
        if name in self._templates:
            self._active_name = name
            logger.info(f"Strategy selected: {name}")

    def update_param(self, param: str, value) -> None:
        cfg = self.active
        if hasattr(cfg, param):
            setattr(cfg, param, value)
            logger.info(f"Strategy param updated: {param}={value}")

    def update_time_filter(self, **kwargs) -> None:
        tf = self.active.time_filter
        for k, v in kwargs.items():
            if hasattr(tf, k):
                setattr(tf, k, v)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._indicators.compute(df.copy(), self.active)

    def signal(self, df: pd.DataFrame) -> dict:
        return self._signals.generate(df, self.active)

    def get_config_dict(self) -> dict:
        """Връща настройките на активната стратегия като речник (за UI)."""
        cfg = self.active
        return {
            "name":            cfg.name,
            "description":     cfg.description,
            "style":           cfg.style,
            "timeframes":      cfg.timeframes,
            "rsi_period":      cfg.rsi_period,
            "rsi_oversold":    cfg.rsi_oversold,
            "rsi_overbought":  cfg.rsi_overbought,
            "ema_fast":        cfg.ema_fast,
            "ema_medium":      cfg.ema_medium,
            "ema_slow":        cfg.ema_slow,
            "macd_fast":       cfg.macd_fast,
            "macd_slow":       cfg.macd_slow,
            "macd_signal":     cfg.macd_signal,
            "bb_period":       cfg.bb_period,
            "bb_std":          cfg.bb_std,
            "atr_period":      cfg.atr_period,
            "vol_multiplier":  cfg.vol_multiplier,
            "min_confidence":  cfg.min_confidence,
            "time_filter_on":  cfg.time_filter.enabled,
            "sessions":        cfg.time_filter.sessions,
            "custom_start":    cfg.time_filter.custom_start,
            "custom_end":      cfg.time_filter.custom_end,
        }
