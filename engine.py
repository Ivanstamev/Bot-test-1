"""
engine.py
══════════════════════════════════════════════════════════════
TradingBot — ядрото на бота:
  • API връзка с MEXC и BingX (чрез ccxt)
  • Spot / Futures с leverage
  • Демо режим (симулиран баланс)
  • Backtest (7 / 30 дни) с equity curve
  • Live цикъл и управление на позиции
══════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import ccxt
import pandas as pd
import numpy as np

from risk_management  import RiskManager, RiskConfig
from strategy_manager import StrategyManager
from logic            import AIBrain

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  BOT CONFIG
# ─────────────────────────────────────────────────────────────

class BotConfig:
    def __init__(self):
        self.exchange_id:  str   = os.getenv("DEFAULT_EXCHANGE",  "mexc")
        self.symbol:       str   = os.getenv("DEFAULT_SYMBOL",    "BTC/USDT")
        self.mode:         str   = os.getenv("DEFAULT_MODE",      "futures")
        self.leverage:     int   = int(os.getenv("DEFAULT_LEVERAGE", 10))
        self.timeframe:    str   = "15m"
        self.demo_mode:    bool  = True
        self.demo_balance: float = 10_000.0
        self.use_ai:       bool  = False
        self.use_corr:     bool  = True


# ─────────────────────────────────────────────────────────────
#  EXCHANGE CONNECTOR
# ─────────────────────────────────────────────────────────────

class ExchangeConnector:
    """Обвивка около ccxt — MEXC и BingX."""

    EXCHANGE_MAP = {
        "mexc":  ccxt.mexc,
        "bingx": ccxt.bingx,
    }

    def __init__(self, exchange_id: str, mode: str = "futures"):
        self.exchange_id = exchange_id.lower()
        self.mode        = mode
        self._ex: Optional[ccxt.Exchange] = None

    def connect(
        self,
        api_key:    str = "",
        api_secret: str = "",
    ) -> bool:
        """Свързва се с борсата. Връща True при успех."""
        cls = self.EXCHANGE_MAP.get(self.exchange_id)
        if cls is None:
            logger.error(f"Непозната борса: {self.exchange_id}")
            return False
        try:
            params = {
                "apiKey":  api_key    or os.getenv(f"{self.exchange_id.upper()}_API_KEY",    ""),
                "secret":  api_secret or os.getenv(f"{self.exchange_id.upper()}_API_SECRET", ""),
                "options": {},
            }
            if self.mode == "futures":
                if self.exchange_id == "mexc":
                    params["options"]["defaultType"] = "swap"
                elif self.exchange_id == "bingx":
                    params["options"]["defaultType"] = "swap"

            self._ex = cls(params)
            self._ex.load_markets()
            logger.info(f"Свързан с {self.exchange_id} ({self.mode})")
            return True
        except Exception as e:
            logger.error(f"Грешка при свързване: {e}")
            return False

    def get_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 300
    ) -> Optional[pd.DataFrame]:
        if self._ex is None:
            return None
        try:
            raw = self._ex.fetch_ohlcv(symbol, timeframe, limit=limit)
            df  = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df.astype(float)
        except Exception as e:
            logger.error(f"OHLCV error {symbol} {timeframe}: {e}")
            return None

    def get_balance(self) -> dict:
        if self._ex is None:
            return {}
        try:
            bal = self._ex.fetch_balance()
            return bal.get("USDT", {})
        except Exception as e:
            logger.error(f"Balance error: {e}")
            return {}

    def place_order(
        self,
        symbol:    str,
        side:      str,    # "buy" | "sell"
        qty:       float,
        order_type: str = "market",
        price:     Optional[float] = None,
        params:    dict = None,
    ) -> Optional[dict]:
        if self._ex is None:
            logger.warning("Не е свързан с борса")
            return None
        try:
            order = self._ex.create_order(
                symbol    = symbol,
                type      = order_type,
                side      = side,
                amount    = qty,
                price     = price,
                params    = params or {},
            )
            logger.info(f"Поръчка изпратена: {side} {qty} {symbol} @ {price or 'market'}")
            return order
        except Exception as e:
            logger.error(f"Order error: {e}")
            return None

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        if self._ex is None:
            return False
        try:
            self._ex.set_leverage(leverage, symbol)
            return True
        except Exception as e:
            logger.warning(f"Leverage error: {e}")
            return False

    def get_ticker(self, symbol: str) -> Optional[dict]:
        if self._ex is None:
            return None
        try:
            return self._ex.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Ticker error: {e}")
            return None

    @property
    def connected(self) -> bool:
        return self._ex is not None


# ─────────────────────────────────────────────────────────────
#  DEMO ACCOUNT
# ─────────────────────────────────────────────────────────────

class DemoAccount:
    """Симулирана сметка за тестване без реални средства."""

    def __init__(self, initial_balance: float = 10_000.0):
        self.initial  = initial_balance
        self.balance  = initial_balance
        self.position: Optional[dict] = None
        self.trades:   list[dict]     = []
        self._price_history: list[float] = []

    def reset(self, new_balance: float) -> None:
        self.__init__(new_balance)

    def open_position(
        self, direction: str, entry: float, qty: float,
        tp: float, sl: float, strategy: str,
    ) -> None:
        self.position = {
            "direction": direction,
            "entry":     entry,
            "qty":       qty,
            "tp":        tp,
            "sl":        sl,
            "strategy":  strategy,
            "opened_at": datetime.utcnow().isoformat(),
            "margin":    entry * qty,
        }

    def update_position(
        self,
        current_price: float,
        trailing_stop: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Проверява дали позицията е ударила TP/SL/TrailingStop.
        Връща dict с резултата или None.
        """
        if self.position is None:
            return None

        p   = self.position
        dir = p["direction"]
        tp  = p["tp"]
        sl  = p["sl"] if trailing_stop is None else trailing_stop

        hit_tp = (dir == "LONG"  and current_price >= tp) or \
                 (dir == "SHORT" and current_price <= tp)
        hit_sl = (dir == "LONG"  and current_price <= sl) or \
                 (dir == "SHORT" and current_price >= sl)

        if hit_tp:
            return self._close(current_price, "TP")
        if hit_sl:
            return self._close(current_price, "SL" if trailing_stop is None else "Trailing SL")
        return None

    def _close(self, exit_price: float, reason: str) -> dict:
        p = self.position
        if p["direction"] == "LONG":
            pnl_pct  = (exit_price - p["entry"]) / p["entry"] * 100
        else:
            pnl_pct  = (p["entry"] - exit_price) / p["entry"] * 100

        pnl_usdt = pnl_pct / 100 * p["margin"]
        self.balance += pnl_usdt

        trade = {
            **p,
            "exit":       exit_price,
            "exit_reason": reason,
            "pnl_pct":    round(pnl_pct, 3),
            "pnl_usdt":   round(pnl_usdt, 2),
            "closed_at":  datetime.utcnow().isoformat(),
            "balance":    round(self.balance, 2),
        }
        self.trades.append(trade)
        self.position = None
        return trade

    def force_close(self, current_price: float) -> Optional[dict]:
        if self.position:
            return self._close(current_price, "Manual Close")
        return None

    @property
    def pnl(self) -> float:
        return round(self.balance - self.initial, 2)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t["pnl_usdt"] > 0)
        return round(wins / len(self.trades) * 100, 1)

    @property
    def equity_curve(self) -> list[dict]:
        """Equity curve за plotly chart."""
        curve = [{"time": "Start", "balance": self.initial}]
        running = self.initial
        for t in self.trades:
            running += t["pnl_usdt"]
            curve.append({
                "time":    t.get("closed_at", ""),
                "balance": round(running, 2),
                "trade":   f"{t['direction']} {t['exit_reason']} {t['pnl_pct']:+.2f}%"
            })
        return curve


# ─────────────────────────────────────────────────────────────
#  BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Backtест върху реални OHLCV данни от борсата.
    Поддържа 7 и 30 дни с пълна хронология и equity curve.
    """

    def __init__(
        self,
        strategy_manager: StrategyManager,
        risk_manager:     RiskManager,
    ):
        self.sm = strategy_manager
        self.rm = risk_manager

    def run(
        self,
        df:          pd.DataFrame,
        initial_bal: float = 10_000.0,
    ) -> dict:
        """
        Параметри
        ---------
        df           : OHLCV данни (достатъчно история)
        initial_bal  : начален баланс за теста

        Връща
        ------
        {
          "trades":       list[dict],
          "equity_curve": list[dict],
          "stats":        dict,
        }
        """
        df_ind  = self.sm.compute(df.copy())
        balance = initial_bal
        trades  = []
        equity  = [{"time": str(df_ind.index[0]), "balance": initial_bal}]

        position: Optional[dict] = None

        for i in range(50, len(df_ind)):
            window  = df_ind.iloc[:i]
            row     = df_ind.iloc[i]
            price   = row["close"]
            atr     = row.get("atr", None)
            ts      = str(df_ind.index[i])

            # Управление на отворена позиция
            if position:
                hit_tp = (position["dir"] == "LONG"  and price >= position["tp"]) or \
                         (position["dir"] == "SHORT" and price <= position["tp"])
                hit_sl = (position["dir"] == "LONG"  and price <= position["sl"]) or \
                         (position["dir"] == "SHORT" and price >= position["sl"])

                # Трейлинг стоп
                if self.rm.cfg.trailing_stop:
                    tr = self.rm.tick_trailing(price, atr)
                    if tr["hit"]:
                        hit_sl = True
                        position["sl"] = tr["stop"]

                if hit_tp or hit_sl:
                    reason = "TP" if hit_tp else "SL/Trail"
                    if position["dir"] == "LONG":
                        pnl_pct = (price - position["entry"]) / position["entry"] * 100
                    else:
                        pnl_pct = (position["entry"] - price) / position["entry"] * 100
                    pnl_usdt = pnl_pct / 100 * position["margin"]
                    balance += pnl_usdt
                    self.rm.close_position(pnl_usdt)

                    trades.append({
                        "direction": position["dir"],
                        "entry":     round(position["entry"], 2),
                        "exit":      round(price, 2),
                        "tp":        round(position["tp"], 2),
                        "sl":        round(position["sl"], 2),
                        "pnl_pct":   round(pnl_pct, 3),
                        "pnl_usdt":  round(pnl_usdt, 2),
                        "reason":    reason,
                        "opened":    position["opened"],
                        "closed":    ts,
                        "balance":   round(balance, 2),
                    })
                    equity.append({"time": ts, "balance": round(balance, 2)})
                    position = None
                    continue

            # Нов сигнал (само ако нямаме позиция)
            if position is None and not self.rm.should_pause:
                sig_data = self.sm.signal(window)
                sig = sig_data.get("signal", "HOLD")
                conf = sig_data.get("confidence", 0)

                if sig != "HOLD" and conf >= self.sm.active.min_confidence:
                    pos_plan = self.rm.build_position(
                        balance=balance, entry=price,
                        direction=sig, atr=atr,
                    )
                    qty = pos_plan["sizing"]["qty"]
                    if qty > 0:
                        self.rm.open_trailing(sig, price, pos_plan["levels"]["sl"])
                        position = {
                            "dir":    sig,
                            "entry":  price,
                            "tp":     pos_plan["levels"]["tp"],
                            "sl":     pos_plan["levels"]["sl"],
                            "margin": pos_plan["sizing"]["margin_used"],
                            "opened": ts,
                        }

        # Затвори отворена позиция в края
        if position and len(df_ind) > 0:
            price = df_ind.iloc[-1]["close"]
            if position["dir"] == "LONG":
                pnl_pct = (price - position["entry"]) / position["entry"] * 100
            else:
                pnl_pct = (position["entry"] - price) / position["entry"] * 100
            pnl_usdt = pnl_pct / 100 * position["margin"]
            balance += pnl_usdt
            trades.append({
                "direction": position["dir"],
                "entry":     round(position["entry"], 2),
                "exit":      round(price, 2),
                "tp":        round(position["tp"], 2),
                "sl":        round(position["sl"], 2),
                "pnl_pct":   round(pnl_pct, 3),
                "pnl_usdt":  round(pnl_usdt, 2),
                "reason":    "End of Test",
                "opened":    position["opened"],
                "closed":    str(df_ind.index[-1]),
                "balance":   round(balance, 2),
            })
            equity.append({"time": str(df_ind.index[-1]), "balance": round(balance, 2)})

        # Статистики
        wins   = [t for t in trades if t["pnl_usdt"] > 0]
        losses = [t for t in trades if t["pnl_usdt"] <= 0]
        total  = len(trades)
        pnl    = balance - initial_bal
        max_dd = self._max_drawdown(equity)

        stats = {
            "initial":        initial_bal,
            "final":          round(balance, 2),
            "pnl":            round(pnl, 2),
            "return_pct":     round(pnl / initial_bal * 100, 2),
            "total_trades":   total,
            "wins":           len(wins),
            "losses":         len(losses),
            "win_rate":       round(len(wins) / total * 100, 1) if total else 0,
            "avg_win":        round(sum(t["pnl_usdt"] for t in wins) / len(wins), 2) if wins else 0,
            "avg_loss":       round(sum(t["pnl_usdt"] for t in losses) / len(losses), 2) if losses else 0,
            "max_drawdown":   round(max_dd, 2),
            "profit_factor":  self._profit_factor(wins, losses),
        }

        return {"trades": trades, "equity_curve": equity, "stats": stats}

    @staticmethod
    def _max_drawdown(equity: list[dict]) -> float:
        if not equity:
            return 0.0
        balances = [e["balance"] for e in equity]
        peak     = balances[0]
        max_dd   = 0.0
        for b in balances:
            if b > peak:
                peak = b
            dd = (peak - b) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @staticmethod
    def _profit_factor(wins: list, losses: list) -> float:
        gross_profit = sum(t["pnl_usdt"] for t in wins)
        gross_loss   = abs(sum(t["pnl_usdt"] for t in losses))
        if gross_loss == 0:
            return round(gross_profit, 2)
        return round(gross_profit / gross_loss, 2)


# ─────────────────────────────────────────────────────────────
#  TRADING BOT (главен клас)
# ─────────────────────────────────────────────────────────────

class TradingBot:
    """
    Главен клас — обединява всичко.

    Употреба (от app.py):
    ---------------------
        bot = TradingBot()
        bot.setup(exchange="mexc", api_key="...", api_secret="...")
        bot.set_demo(True, 5000)
        bot.select_strategy("NEXUS SMC Elite")
        bot.start()                          # стартира цикъла
        signal = bot.get_latest_signal()     # за UI
    """

    def __init__(self, config: Optional[BotConfig] = None):
        self.cfg      = config or BotConfig()
        self.strategy = StrategyManager()
        self.risk     = RiskManager(RiskConfig(
            leverage = self.cfg.leverage,
            mode     = self.cfg.mode,
        ))
        self.brain    = AIBrain(use_ai=self.cfg.use_ai)
        self.exchange = ExchangeConnector(self.cfg.exchange_id, self.cfg.mode)
        self.demo     = DemoAccount(self.cfg.demo_balance)
        self.backtest = BacktestEngine(self.strategy, self.risk)

        self._running       = False
        self._latest_signal: dict = {}
        self._mtf_cache:    dict  = {}
        self._signals_log:  list  = []   # Всички сигнали за графиката

    # ── Setup ────────────────────────────────────────────────
    def setup(
        self,
        exchange:   str  = "mexc",
        api_key:    str  = "",
        api_secret: str  = "",
        mode:       str  = "futures",
        leverage:   int  = 10,
    ) -> bool:
        self.cfg.exchange_id = exchange
        self.cfg.mode        = mode
        self.cfg.leverage    = leverage
        self.exchange        = ExchangeConnector(exchange, mode)
        success = self.exchange.connect(api_key, api_secret)
        if success and mode == "futures":
            self.exchange.set_leverage(self.cfg.symbol, leverage)
        self.risk.update_config(leverage=leverage, mode=mode)
        return success

    def set_demo(self, enabled: bool, balance: Optional[float] = None) -> None:
        self.cfg.demo_mode = enabled
        if balance:
            self.demo.reset(balance)

    def select_strategy(self, name: str) -> None:
        self.strategy.select(name)

    def set_symbol(self, symbol: str) -> None:
        self.cfg.symbol = symbol

    def set_timeframe(self, tf: str) -> None:
        self.cfg.timeframe = tf

    # ── Data Fetching ─────────────────────────────────────────
    def _fetch_df(
        self, symbol: Optional[str] = None, timeframe: Optional[str] = None, limit: int = 300
    ) -> Optional[pd.DataFrame]:
        sym = symbol or self.cfg.symbol
        tf  = timeframe or self.cfg.timeframe

        if self.cfg.demo_mode:
            return self._generate_demo_ohlcv(limit=limit)

        return self.exchange.get_ohlcv(sym, tf, limit)

    @staticmethod
    def _generate_demo_ohlcv(limit: int = 300, base: float = 43_000.0) -> pd.DataFrame:
        """Синтетични OHLCV данни за демо режим."""
        rng    = pd.date_range(end=datetime.utcnow(), periods=limit, freq="15min")
        prices = [base]
        for _ in range(limit - 1):
            chg = np.random.normal(0.0002, 0.004)
            prices.append(max(prices[-1] * (1 + chg), 1))

        data = {
            "open":   [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            "high":   [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
            "low":    [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
            "close":  prices,
            "volume": [abs(np.random.normal(1000, 300)) for _ in prices],
        }
        df = pd.DataFrame(data, index=rng)
        return df.astype(float)

    def _fetch_mtf(self) -> dict[str, pd.DataFrame]:
        tfs = {"5m": 100, "15m": 200, "1h": 200, "4h": 150, "1d": 100}
        result = {}
        for tf, lim in tfs.items():
            df = self._fetch_df(timeframe=tf, limit=lim)
            if df is not None:
                result[tf] = df
        self._mtf_cache = result
        return result

    # ── Main Loop ─────────────────────────────────────────────
    def tick(self) -> dict:
        """
        Един цикъл на бота. Извиква се от Streamlit на интервал.
        Връща последния сигнал и статус.
        """
        df = self._fetch_df()
        if df is None or len(df) < 50:
            return {"error": "Недостатъчно данни"}

        df_ind = self.strategy.compute(df)
        strat_sig = self.strategy.signal(df_ind)

        # Multi-TF (на по-рядко — кешира)
        mtf = self._mtf_cache or self._fetch_mtf()

        # ETH за корелация
        eth_df = self._fetch_df("ETH/USDT", "1h", 50) if self.brain.corr_filter.enabled else None

        # AI Brain финално решение
        final = self.brain.decide(
            strategy_signal = strat_sig,
            symbol          = self.cfg.symbol,
            mtf_data        = mtf,
            eth_df          = eth_df,
        )

        current_price = df["close"].iloc[-1]
        atr           = df_ind["atr"].iloc[-1] if "atr" in df_ind.columns else None

        # Актуализирай дневна статистика
        self.risk.new_day(
            self.demo.balance if self.cfg.demo_mode
            else float(self.exchange.get_balance().get("free", 0))
        )

        # Управление на позиция
        if self.cfg.demo_mode and self.demo.position:
            tr = self.risk.tick_trailing(current_price, atr)
            closed = self.demo.update_position(
                current_price,
                trailing_stop=tr["stop"] if tr["activated"] else None,
            )
            if closed:
                self.risk.close_position(closed["pnl_usdt"])

        # Отвори нова позиция
        sig = final["signal"]
        if (sig in ("LONG", "SHORT") and
                not self.risk.should_pause and
                (not self.cfg.demo_mode or self.demo.position is None)):

            bal = self.demo.balance if self.cfg.demo_mode else \
                  float(self.exchange.get_balance().get("free", 0))

            pos_plan = self.risk.build_position(
                balance=bal, entry=current_price, direction=sig, atr=atr
            )
            levels = pos_plan["levels"]
            qty    = pos_plan["sizing"]["qty"]

            if qty > 0:
                self.risk.open_trailing(sig, current_price, levels["sl"])

                if self.cfg.demo_mode:
                    self.demo.open_position(
                        direction = sig,
                        entry     = current_price,
                        qty       = qty,
                        tp        = levels["tp"],
                        sl        = levels["sl"],
                        strategy  = self.strategy._active_name,
                    )
                else:
                    side = "buy" if sig == "LONG" else "sell"
                    self.exchange.place_order(self.cfg.symbol, side, qty)

                # Запис на сигнала за графиката
                self._signals_log.append({
                    "time":       datetime.utcnow().isoformat(),
                    "price":      current_price,
                    "signal":     sig,
                    "confidence": final["confidence"],
                    "strategy":   self.strategy._active_name,
                })

        self._latest_signal = {
            **final,
            "current_price": current_price,
            "atr":           atr,
            "strategy":      self.strategy._active_name,
            "demo_balance":  self.demo.balance,
            "demo_position": self.demo.position,
            "demo_pnl":      self.demo.pnl,
            "win_rate":      self.demo.win_rate,
            "risk_status":   self.risk.get_status(),
        }
        return self._latest_signal

    def get_latest_signal(self) -> dict:
        return self._latest_signal

    def get_signals_log(self) -> list:
        return self._signals_log

    # ── Backtest Wrapper ──────────────────────────────────────
    def run_backtest(self, days: int = 7) -> dict:
        limit = days * 24 * 4   # 15-min candles
        df    = self._fetch_df(limit=limit)
        if df is None:
            return {"error": "Не могат да се заредят данни за бектест"}
        return self.backtest.run(df, initial_bal=self.demo.initial)

    # ── Convenience ──────────────────────────────────────────
    @property
    def is_demo(self) -> bool:
        return self.cfg.demo_mode

    @property
    def connected(self) -> bool:
        return self.exchange.connected
