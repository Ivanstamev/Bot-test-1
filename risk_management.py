"""
risk_management.py
══════════════════════════════════════════════════════════════
Динамичен Risk Management:
  • Position Sizing (% от баланса)
  • Trailing Stop Loss (автоматично местене на стопа)
  • Фиксиран / ATR-базиран TP & SL
  • Max drawdown guard
  • Daily loss limit
══════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  DATACLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    """Потребителски настройки за риска — редактират се от UI."""

    # Position sizing
    risk_pct: float = 1.0          # % от баланса рискуван на сделка (1–5)
    max_position_pct: float = 20.0 # Макс. % от баланса в една позиция

    # Take-profit / Stop-loss (ако не се ползва ATR)
    tp_pct: float = 2.0            # % TP от входната цена
    sl_pct: float = 1.0            # % SL от входната цена

    # ATR-базирани нива (приоритет пред фиксираните, ако use_atr=True)
    use_atr: bool = True
    atr_tp_mult: float = 2.5       # TP = вход ± ATR × mult
    atr_sl_mult: float = 1.0       # SL = вход ∓ ATR × mult

    # Trailing Stop
    trailing_stop: bool = True
    trailing_pct: float = 0.8      # % на изоставане на трейлинг стопа
    trailing_atr_mult: float = 1.5 # Ако use_atr: трейл = ATR × mult
    trailing_activation_pct: float = 0.5  # Активира се след X% в печалба

    # Дневни лимити
    daily_loss_limit_pct: float = 5.0   # Спира бота при X% загуба за деня
    max_consecutive_losses: int = 4      # Пауза след N поредни загуби

    # Leverage
    leverage: int = 10
    mode: str = "futures"  # "spot" | "futures"


@dataclass
class TrailingState:
    """Вътрешно състояние на трейлинг стопа за активна позиция."""
    active: bool = False
    direction: str = "LONG"          # "LONG" | "SHORT"
    entry_price: float = 0.0
    highest_price: float = 0.0       # за LONG — най-висока достигната цена
    lowest_price: float = float("inf")  # за SHORT — най-ниска
    current_stop: float = 0.0
    activated: bool = False          # дали е мин. activation_pct в печалба


@dataclass
class DailyStats:
    """Дневна статистика — нулира се всеки ден."""
    date: str = ""
    starting_balance: float = 0.0
    realized_pnl: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0
    bot_paused: bool = False


# ─────────────────────────────────────────────────────────────
#  POSITION SIZING
# ─────────────────────────────────────────────────────────────

class PositionSizer:
    """
    Изчислява размера на позицията спрямо риска.

    Формула (Risk-based):
        risk_amount  = balance × risk_pct / 100
        position_qty = risk_amount / (entry × sl_distance_pct / 100)

    За фючърси се прилага leverage.
    """

    def __init__(self, config: RiskConfig):
        self.cfg = config

    def calculate(
        self,
        balance: float,
        entry_price: float,
        sl_price: float,
        direction: str = "LONG",
    ) -> dict:
        """
        Връща речник с всички параметри за позицията.

        Parameters
        ----------
        balance      : текущ свободен баланс в USDT
        entry_price  : планирана входна цена
        sl_price     : планирана цена за стоп
        direction    : "LONG" | "SHORT"
        """
        if entry_price <= 0 or sl_price <= 0:
            return self._empty()

        # Разстояние до стопа в %
        if direction == "LONG":
            sl_dist_pct = (entry_price - sl_price) / entry_price * 100
        else:
            sl_dist_pct = (sl_price - entry_price) / entry_price * 100

        if sl_dist_pct <= 0:
            logger.warning("SL е по-добър от входа — невалидна позиция.")
            return self._empty()

        # Сума в риск
        risk_amount = balance * self.cfg.risk_pct / 100

        # Брой монети
        qty = risk_amount / (entry_price * sl_dist_pct / 100)

        # За фючърси — ефективен марджин
        if self.cfg.mode == "futures":
            notional  = qty * entry_price
            margin    = notional / self.cfg.leverage
            eff_notional = notional
        else:
            margin       = qty * entry_price
            eff_notional = margin

        # Максимален размер на позицията
        max_notional = balance * self.cfg.max_position_pct / 100
        if self.cfg.mode == "futures":
            max_notional *= self.cfg.leverage

        if eff_notional > max_notional:
            scale         = max_notional / eff_notional
            qty          *= scale
            margin       *= scale
            eff_notional *= scale

        # Закръгляне
        qty = self._round_qty(qty, entry_price)

        return {
            "qty":           qty,
            "entry_price":   entry_price,
            "sl_price":      sl_price,
            "risk_amount":   round(risk_amount, 2),
            "margin_used":   round(margin, 2),
            "notional":      round(eff_notional, 2),
            "sl_dist_pct":   round(sl_dist_pct, 3),
            "leverage":      self.cfg.leverage if self.cfg.mode == "futures" else 1,
            "mode":          self.cfg.mode,
        }

    @staticmethod
    def _round_qty(qty: float, price: float) -> float:
        """Закръглява qty спрямо цената (по-скъп актив = повече десетични)."""
        if price >= 10_000:
            return round(qty, 4)
        if price >= 1_000:
            return round(qty, 3)
        if price >= 100:
            return round(qty, 2)
        return round(qty, 1)

    @staticmethod
    def _empty() -> dict:
        return {"qty": 0, "entry_price": 0, "sl_price": 0,
                "risk_amount": 0, "margin_used": 0, "notional": 0,
                "sl_dist_pct": 0, "leverage": 1, "mode": "spot"}


# ─────────────────────────────────────────────────────────────
#  TP / SL CALCULATOR
# ─────────────────────────────────────────────────────────────

class TPSLCalculator:
    """
    Изчислява Take-Profit и Stop-Loss.
    Поддържа два режима:
      1. Фиксиран % (tp_pct / sl_pct)
      2. ATR-базиран (atr_tp_mult / atr_sl_mult)
    """

    def __init__(self, config: RiskConfig):
        self.cfg = config

    def calculate(
        self,
        entry: float,
        direction: str,
        atr: Optional[float] = None,
    ) -> dict:
        """
        Параметри
        ---------
        entry     : входна цена
        direction : "LONG" | "SHORT"
        atr       : стойност на ATR (ако use_atr=True)
        """
        use_atr = self.cfg.use_atr and atr and atr > 0

        if use_atr:
            tp_dist = atr * self.cfg.atr_tp_mult
            sl_dist = atr * self.cfg.atr_sl_mult
        else:
            tp_dist = entry * self.cfg.tp_pct / 100
            sl_dist = entry * self.cfg.sl_pct / 100

        if direction == "LONG":
            tp = entry + tp_dist
            sl = entry - sl_dist
        else:
            tp = entry - tp_dist
            sl = entry + sl_dist

        rr = tp_dist / sl_dist if sl_dist > 0 else 0

        return {
            "tp":        round(tp, 4),
            "sl":        round(sl, 4),
            "tp_dist":   round(tp_dist, 4),
            "sl_dist":   round(sl_dist, 4),
            "rr_ratio":  round(rr, 2),
            "atr_based": use_atr,
        }


# ─────────────────────────────────────────────────────────────
#  TRAILING STOP ENGINE
# ─────────────────────────────────────────────────────────────

class TrailingStopEngine:
    """
    Трейлинг Стоп — мести стопа автоматично при движение в посоката ни.

    Алгоритъм за LONG:
      1. Активиране: цена > вход × (1 + activation_pct/100)
      2. При всеки тик, ако цена > highest_price → update highest
      3. new_stop = highest × (1 - trailing_pct/100)
      4. Ако current_price <= new_stop → изход (стопиран)

    За SHORT — огледален.
    """

    def __init__(self, config: RiskConfig):
        self.cfg   = config
        self.state = TrailingState()

    def open_position(
        self,
        direction: str,
        entry_price: float,
        initial_sl: float,
    ) -> None:
        """Инициализира трейлинг стопа при влизане в позиция."""
        self.state = TrailingState(
            active        = True,
            direction     = direction,
            entry_price   = entry_price,
            highest_price = entry_price,
            lowest_price  = entry_price,
            current_stop  = initial_sl,
            activated     = False,
        )
        logger.info(f"TrailingStop open: {direction} @ {entry_price}, init_sl={initial_sl}")

    def update(self, current_price: float, atr: Optional[float] = None) -> dict:
        """
        Обновява трейлинг стопа при всеки ценови тик.

        Връща:
          {
            "stop":       float  — текущ стоп
            "activated":  bool   — дали е активиран трейлингът
            "hit":        bool   — дали цената е ударила стопа
            "moved":      bool   — дали стопът се е преместил
          }
        """
        s = self.state
        if not s.active:
            return {"stop": 0, "activated": False, "hit": False, "moved": False}

        old_stop = s.current_stop

        # Изчисли разстояние на трейлинга
        if self.cfg.use_atr and atr and atr > 0:
            trail_dist = atr * self.cfg.trailing_atr_mult
        else:
            trail_dist = current_price * self.cfg.trailing_pct / 100

        activation_dist = s.entry_price * self.cfg.trailing_activation_pct / 100

        if s.direction == "LONG":
            # Активирай ако сме достатъчно в печалба
            if not s.activated and current_price >= s.entry_price + activation_dist:
                s.activated = True
                logger.info(f"Trailing ACTIVATED at {current_price:.4f}")

            if current_price > s.highest_price:
                s.highest_price = current_price

            if s.activated:
                new_stop = s.highest_price - trail_dist
                if new_stop > s.current_stop:
                    s.current_stop = new_stop

            hit = current_price <= s.current_stop

        else:  # SHORT
            if not s.activated and current_price <= s.entry_price - activation_dist:
                s.activated = True
                logger.info(f"Trailing ACTIVATED at {current_price:.4f}")

            if current_price < s.lowest_price:
                s.lowest_price = current_price

            if s.activated:
                new_stop = s.lowest_price + trail_dist
                if new_stop < s.current_stop:
                    s.current_stop = new_stop

            hit = current_price >= s.current_stop

        if hit:
            logger.info(f"TrailingStop HIT at {current_price:.4f}, stop was {s.current_stop:.4f}")
            s.active = False

        return {
            "stop":      round(s.current_stop, 4),
            "activated": s.activated,
            "hit":       hit,
            "moved":     s.current_stop != old_stop,
            "highest":   round(s.highest_price, 4),
            "lowest":    round(s.lowest_price, 4),
        }

    def close(self) -> None:
        """Затваря трейлинг стопа (позицията е затворена)."""
        self.state.active = False

    @property
    def current_stop(self) -> float:
        return self.state.current_stop

    @property
    def is_active(self) -> bool:
        return self.state.active


# ─────────────────────────────────────────────────────────────
#  DAILY GUARD
# ─────────────────────────────────────────────────────────────

class DailyGuard:
    """
    Защита от свръхтъргуване и дневни загуби.
    Автоматично спира бота при:
      • достигане на daily_loss_limit_pct
      • max_consecutive_losses поредни загуби
    """

    def __init__(self, config: RiskConfig):
        self.cfg   = config
        self.stats = DailyStats()

    def new_day(self, balance: float) -> None:
        from datetime import date
        today = str(date.today())
        if self.stats.date != today:
            self.stats = DailyStats(date=today, starting_balance=balance)
            logger.info(f"New trading day: {today}, balance={balance}")

    def record_trade(self, pnl_usdt: float) -> bool:
        """
        Записва сделка. Връща True ако ботът трябва да спре.
        """
        self.stats.trades += 1
        self.stats.realized_pnl += pnl_usdt

        if pnl_usdt > 0:
            self.stats.wins += 1
            self.stats.consecutive_losses = 0
        else:
            self.stats.losses += 1
            self.stats.consecutive_losses += 1

        # Проверка за дневна загуба
        if self.stats.starting_balance > 0:
            loss_pct = -self.stats.realized_pnl / self.stats.starting_balance * 100
            if loss_pct >= self.cfg.daily_loss_limit_pct:
                self.stats.bot_paused = True
                logger.warning(f"Daily loss limit hit: {loss_pct:.2f}%. Bot PAUSED.")
                return True

        # Проверка за поредни загуби
        if self.stats.consecutive_losses >= self.cfg.max_consecutive_losses:
            self.stats.bot_paused = True
            logger.warning(f"{self.stats.consecutive_losses} consecutive losses. Bot PAUSED.")
            return True

        return False

    @property
    def should_pause(self) -> bool:
        return self.stats.bot_paused

    def resume(self) -> None:
        self.stats.bot_paused = False
        self.stats.consecutive_losses = 0


# ─────────────────────────────────────────────────────────────
#  UNIFIED RISK MANAGER
# ─────────────────────────────────────────────────────────────

class RiskManager:
    """
    Главен клас — комбинира всички риск компоненти.

    Използване:
    -----------
        cfg = RiskConfig(risk_pct=1.5, trailing_stop=True, leverage=10)
        rm  = RiskManager(cfg)

        # При нов сигнал:
        pos   = rm.build_position(balance=5000, entry=43000, direction="LONG", atr=200)
        rm.open_trailing(direction="LONG", entry=43000, sl=pos["levels"]["sl"])

        # При всеки тик:
        trail = rm.tick_trailing(current_price=43500, atr=210)
        if trail["hit"]:
            rm.close_position(pnl_usdt=-50)
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.cfg      = config or RiskConfig()
        self.sizer    = PositionSizer(self.cfg)
        self.tpsl     = TPSLCalculator(self.cfg)
        self.trailing = TrailingStopEngine(self.cfg)
        self.daily    = DailyGuard(self.cfg)

    def update_config(self, **kwargs) -> None:
        """Обновява конфигурацията в реално време от UI."""
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        # Ре-инициализирай компонентите
        self.sizer    = PositionSizer(self.cfg)
        self.tpsl     = TPSLCalculator(self.cfg)

    def build_position(
        self,
        balance: float,
        entry: float,
        direction: str,
        atr: Optional[float] = None,
    ) -> dict:
        """
        Изгражда пълен план за позиция:
        qty, TP, SL, margin, R:R.
        """
        levels  = self.tpsl.calculate(entry, direction, atr)
        sizing  = self.sizer.calculate(balance, entry, levels["sl"], direction)

        return {
            "direction":  direction,
            "entry":      entry,
            "levels":     levels,
            "sizing":     sizing,
            "summary": {
                "qty":       sizing["qty"],
                "tp":        levels["tp"],
                "sl":        levels["sl"],
                "rr":        levels["rr_ratio"],
                "risk_usdt": sizing["risk_amount"],
                "margin":    sizing["margin_used"],
                "notional":  sizing["notional"],
            }
        }

    def open_trailing(self, direction: str, entry: float, sl: float) -> None:
        if self.cfg.trailing_stop:
            self.trailing.open_position(direction, entry, sl)

    def tick_trailing(
        self, current_price: float, atr: Optional[float] = None
    ) -> dict:
        if self.cfg.trailing_stop and self.trailing.is_active:
            return self.trailing.update(current_price, atr)
        return {"stop": 0, "activated": False, "hit": False, "moved": False}

    def close_position(self, pnl_usdt: float) -> bool:
        """Записва затворена позиция. Връща True ако ботът трябва да спре."""
        self.trailing.close()
        return self.daily.record_trade(pnl_usdt)

    def new_day(self, balance: float) -> None:
        self.daily.new_day(balance)

    @property
    def should_pause(self) -> bool:
        return self.daily.should_pause

    def get_status(self) -> dict:
        s = self.daily.stats
        return {
            "date":               s.date,
            "daily_pnl":          round(s.realized_pnl, 2),
            "daily_trades":       s.trades,
            "wins":               s.wins,
            "losses":             s.losses,
            "consecutive_losses": s.consecutive_losses,
            "paused":             s.bot_paused,
            "trailing_active":    self.trailing.is_active,
            "trailing_stop":      self.trailing.current_stop,
        }
