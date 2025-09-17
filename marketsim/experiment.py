"""High level experiment orchestration and configuration helpers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

from .fourheap import constants
from .fourheap.order import MatchedOrder
from .market.market import Market
from .simulator.sampled_arrival_simulator import SimulatorSampledArrival
from .simulator.simulator import Simulator


TradeCallback = Callable[["TradeEvent"], None]
TopOfBookCallback = Callable[["TopOfBookEvent"], None]


@dataclass
class TradeEvent:
    """Normalized trade information emitted by an :class:`Exchange`."""

    price: float
    qty: float
    side: str
    time: int
    agent_id: int


@dataclass
class TopOfBookEvent:
    """Snapshot of the current best bid and ask state."""

    best_bid: Optional[float]
    bid_size: float
    best_ask: Optional[float]
    ask_size: float
    time: int


class Exchange:
    """Simple event hub that exposes market activity via callbacks."""

    def __init__(self, name: str = "market-0") -> None:
        self.name = name
        self._market: Optional[Market] = None
        self._trade_handlers: List[TradeCallback] = []
        self._top_handlers: List[TopOfBookCallback] = []

    def bind_market(self, market: Market) -> None:
        """Attach the underlying :class:`Market` so book snapshots can be emitted."""

        self._market = market

    # Subscription helpers -----------------------------------------------------------------
    def on_trade(self, callback: TradeCallback) -> TradeCallback:
        self._trade_handlers.append(callback)
        return callback

    def on_top_of_book(self, callback: TopOfBookCallback) -> TopOfBookCallback:
        self._top_handlers.append(callback)
        return callback

    # Hooks used by the simulators ----------------------------------------------------------
    def handle_trade(self, matched_order: MatchedOrder) -> None:
        if not self._trade_handlers:
            return

        side = "BUY" if matched_order.order.order_type == constants.BUY else "SELL"
        event = TradeEvent(
            price=matched_order.price,
            qty=float(matched_order.order.quantity),
            side=side,
            time=matched_order.time,
            agent_id=matched_order.order.agent_id,
        )
        for handler in list(self._trade_handlers):
            handler(event)

    def handle_top_of_book(self, market: Market) -> None:
        if not self._top_handlers:
            return

        # Ensure we keep a reference to the latest market instance.
        if self._market is None:
            self._market = market

        book = market.order_book
        bid_order = book.buy_unmatched.peek_order()
        ask_order = book.sell_unmatched.peek_order()

        event = TopOfBookEvent(
            best_bid=self._finite_or_none(book.get_best_bid()),
            bid_size=float(bid_order.quantity) if bid_order is not None else 0.0,
            best_ask=self._finite_or_none(book.get_best_ask()),
            ask_size=float(ask_order.quantity) if ask_order is not None else 0.0,
            time=market.get_time(),
        )
        for handler in list(self._top_handlers):
            handler(event)

    @staticmethod
    def _finite_or_none(value: float) -> Optional[float]:
        if math.isinf(value):
            return None
        return value


@dataclass
class Config:
    """Declarative configuration used to build experiments."""

    simulator_cls: Type[Any]
    simulator_kwargs: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None

    @classmethod
    def single_market_default(
        cls,
        n_steps: int = 2_000,
        seed: Optional[int] = None,
        include_mm: bool = True,
        include_noise: bool = True,
        include_adversarial: bool = False,
        *,
        background_agents: int = 15,
        arrival_rate: float = 5e-3,
        mean: float = 1e5,
        r: float = 0.05,
        shock_var: float = 1e5,
        q_max: int = 10,
        shade: Optional[Sequence[float]] = None,
        pv_var: float = 5e6,
        eta: float = 0.2,
    ) -> "Config":
        """Return a configuration mirroring the original quick-start defaults."""

        # The original environment only models zero-intelligence background traders.
        # We keep the API surface (include_mm/include_adversarial) for compatibility,
        # but they currently do not toggle additional agent types.
        _ = include_adversarial  # explicitly unused for now

        num_agents = background_agents if include_noise else 0
        simulator_cls: Type[Any] = SimulatorSampledArrival if include_mm else Simulator
        simulator_kwargs: Dict[str, Any] = {
            "num_background_agents": max(1, num_agents),
            "sim_time": n_steps,
            "lam": arrival_rate,
            "mean": mean,
            "r": r,
            "shock_var": shock_var,
            "q_max": q_max,
            "pv_var": pv_var,
        }

        if simulator_cls is SimulatorSampledArrival:
            simulator_kwargs.update({
                "shade": list(shade) if shade is not None else None,
                "eta": eta,
                "lam_r": arrival_rate,
            })
        elif shade is not None:
            simulator_kwargs["zi_shade"] = list(shade)

        return cls(simulator_cls=simulator_cls, simulator_kwargs=simulator_kwargs, seed=seed)

    def build_simulator(self, *, market_observers: Optional[Sequence[object]] = None) -> Any:
        kwargs = dict(self.simulator_kwargs)
        if market_observers is not None:
            kwargs["market_observers"] = market_observers
        return self.simulator_cls(**kwargs)


@dataclass
class RunResult:
    steps: int
    trades: int
    final_fundamental: Optional[float]

    def summary(self) -> Dict[str, Optional[float]]:
        return {
            "steps": self.steps,
            "trades": self.trades,
            "final_fundamental": self.final_fundamental,
        }


class Experiment:
    """Coordinator that wires simulators to event feeds."""

    def __init__(self, config: Config):
        self.config = config
        self.simulator = self._build_simulator()
        self.exchange: Exchange = self._exchanges[0]
        self._trade_count = 0
        self.exchange.on_trade(self._record_trade)

    def _build_simulator(self) -> Any:
        if self.config.seed is not None:
            random.seed(self.config.seed)
            try:
                import numpy as np

                np.random.seed(self.config.seed)
            except Exception:
                pass
            try:
                import torch

                torch.manual_seed(self.config.seed)
            except Exception:
                pass

        self._exchanges: List[Exchange] = []
        market_observers: List[Exchange] = []

        # Prepare one exchange per market to keep the interface uniform.
        num_markets = self.config.simulator_kwargs.get("num_assets", 1)
        for idx in range(num_markets):
            exchange = Exchange(name=f"market-{idx}")
            self._exchanges.append(exchange)
            market_observers.append(exchange)

        simulator = self.config.build_simulator(market_observers=market_observers)
        for idx, exchange in enumerate(self._exchanges[: len(simulator.markets)]):
            exchange.bind_market(simulator.markets[idx])
        return simulator

    def _record_trade(self, _: TradeEvent) -> None:
        self._trade_count += 1

    def run(self) -> RunResult:
        self._trade_count = 0
        self.simulator.run()
        final_market = self.simulator.markets[0] if self.simulator.markets else None
        final_fundamental = (
            final_market.get_final_fundamental() if final_market is not None else None
        )
        steps = int(self.config.simulator_kwargs.get("sim_time", 0))
        return RunResult(steps=steps, trades=self._trade_count, final_fundamental=final_fundamental)

    @property
    def exchange(self) -> Exchange:
        return self._exchanges[0]

    @exchange.setter
    def exchange(self, value: Exchange) -> None:
        self._exchanges[0] = value
