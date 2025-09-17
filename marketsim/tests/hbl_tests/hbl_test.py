import pytest

from marketsim import Config, Experiment


def test_experiment_emits_top_of_book_events():
    cfg = Config.single_market_default(
        n_steps=50,
        seed=123,
        background_agents=5,
        arrival_rate=0.5,
    )
    exp = Experiment(cfg)

    top_events = []
    trades = []
    exp.exchange.on_top_of_book(top_events.append)
    exp.exchange.on_trade(trades.append)

    result = exp.run()

    summary = result.summary()
    assert summary["steps"] == 50
    assert top_events, "expected at least one top-of-book event"
    assert result.trades == len(trades)
    for event in top_events:
        assert event.time >= 0
        assert event.bid_size >= 0
        assert event.ask_size >= 0
