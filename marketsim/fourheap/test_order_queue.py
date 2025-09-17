import pytest

from marketsim.fourheap.order import Order
from marketsim.fourheap.order_queue import OrderQueue


def test_order_queue_operations():
    order_queue = OrderQueue(is_max_heap=True)

    order1 = Order(price=100.0, order_type=1, quantity=10.0, agent_id=1, time=1, order_id=1)
    order2 = Order(price=200.0, order_type=1, quantity=5.0, agent_id=2, time=2, order_id=2)

    order_queue.add_order(order1)
    assert order_queue.count() == pytest.approx(order1.quantity)
    assert order_queue.contains(order1.order_id)

    order_queue.add_order(order2)
    assert order_queue.count() == pytest.approx(order1.quantity + order2.quantity)
    assert order_queue.contains(order2.order_id)

    top_order = order_queue.peek_order()
    assert top_order.order_id in {order1.order_id, order2.order_id}

    order_queue.remove(order1.order_id)
    assert not order_queue.contains(order1.order_id)
    assert order_queue.count() == pytest.approx(order2.quantity)

    popped_order = order_queue.push_to()
    assert popped_order == order2
    assert order_queue.count() == 0
    assert order_queue.is_empty()

    order_queue.add_order(order1)
    order_queue.clear()
    assert order_queue.count() == 0
    assert order_queue.is_empty()


def test_order_comparisons():
    # Create some Order instances
    buy_order1 = Order(price=200.0, order_type=1, quantity=20.0, agent_id=3, time=1, order_id=3)
    buy_order2 = Order(price=200.0, order_type=1, quantity=15.0, agent_id=4, time=2, order_id=4)
    sell_order1 = Order(price=100.0, order_type=-1, quantity=10.0, agent_id=1, time=1, order_id=1)
    sell_order2 = Order(price=100.0, order_type=-1, quantity=5.0, agent_id=2, time=2, order_id=2)

    # Test __gt__ method for buy orders
    assert buy_order1 > buy_order2
    assert not (buy_order2 > buy_order1)

    # Test __gt__ method for sell orders
    assert sell_order1 > sell_order2
    assert not (sell_order2 > sell_order1)


# if you want to run this file directly, you can use the following command
if __name__ == "__main__":
    pytest.main()
