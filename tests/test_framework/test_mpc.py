from framework.mpc.broker import MessageBroker
from framework.mpc.message import Message


def test_mpc_routes_through_orc():
    broker = MessageBroker()
    msg = Message(
        message_id="1",
        message_type="TASK",
        sender="vra",
        recipient="ga",
        payload={"x": 1},
        priority=1,
        ttl_seconds=60,
    )
    broker.publish(msg)
    got = broker.drain_for("orc")
    assert len(got) == 1
    assert got[0].recipient == "orc"
