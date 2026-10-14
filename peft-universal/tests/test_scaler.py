import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serving"))
from scaler import Policy, parse_waiting  # noqa: E402

METRICS = """# HELP vllm:num_requests_waiting Number of requests waiting.
vllm:num_requests_waiting{engine="0",model_name="m"} 7.0
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_requests_waiting{engine="1",model_name="m"} 5.0
"""


def test_parse_waiting():
    assert parse_waiting(METRICS) == 12.0


def test_policy_patience_cooldown_bounds():
    p = Policy(min_nodes=1, max_nodes=2, patience=2, cooldown_s=100)
    assert p.decide(1, 20, now=1000) == 0          # first high reading: wait
    assert p.decide(1, 20, now=1001) == +1         # second: scale up
    assert p.decide(2, 20, now=1002) == 0          # cooldown
    assert p.decide(2, 20, now=1200) == 0          # at max_nodes
    assert p.decide(2, 20, now=1201) == 0
    assert p.decide(2, 0, now=1300) == 0
    assert p.decide(2, 0, now=1301) == -1          # sustained idle: scale down
    assert Policy(min_nodes=1).decide(0, 0, now=5000) == +1   # below minimum
