"""Reference autoscaler for plain VMs (no Kubernetes). Kubernetes users: use serving/k8s/keda-scaledobject.yaml.

Loop: read `vllm:num_requests_waiting` from every node's /metrics -> decide -> create/delete Hyperstack VMs
-> tell the gateway the new upstream list.

Hyperstack calls used: POST /core/virtual-machines and GET /core/virtual-machines/{id} (shown in the
official API quickstart). DELETE path is taken from the 'Delete virtual machine' reference page title;
confirm it in the API reference before enabling scale-down (--allow-delete).
New nodes are prepared over SSH with scripts/serve_node_bootstrap.sh.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from dataclasses import dataclass, field

import httpx

API = "https://infrahub-api.nexgencloud.com/v1"
WAITING = re.compile(r"^vllm:num_requests_waiting\{[^}]*\}\s+([0-9.eE+-]+)", re.M)


@dataclass
class Policy:
    min_nodes: int = 1
    max_nodes: int = 4
    scale_up_waiting: float = 8.0     # avg queued requests per node
    scale_down_waiting: float = 0.5
    patience: int = 3                 # consecutive checks before acting
    cooldown_s: int = 900             # GPU VMs take minutes to boot + load weights
    up_streak: int = 0
    down_streak: int = 0
    last_action: float = field(default=0.0)

    def decide(self, n_nodes: int, avg_waiting: float, now: float) -> int:
        """Return +1, -1 or 0."""
        if now - self.last_action < self.cooldown_s:
            return 0
        self.up_streak = self.up_streak + 1 if avg_waiting >= self.scale_up_waiting else 0
        self.down_streak = self.down_streak + 1 if avg_waiting <= self.scale_down_waiting else 0
        if n_nodes < self.min_nodes:
            return +1
        if self.up_streak >= self.patience and n_nodes < self.max_nodes:
            self.up_streak, self.last_action = 0, now
            return +1
        if self.down_streak >= self.patience and n_nodes > self.min_nodes:
            self.down_streak, self.last_action = 0, now
            return -1
        return 0


def parse_waiting(metrics_text: str) -> float:
    return sum(float(m) for m in WAITING.findall(metrics_text))


class Hyperstack:
    def __init__(self, env_name: str, key_name: str, flavor: str, image: str) -> None:
        self.h = {"api_key": os.environ["HYPERSTACK_API_KEY"]}
        self.env_name, self.key_name, self.flavor, self.image = env_name, key_name, flavor, image

    def create(self, name: str, allowed_cidr: str) -> int:
        rules = [{"direction": "ingress", "protocol": "tcp", "port_range_min": p, "port_range_max": p,
                  "ethertype": "IPv4", "remote_ip_prefix": allowed_cidr} for p in (22, 8000)]
        body = {"name": name, "environment_name": self.env_name, "image_name": self.image,
                "flavor_name": self.flavor, "key_name": self.key_name, "count": 1,
                "assign_floating_ip": True, "security_rules": rules}
        r = httpx.post(f"{API}/core/virtual-machines", headers=self.h, json=body, timeout=60)
        r.raise_for_status()
        return r.json()["instances"][0]["id"]

    def wait_active(self, vm_id: int, timeout_s: int = 1800) -> str:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            inst = httpx.get(f"{API}/core/virtual-machines/{vm_id}", headers=self.h, timeout=30).json()["instance"]
            if inst["status"] == "ACTIVE" and inst.get("floating_ip"):
                return inst["floating_ip"]
            time.sleep(15)
        raise TimeoutError(f"VM {vm_id} not ACTIVE after {timeout_s}s")

    def delete(self, vm_id: int) -> None:
        r = httpx.delete(f"{API}/core/virtual-machines/{vm_id}", headers=self.h, timeout=60)  # VERIFY path
        r.raise_for_status()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", required=True, help="http://gateway:8080")
    ap.add_argument("--nodes", nargs="*", default=[], help="existing 'vm_id=http://ip:8000' entries")
    ap.add_argument("--environment", required=True)
    ap.add_argument("--keypair", required=True)
    ap.add_argument("--flavor", required=True, help="e.g. n3-L40x1 (list flavors in the console/API)")
    ap.add_argument("--image", required=True, help="pick an image whose driver supports CUDA 13 (see guide)")
    ap.add_argument("--ssh-key", required=True)
    ap.add_argument("--allowed-cidr", required=True, help="gateway/office CIDR allowed to reach 22/8000")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--allow-delete", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    nodes = dict(n.split("=", 1) for n in a.nodes)  # vm_id -> url
    hs = None if a.dry_run else Hyperstack(a.environment, a.keypair, a.flavor, a.image)
    pol = Policy()
    admin = {"Authorization": f"Bearer {os.environ['GATEWAY_ADMIN_KEY']}"}
    while True:
        waiting = 0.0
        for url in nodes.values():
            try:
                waiting += parse_waiting(httpx.get(f"{url}/metrics", timeout=5).text)
            except httpx.HTTPError:
                pass
        avg = waiting / max(1, len(nodes))
        action = pol.decide(len(nodes), avg, time.time())
        print(f"nodes={len(nodes)} avg_waiting={avg:.1f} action={action}", flush=True)
        if action and a.dry_run:
            pass
        elif action == +1:
            vm_id = hs.create(f"vllm-{int(time.time())}", a.allowed_cidr)
            ip = hs.wait_active(vm_id)
            subprocess.run(["ssh", "-o", "StrictHostKeyChecking=accept-new", "-i", a.ssh_key, f"ubuntu@{ip}",
                            "bash -s"], stdin=open("scripts/serve_node_bootstrap.sh"), check=True)
            nodes[str(vm_id)] = f"http://{ip}:8000"
        elif action == -1 and a.allow_delete:
            vm_id, _ = nodes.popitem()
            httpx.post(f"{a.gateway}/admin/upstreams", headers=admin, json={"upstreams": list(nodes.values())})
            time.sleep(60)  # drain in-flight requests
            hs.delete(int(vm_id))
        if action == +1 and not a.dry_run:
            httpx.post(f"{a.gateway}/admin/upstreams", headers=admin, json={"upstreams": list(nodes.values())})
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
