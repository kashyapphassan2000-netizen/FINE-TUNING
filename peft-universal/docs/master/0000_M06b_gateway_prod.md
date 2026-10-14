# LAB MANUAL — PART 6B: The Production Front Door — NGINX, TLS, logging, retries

:::why Concept (what, why, when)
Part 6 put the gateway (API keys, per-user limits, metering) in front of vLLM. On the public internet you also need a **front door**: HTTPS with certificates that renew themselves, a first rate limit per IP, structured logs of every request, and automatic retries on another replica when one drops. NGINX does all four. **When:** before you publish the URL to anyone outside your team.
:::

:::honest What was executed
`serving/nginx/nginx.conf` was loaded by **nginx 1.24** with a test domain and a self-signed certificate: `nginx -t` passed (with payload logging off and on), and a live test showed the HTTP→HTTPS redirect, TLS, **failover from a dead replica to a live one**, the JSON access log recording both attempts, and rate limiting. Let's Encrypt issuance itself was **not** executed (it needs a real domain pointing at a real server).
:::

```text the chain
client ──HTTPS──> NGINX (TLS, per-IP limit, JSON logs, retry on another replica)
                    └──> gateway replicas (API keys, per-key limits, metering)
                            └──> vLLM replicas
```

## SECTION 1 — Read the configuration

**WHERE:** VS Code, `serving/nginx/nginx.conf`

**GOAL:** know what every block does before you run it

@file serving/nginx/nginx.conf

|Block|What it does|
|---|---|
|`log_format llm_json`|one JSON line per request: id, client, status, duration, **which replica(s) answered**, their status|
|`llm_payload` (commented out)|optional logging of prompt bodies — off by default (privacy)|
|`limit_req_zone … rate=10r/s`|per-IP limit; `burst=20` absorbs short spikes; excess gets HTTP 429|
|`upstream llm_gateway`|the list of gateway replicas; `max_fails=3 fail_timeout=30s` takes a failing one out of rotation|
|`server { listen 80 }`|serves the Let's Encrypt challenge and redirects everything else to HTTPS|
|`ssl_certificate …/live/YOUR_DOMAIN/`|the certificate files certbot writes|
|`location /admin/`|key issuance and usage: only from your private network|
|`proxy_buffering off`, `proxy_read_timeout 300s`|streaming answers flow token by token; long generations are not cut off|
|`proxy_next_upstream … non_idempotent`, `tries 2`, `timeout 10s`|if a replica errors or is down, retry once on another replica|

:::warn Watch out: what a retry costs
Chat requests are POSTs, which NGINX never retries unless told to (`non_idempotent`). Enabling it means a request that failed on a dying replica is **re-run** on another: the user gets an answer instead of an error, but you pay for the tokens twice when the first replica had already started. Streams that already began sending are never retried. Keep `tries` at 2.
:::

## SECTION 2 — Point a domain at the server

**WHERE:** your DNS provider's website

**GOAL:** `YOUR_DOMAIN` resolves to the serving machine

Step 1. Create an **A record**: name = `api` (for `api.example.com`), value = the serving VM's public IP.

Step 2. Wait a few minutes.

Step 3. In a terminal, type and press Enter:

```any
nslookup YOUR_DOMAIN
```

**You should see:** the VM's public IP.

Step 4. In the provider's firewall/security group, open ports **80** and **443** to the internet, and **close 8080** (the gateway is now internal).

:::check
- `nslookup` returns the right IP; ports 80/443 open, 8080 closed from outside.
:::

## SECTION 3 — First certificate, then automatic renewal

**WHERE:** the serving VM, project folder

**GOAL:** HTTPS with a real certificate that renews itself

@file scripts/tls_bootstrap.sh

Step 1. Type and press Enter (replace both values):

```gpu
bash scripts/tls_bootstrap.sh YOUR_DOMAIN YOUR_EMAIL
```

**You should see:** certbot reporting a successful certificate, then `TLS ready: https://YOUR_DOMAIN/healthz`.

Step 2. Type and press Enter:

```gpu
curl -s https://YOUR_DOMAIN/healthz
```

**You should see:** the gateway's health JSON — over HTTPS, with no certificate warning.

Renewal is automatic: the `certbot` container tries `certbot renew` twice a day (it only renews certificates close to expiry), and the `nginx` container reloads every six hours to pick the new files up without dropping connections.

|If you see…|It means…|Do this next|
|---|---|---|
|certbot: "Connection refused" / "Timeout during connect"|port 80 is not reachable from the internet|open port 80; check the A record|
|certbot: "too many certificates already issued"|Let's Encrypt rate limit|wait, and test with certbot's `--staging` flag first next time|
|browser warns about the certificate|the throwaway self-signed certificate is still loaded|`docker compose ... exec nginx nginx -s reload`|

:::check
- `https://YOUR_DOMAIN/healthz` works without warnings.
:::

## SECTION 4 — Run the full production stack

**WHERE:** serving VM

**GOAL:** NGINX + certbot in front of gateway + vLLM + Prometheus

@file serving/docker-compose.prod.yml

Step 1. Type and press Enter:

```gpu
docker compose version
```

**You should see:** v2.24 or newer (the overlay uses `!reset` to un-publish the gateway port).

Step 2. Type and press Enter:

```gpu
docker compose -f serving/docker-compose.yml -f serving/docker-compose.prod.yml up -d
```

Step 3. Type and press Enter:

```gpu
docker compose -f serving/docker-compose.yml -f serving/docker-compose.prod.yml ps
```

**You should see:** `nginx`, `certbot`, `gateway`, `vllm`, `prometheus` all running.

:::check
- All five services are up, and only ports 80/443 are published.
:::

## SECTION 5 — What the logs and retries look like (real test)

@lab nginx_frontdoor

Read block 3: the first attempt went to the dead replica (`502`), NGINX retried on the live one (`200`), the user saw only the `200`, and the log kept both — which is how you find a failing replica before users complain.

Step 1. To watch live traffic, type and press Enter:

```gpu
docker compose -f serving/docker-compose.yml -f serving/docker-compose.prod.yml exec nginx tail -f /var/log/nginx/access.json
```

## SECTION 6 — Payload logging, only if you must

**WHERE:** `serving/nginx/nginx.conf`

**GOAL:** log prompt bodies for debugging, knowingly

Step 1. Uncomment the two lines under `OPTIONAL payload logging`.

Step 2. Uncomment the `access_log /var/log/nginx/payload.json llm_payload;` line inside `location /v1/`.

Step 3. Reload: `docker compose ... exec nginx nginx -s reload`.

:::warn Watch out: prompts are personal data
Logging bodies means storing what users typed — names, account numbers, health details. Decide a retention period, restrict who can read the file, delete it on schedule, and tell users in your terms. Turn it off again when the debugging is done.
:::

## 6B.7 Kong instead of NGINX?

Kong (built on NGINX) adds API-gateway features — key management, plugins, a database-backed admin API — that overlap with this project's gateway. If your company already runs Kong, put its rate-limiting and key-auth plugins in front of vLLM instead of NGINX + gateway. This book did not execute a Kong configuration, so none is shipped; the NGINX setup above is the tested one.

## 6B.8 Common mistake at this level

**Leaving the old port open.** Teams add NGINX and TLS but the gateway's port 8080 is still published to the internet, so anyone can bypass HTTPS, the per-IP limit and the logs. **How to spot it:** from another network, `curl http://SERVER_IP:8080/healthz` must fail. The production overlay removes the published port; the firewall should block it too.
