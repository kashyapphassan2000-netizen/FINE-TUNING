#!/usr/bin/env bash
# First Let's Encrypt certificate for the NGINX front door (renewals are automatic afterwards).
# Requirements: YOUR_DOMAIN's DNS A record points at this machine; ports 80 and 443 are open.
# Usage: bash scripts/tls_bootstrap.sh api.example.com you@example.com
set -euo pipefail
DOMAIN=${1:?domain}
EMAIL=${2:?email for expiry notices}
cd "$(dirname "$0")/.."
sed -i "s/YOUR_DOMAIN/${DOMAIN}/g" serving/nginx/nginx.conf
COMPOSE="docker compose -f serving/docker-compose.yml -f serving/docker-compose.prod.yml"
# 1. a throwaway self-signed certificate so NGINX can start and serve the ACME challenge
$COMPOSE run --rm --entrypoint sh certbot -c "mkdir -p /etc/letsencrypt/live/${DOMAIN} && \
  openssl req -x509 -nodes -newkey rsa:2048 -days 1 -subj /CN=localhost \
  -keyout /etc/letsencrypt/live/${DOMAIN}/privkey.pem -out /etc/letsencrypt/live/${DOMAIN}/fullchain.pem"
$COMPOSE up -d nginx
# 2. replace it with the real certificate (HTTP-01 challenge through NGINX's port 80)
$COMPOSE run --rm --entrypoint sh certbot -c "rm -rf /etc/letsencrypt/live/${DOMAIN} /etc/letsencrypt/archive/${DOMAIN} /etc/letsencrypt/renewal/${DOMAIN}.conf"
$COMPOSE run --rm certbot certonly --webroot -w /var/www/certbot -d "${DOMAIN}" --email "${EMAIL}" --agree-tos --no-eff-email
$COMPOSE exec nginx nginx -s reload
echo "TLS ready: https://${DOMAIN}/healthz"
