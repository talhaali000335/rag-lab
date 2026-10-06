#!/usr/bin/env bash
# Runs ON the EC2 server (sent by GitHub Actions through AWS SSM).
# Expects these environment variables: IMAGE, REGISTRY, AWS_REGION, COMPOSE_B64, CADDY_B64
set -euo pipefail
export HOME=/root
export PATH="$PATH:/usr/local/bin:/snap/bin"

cd /opt/raglab
echo "$COMPOSE_B64" | base64 -d > docker-compose.prod.yml
echo "$CADDY_B64" | base64 -d > Caddyfile
echo "IMAGE=$IMAGE" > .env

aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker compose -f docker-compose.prod.yml pull web
docker compose -f docker-compose.prod.yml up -d --remove-orphans

for i in $(seq 1 30); do
  if docker compose -f docker-compose.prod.yml exec -T web python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=3)" >/dev/null 2>&1; then
    echo "healthy: $IMAGE"
    docker image prune -f >/dev/null
    exit 0
  fi
  sleep 3
done

echo "web did not become healthy; last logs:"
docker compose -f docker-compose.prod.yml logs web --tail 50
exit 1