#!/usr/bin/env bash
# Rebuild both Docker images with the current mlruns/ and push to GHCR.
#
# Run from the project root after training a new model:
#   bash scripts/rebuild_docker.sh
#
# Requires a one-time login:
#   docker login ghcr.io -u <github-username> --password <PAT with write:packages>
set -euo pipefail

API_IMAGE="ghcr.io/yihoolan/mlb-pitch-predictor-api:latest"
STREAMLIT_IMAGE="ghcr.io/yihoolan/mlb-pitch-predictor-streamlit:latest"

echo "Building API image..."
docker build -t "$API_IMAGE" -f Dockerfile .

echo "Building Streamlit image..."
docker build -t "$STREAMLIT_IMAGE" -f Dockerfile.streamlit .

echo "Pushing images to GHCR..."
docker push "$API_IMAGE"
docker push "$STREAMLIT_IMAGE"

echo "Done. Users can now run: docker compose pull && docker compose up"
