#!/bin/bash
set -e

DOCKERHUB_USER="${DOCKERHUB_USER:-hagu5}"
IMAGE_NAME="ml-service"

SHA=$(git rev-parse --short HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD | sed 's/[^a-zA-Z0-9._-]/-/g')
TAG="${BRANCH}-${SHA}"

echo "Building image: ${IMAGE_NAME}:${TAG}"

docker build -t "${IMAGE_NAME}:${TAG}" -t "${IMAGE_NAME}:latest" .

echo "Tagging for DockerHub..."
docker tag "${IMAGE_NAME}:latest" "${DOCKERHUB_USER}/${IMAGE_NAME}:latest"
docker tag "${IMAGE_NAME}:${TAG}" "${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"

echo "Pushing to DockerHub..."
docker push "${DOCKERHUB_USER}/${IMAGE_NAME}:latest"
docker push "${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"

echo "Done: ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"
