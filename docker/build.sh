#!/usr/bin/env bash
set -euo pipefail

TAG="base"
PUSH=false
NO_CACHE=false
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  echo "Usage: bash docker/build.sh [--tag <tag>] [--push] [--no-cache]"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --tag)
        TAG="${2:-}"
        if [[ -z "${TAG}" ]]; then
          echo "Error: --tag requires a value"
          exit 1
        fi
        shift 2
        ;;
      --push)
        PUSH=true
        shift
        ;;
      --no-cache)
        NO_CACHE=true
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown argument: $1"
        usage
        exit 1
        ;;
    esac
  done
}

build_image() {
  local image="ghcr.io/serene/mowing-terrain-seg:${TAG}"
  local build_args=()

  if [[ "${NO_CACHE}" == "true" ]]; then
    build_args+=(--no-cache)
  fi

  echo "Building ${image}"
  docker build \
    "${build_args[@]}" \
    -t "${image}" \
    -f "${REPO_ROOT}/docker/Dockerfile" \
    "${REPO_ROOT}"

  if [[ "${PUSH}" == "true" ]]; then
    echo "Pushing ${image}"
    docker push "${image}"
  fi

  echo "Done: ${image}"
}

parse_args "$@"
build_image
