#!/usr/bin/env bash
##===- utils/run-docker.sh - start docker container ----------*- Script -*-===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Make it easier to run tests in a clean testing environment (the same as the
# nightly tests) to tease out implicit dependencies on your local environment.
#
# Assumes that you've got a working docker set up which your user is authorized
# to invoke. If you lack this, follow these guides:
# 1) https://docs.docker.com/engine/install/ubuntu/
# 2) https://docs.docker.com/engine/install/linux-postinstall/
#       "Manage Docker as a non-root user"
#       "Configure Docker to start on boot"
#
##===----------------------------------------------------------------------===##

CMD=${1:-"./utils/run-tests-docker.sh"}
VER=${2:-"v5"}
REPO_ROOT=$(cd "$(dirname "$BASH_SOURCE[0]")/.." && pwd)

cd $REPO_ROOT
docker run -it --rm -v $REPO_ROOT:$REPO_ROOT -u $UID:$(id -g) -w $REPO_ROOT \
  ghcr.io/circt/images/circt-integration-test:$VER $CMD
