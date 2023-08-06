#!/usr/bin/env bash
##===- utils/update-docs-dialects.sh - build dialect diagram -*- Script -*-===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Renders the `docs/dialects.dot` diagram using graphviz.
#
##===----------------------------------------------------------------------===##

set -e
DOCS_DIR=$(cd "$(dirname "$BASH_SOURCE[0]")/../docs" && pwd)

# Update the rendered diagrams in the docs.
dot -Tpng $DOCS_DIR/dialects.dot > $DOCS_DIR/includes/img/dialects.png
dot -Tsvg $DOCS_DIR/dialects.dot > $DOCS_DIR/includes/img/dialects.svg
