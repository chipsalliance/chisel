#!/bin/bash
set -e

git config --global user.email "schuyler.eldridge@gmail.com"
git config --global user.name "seldridge"
git config --global push.default simple

sbt docs/publishMicrosite
