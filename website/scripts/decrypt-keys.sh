#!/bin/sh

openssl aes-256-cbc -K $encrypted_5a031e7fc265_key -iv $encrypted_5a031e7fc265_iv -in travis-deploy-key.enc -out travis-deploy-key -d
chmod 600 travis-deploy-key
cp travis-deploy-key ~/.ssh/id_rsa
