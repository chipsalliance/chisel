#!/bin/sh

openssl aes-256-cbc -K $encrypted_[your_number]_key -iv $encrypted_[your_number]_iv -in travis-deploy-key.enc -out travis-deploy-key -d;
chmod 600 travis-deploy-key;
cp travis-deploy-key ~/.ssh/id_rsa;
