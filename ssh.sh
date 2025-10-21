#!/usr/bin/env bash
# run like `source ssh.sh` on tmp runpod, after sending your private key and this script

# start ssh agent, add key, go to /workspace
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519_gh

if [[ -n "$SSH_CONNECTION" && -d /workspace/ ]]; then
  echo "🐧 Running on remote runpod with storage attached - moving to /workspace"
  cd /workspace
fi
