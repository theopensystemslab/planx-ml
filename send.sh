#!/usr/bin/env bash
# run like `./send.sh host private_key` on local, to prepare remote to continue setup

if [[ -z "${1-}" ]]; then
    REMOTE="pod"
else
    REMOTE="$1"
fi

if [[ -z "${2-}" ]]; then
    PRIVATE_KEY="id_ed25519_gh"
else
    PRIVATE_KEY="$2"
fi

# move private key (for GitHub), ssh.sh script and env/tmux conf files (if they exist) to remote
scp ~/.ssh/$PRIVATE_KEY "$REMOTE:~/.ssh/id_ed25519_gh"
scp ssh.sh "$REMOTE:ssh.sh"

scp .env "$REMOTE:.env" || true
scp .tmux.conf "$REMOTE:.tmux.conf" || true
