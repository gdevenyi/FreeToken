#!/usr/bin/env bash
# Sync the port worktree to the chatdnp test tree and run a command there with CUDA hidden.
# usage: srv_test.sh pytest <args>   |  srv_test.sh py <script-on-server> <args>  |  srv_test.sh sh '<command>'
set -u
WT=${WT:?set WT to the local worktree}
SSH="ssh -o BatchMode=yes -J douglas localadmin@chatdnp"
until timeout 25 $SSH 'grep -q BUILDDONE ~/scratch/agents/claude/d1031/setup.log'; do echo "waiting for the server build" >&2; sleep 60; done
rsync -a --delete --exclude='*.so' --exclude=__pycache__ --exclude=.git --exclude=.venv -e "ssh -o BatchMode=yes -J douglas" $WT/python $WT/tests $WT/pyproject.toml $WT/setup.py localadmin@chatdnp:ft-1031-port/ || exit 3
timeout 30 $SSH 'cd ~/ft-1031-port && ln -sfn ~/ft-1031/.venv .venv && cd ~/ft-1031/python && find . -name "*.so" | while read f; do mkdir -p ~/ft-1031-port/python/$(dirname "$f"); cp -n "$f" ~/ft-1031-port/python/"$f"; done'
cmd=$1; shift
case $cmd in
  pytest) timeout 2400 $SSH "cd ~/ft-1031-port && CUDA_VISIBLE_DEVICES= PYTHONPATH=\$PWD/python .venv/bin/python -m pytest -q -p no:cacheprovider --timeout=300 $*" ;;
  py) timeout 1200 $SSH "cd ~/ft-1031-port && CUDA_VISIBLE_DEVICES= PYTHONPATH=\$PWD/python .venv/bin/python $*" ;;
  sh) timeout 1200 $SSH "cd ~/ft-1031-port && CUDA_VISIBLE_DEVICES= PYTHONPATH=\$PWD/python $*" ;;
esac
