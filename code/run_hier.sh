#!/bin/bash
# 层级意图检索：验证集确认 -> 最终 3 种子 -> 机制消融。单卡串行。
cd "$(dirname "$0")"
L=../logs
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
log() { echo "[$(date +%T)] $*" | tee -a $L/run_hier.log; }
job() {
  local n=$1; shift
  if ls ../runs/$n/eval_*.json >/dev/null 2>&1; then log "skip $n"; return; fi
  log "start $n $*"
  python train.py --name $n "$@" > $L/$n.log 2>&1 && log "done $n" || log "FAILED $n"
}
until [ -f ../data/docs_topiocqa/emb.npy ]; do sleep 60; done
# 内部验证集（设计确认，dev 不参与）
job V_hier        --kind concat --two_level --epochs 2 --holdout 200 --eval_split val
job V_hier_shared --kind concat --two_level --shared_level --epochs 2 --holdout 200 --eval_split val
# 最终：3 个种子
for s in 42 43 44; do job H_hier_s$s --kind concat --two_level --epochs 2 --seed $s; done
# 机制消融（2 epoch，seed 42）
job H_shared_s42 --kind concat --two_level --shared_level --epochs 2 --seed 42
job H_nodocloss_s42 --kind concat --two_level --lam_doc 0 --epochs 2 --seed 42
log "ALL DONE"
