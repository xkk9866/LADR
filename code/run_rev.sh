#!/bin/bash
# 审稿修订实验队列（单卡串行）
cd "$(dirname "$0")"
L=../logs
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
log() { echo "[$(date +%T)] $*" | tee -a $L/run_rev.log; }
job() {
  local n=$1; shift
  if ls ../runs/$n/eval_*.json >/dev/null 2>&1; then log "skip $n"; return; fi
  log "start $n $*"
  python train.py --name $n "$@" > $L/$n.log 2>&1 && log "done $n" || log "FAILED $n"
}
xeval() {  # xeval <run> <kind-args...> -- cross-dataset evaluation
  local n=$1; local tag=$2; shift 2
  [ -f ../runs/$n/eval_$tag.json ] && { log "skip x $n $tag"; return; }
  log "xeval $n $tag"; python train.py --name $n "$@" --eval_only > $L/x_${n}_$tag.log 2>&1 || log "FAILED x $n"
}
# 1) OR-QuAC 文档层级
[ -f ../data/docs_orquac/emb.npy ] || python build_docs.py orquac > $L/build_docs_orquac.log 2>&1
log "orquac docs ready"
# 2) OR-QuAC 上的 HINT（域内）+ 双向迁移
job O_hint --kind concat --two_level --data orquac --epochs 2
xeval O_hint topiocqa_dev --kind concat --two_level --data orquac --eval_data topiocqa
xeval H_hier_s42 orquac_test --kind concat --two_level --eval_data orquac
# 3) 关键消融补足 3 个种子
for s in 43 44; do
  job H_nodocloss_s$s --kind concat --two_level --lam_doc 0 --epochs 2 --seed $s
  job H_shared_s$s --kind concat --two_level --shared_level --epochs 2 --seed $s
done
log "ALL DONE"
