#!/bin/bash
# Full experiment pipeline on the server, GPU 6 only.
# encoding -> document level -> document store -> two parallel training lanes
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=${PY:-$HOME/research/grag/.venv/bin/python}
L=../logs; mkdir -p $L ../runs
log() { echo "[$(date +%F' '%T)] $*" | tee -a $L/run_ldm.log; }
job() {
  local n=$1; shift
  if ls ../runs/$n/eval_*.json >/dev/null 2>&1; then log "skip $n"; return; fi
  log "start $n $*"
  $PY train.py --name $n "$@" > $L/$n.log 2>&1 && log "done $n" || log "FAILED $n"
}
xeval() {
  local n=$1; local tag=$2; shift 2
  [ -f ../runs/$n/eval_$tag.json ] && { log "skip x $n $tag"; return; }
  log "xeval $n $tag"; $PY train.py --name $n "$@" --eval_only > $L/x_${n}_$tag.log 2>&1 && log "done x $n $tag" || log "FAILED x $n"
}
T2="--kind concat --epochs 2"

# 0) corpus encoding (TopiOCQA, then OR-QuAC) in the background
pgrep -f encode_fast >/dev/null || [ -f ../data/index_topiocqa/DONE ] || { $PY encode_fast.py topiocqa > $L/encode.log 2>&1 & }
until [ -f ../data/index_topiocqa/DONE ]; do sleep 60; done
log "topiocqa index ready"
[ -f ../data/docs_topiocqa/emb.npy ] || $PY build_docs.py topiocqa > $L/build_docs_topiocqa.log 2>&1
[ -f ../data/dstore_topiocqa.npy ] || $PY build_dstore.py topiocqa > $L/dstore_topiocqa.log 2>&1
log "topiocqa document level ready"

laneA() {
  for s in 42 43 44; do
    job L_ours_s$s $T2 --latent --seed $s
    job C_concat_s$s $T2 --seed $s
  done
  job L_joint_s42 $T2 --latent --cond 0
  job L_max_s42 $T2 --latent --expert max
}
laneB() {
  for s in 42 43 44; do
    job H_hint_s$s $T2 --two_level --lam_doc 0 --seed $s
  done
  job L_centroid_s42 $T2 --latent --expert centroid
  job L_shared_s42 $T2 --latent --shared_level
  job L_fixtau_s42 $T2 --latent --learn_tau 0
}
laneA; laneB
log "ALL DONE"
