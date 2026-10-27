#!/bin/bash
# Final experiment plan (single local GPU, one job at a time to avoid GPU-memory oversubscription).
cd "$(dirname "$0")"
L=../logs; mkdir -p $L ../runs
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
log() { echo "[$(date +%T)] $*" | tee -a $L/run_final.log; }
job() {  # job <name> <args...>; skips finished runs, retries once on failure
  local n=$1; shift
  if ls ../runs/$n/eval_*.json >/dev/null 2>&1; then log "skip $n"; return; fi
  for try in 1 2; do
    log "start $n $*"
    python train.py --name $n "$@" > $L/$n.log 2>&1 && { log "done $n"; return; }
    log "FAILED $n (try $try)"; sleep 30
  done
}

# ---- main models, 2 epochs, 3 seeds for PRISM and history concatenation ----
for s in 42 43 44; do
  job main_prism_s$s  --kind prism  --epochs 2 --seed $s
  job main_concat_s$s --kind concat --epochs 2 --seed $s
done
job main_gru   --kind gru   --epochs 2
job main_hier  --kind hier  --epochs 2
job main_prism_rec --kind prism_rec --epochs 1

# ---- ablations (1 epoch; reference = prism_v2d_e1) ----
job abl_no_surprise  --kind prism --no_surprise
job abl_no_return    --kind prism --no_return
job abl_one_scale    --kind prism --n_scales 1
job abl_no_order     --kind prism --no_order
job abl_mlp_pred     --kind prism --mlp_pred
job abl_no_feedback  --kind prism --no_feedback
job abl_no_kd        --kind prism --no_kd
job abl_no_shift     --kind prism --no_shift
job abl_no_pred      --kind prism --no_pred
job abl_single_stream --kind prism --single_stream
job abl_no_block     --kind prism --no_block
job abl_concat_e1_s43 --kind concat --seed 43
job abl_prism_e1_s43  --kind prism --seed 43

# ---- OR-QuAC: corpus, zero-shot, in-domain training, cross-dataset transfer ----
until gzip -t ../data/orquac/all_blocks.txt.gz 2>/dev/null; do sleep 120; done
if [ ! -f ../data/index_orquac/DONE ]; then
  python encode_corpus.py orquac > $L/enc_orquac.log 2>&1 && python retrieval.py orquac > $L/int8_orquac.log 2>&1 && touch ../data/index_orquac/DONE
fi
log "orquac index ready"
[ -f ../runs/zeroshot_orquac.json ] || python prepare_aux.py zeroshot orquac > $L/zs_orquac.log 2>&1
python prepare_aux.py teacher orquac > $L/teacher_orquac.log 2>&1
log "orquac aux ready"
job orq_prism  --kind prism  --data orquac --epochs 2
job orq_concat --kind concat --data orquac --epochs 2
for m in prism concat; do
  [ -f ../runs/main_${m}_s42/eval_orquac_test.json ] || python train.py --name main_${m}_s42 --kind $m --eval_only --eval_data orquac > $L/x_$m.log 2>&1
  [ -f ../runs/orq_$m/eval_topiocqa_dev.json ] || python train.py --name orq_$m --kind $m --data orquac --eval_only --eval_data topiocqa > $L/x_orq_$m.log 2>&1
done
log "ALL DONE"
