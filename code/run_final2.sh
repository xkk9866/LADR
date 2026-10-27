#!/bin/bash
# Final experiment plan for the frozen PRISM configuration (v3: 256-token text budget + hierarchical intent memory,
# dual-stream predictive coding, surprise-gated multi-timescale heads, result tokens). One job at a time.
cd "$(dirname "$0")"
L=../logs; mkdir -p $L ../runs
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
log() { echo "[$(date +%T)] $*" | tee -a $L/run_final2.log; }
job() {
  local n=$1; shift
  if ls ../runs/$n/eval_*.json >/dev/null 2>&1; then log "skip $n"; return; fi
  for try in 1 2; do
    log "start $n $*"
    python train.py --name $n "$@" > $L/$n.log 2>&1 && { log "done $n"; return; }
    log "FAILED $n (try $try)"; sleep 30
  done
}

# ---- main (2 epochs, full LLM teacher) ----
for s in 42 43 44; do
  job F_prism_s$s  --kind prism  --epochs 2 --seed $s
  job F_concat_s$s --kind concat --epochs 2 --seed $s
done
job F_views_s42 --kind prism --epochs 2 --views --text_budget 384
job F_hier      --kind hier  --epochs 2
job F_gru       --kind gru   --epochs 2

# ---- ablations (1 epoch, reference A_full) ----
job A_full          --kind prism
job A_no_surprise   --kind prism --no_surprise
job A_no_pred       --kind prism --no_pred
job A_no_shift      --kind prism --no_shift
job A_no_mem        --kind prism --no_mem
job A_single_stream --kind prism --single_stream
job A_no_block      --kind prism --no_block
job A_one_scale     --kind prism --n_scales 1
job A_no_order      --kind prism --no_order
job A_mlp_pred      --kind prism --mlp_pred
job A_no_return     --kind prism --no_return
job A_no_feedback   --kind prism --no_feedback
job A_no_kd         --kind prism --no_kd
job A_concat        --kind concat

# ---- OR-QuAC ----
if [ ! -f ../data/index_orquac/DONE ]; then
  python encode_corpus.py orquac > $L/enc_orquac.log 2>&1 && python retrieval.py orquac > $L/int8_orquac.log 2>&1 && touch ../data/index_orquac/DONE
fi
log "orquac index ready"
[ -f ../runs/zeroshot_orquac.json ] || python prepare_aux.py zeroshot orquac > $L/zs_orquac.log 2>&1
python prepare_aux.py teacher orquac > $L/teacher_orquac.log 2>&1
log "orquac aux ready"
job O_prism  --kind prism  --data orquac --epochs 2
job O_concat --kind concat --data orquac --epochs 2
for m in prism concat; do
  [ -f ../runs/F_${m}_s42/eval_orquac_test.json ] || python train.py --name F_${m}_s42 --kind $m --eval_only --eval_data orquac > $L/x_$m.log 2>&1
  [ -f ../runs/O_$m/eval_topiocqa_dev.json ] || python train.py --name O_$m --kind $m --data orquac --eval_only --eval_data topiocqa > $L/x_O_$m.log 2>&1
done
log "ALL DONE"
