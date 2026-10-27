#!/bin/bash
# Full experiment pipeline on the local GPU. Two training slots run in parallel; stages are serial.
cd "$(dirname "$0")"
L=../logs; mkdir -p $L ../runs
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
log() { echo "[$(date +%T)] $*" | tee -a $L/run_all.log; }

wait_slots() { while [ $(jobs -rp | wc -l) -ge ${1:-2} ]; do sleep 20; done; }
job() {  # job <name> <args...>   (skips finished runs)
  local n=$1; shift
  if { ls ../runs/$n/eval_*.json >/dev/null 2>&1 || [ -f ../runs/$n/.started ]; } && [ -z "$FORCE" ]; then log "skip $n"; return; fi
  wait_slots 1; mkdir -p ../runs/$n; touch ../runs/$n/.started; log "start $n $*"
  ( python train.py --name $n "$@" > $L/$n.log 2>&1; rc=$?; [ $rc -ne 0 ] && rm -f ../runs/$n/.started; log "done $n rc=$rc" ) &
  sleep 30
}

# ---------- stage 0: wait for the TopiOCQA index ----------
until [ $(ls ../data/index_topiocqa/emb_*.npy 2>/dev/null | wc -l) -ge 27 ]; do sleep 60; done
log "topiocqa index complete"

# ---------- stage 1: zero-shot baselines + hard negatives + result feedback ----------
[ -f ../runs/zeroshot_topiocqa.json ] || python prepare_aux.py zeroshot topiocqa > $L/zs_topiocqa.log 2>&1
[ -f ../runs/zeroshot_topiocqa.json ] || { log "zeroshot FAILED"; exit 1; }
log "zeroshot topiocqa done"



# ---------- stage 2: main models on TopiOCQA ----------
job prism_full   --kind prism
job concat       --kind concat
job gru          --kind gru
job hier         --kind hier
# ---------- stage 3: ablations ----------
job abl_no_surprise   --kind prism --no_surprise
job abl_no_gate_attn  --kind prism --no_gate_attn
job abl_no_gate_write --kind prism --no_gate_write
job abl_no_order      --kind prism --no_order
job abl_mlp_pred      --kind prism --mlp_pred
job abl_one_scale     --kind prism --n_scales 1 --m 12
job abl_no_feedback   --kind prism --no_feedback
job abl_no_kd         --kind prism --no_kd
job abl_no_shift      --kind prism --no_shift
job abl_no_pred       --kind prism --no_pred
# ---------- stage 4: seeds for significance ----------
job prism_s43  --kind prism --seed 43
job prism_s44  --kind prism --seed 44
job concat_s43 --kind concat --seed 43
job concat_s44 --kind concat --seed 44
wait

# ---------- stage 5: OR-QuAC ----------
# OR-QuAC corpus encoding (serial: avoids GPU memory oversubscription)
until gzip -t ../data/orquac/all_blocks.txt.gz 2>/dev/null; do sleep 120; done
[ -f ../data/index_orquac/DONE ] || { python encode_corpus.py orquac > $L/enc_orquac.log 2>&1 && touch ../data/index_orquac/DONE; }
log "orquac index done"
python prepare_aux.py zeroshot orquac > $L/zs_orquac.log 2>&1; log "zeroshot orquac done"
python prepare_aux.py teacher orquac > $L/teacher_orquac.log 2>&1
job orq_prism  --kind prism  --data orquac
job orq_concat --kind concat --data orquac
job orq_gru    --kind gru    --data orquac
job orq_hier   --kind hier   --data orquac
wait
# cross-dataset generalisation (zero-shot transfer)
for m in prism_full concat gru hier; do
  python train.py --name $m --kind ${m%_full} --eval_only --eval_data orquac > $L/x_$m.log 2>&1
done
for m in prism concat gru hier; do
  python train.py --name orq_$m --kind $m --data orquac --eval_only --eval_data topiocqa > $L/x_orq_$m.log 2>&1
done
log "ALL DONE"
