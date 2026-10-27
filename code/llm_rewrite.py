"""Context-independent query rewriting with Qwen2.5-3B-Instruct quantised to 4-bit (bitsandbytes NF4).
Used (i) as the LLM-rewrite baseline and (ii) as an offline teacher signal for PRISM (L_kd).
Usage: python llm_rewrite.py topiocqa_dev [topiocqa_train ...]"""
import json, os, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = "Qwen/Qwen2.5-3B-Instruct"
SYS = ("You rewrite the user's last question in an information-seeking conversation into a fully "
       "self-contained search query. Resolve pronouns and ellipsis using the conversation; if the topic "
       "changed, do not carry over the old topic. Output only the rewritten question.")
SHOT = ("Conversation:\nQ: who sang the song mama i'm coming home\nA: Ozzy Osbourne\n"
        "Q: what album was it on\n\nRewrite: what album was the song mama i'm coming home by Ozzy Osbourne on")


def prompt(turns, i, tok, max_hist=6):
    h = []
    for t in turns[max(0, i - max_hist):i]:
        h.append(f"Q: {t['q']}\nA: {t['a'][:200]}")
    conv = "\n".join(h + [f"Q: {turns[i]['q']}"])
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": SHOT + "\n\nConversation:\n" + conv + "\n\nRewrite:"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main(names):
    tok = AutoTokenizer.from_pretrained(M, padding_side="left")
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(M, quantization_config=q, device_map="cuda:0").eval()
    for name in names:
        path = os.path.join(ROOT, "data", f"{name}.json")
        out_path = os.path.join(ROOT, "data", f"{name}.llmrw.json")
        convs = json.load(open(path))
        done = json.load(open(out_path)) if os.path.exists(out_path) else {}
        jobs = [(c["conv_id"], i, prompt(c["turns"], i, tok)) for c in convs for i in range(len(c["turns"]))
                if f"{c['conv_id']}|{i}" not in done]
        jobs.sort(key=lambda x: len(x[2]))
        t0, BS = time.time(), 96
        for b in range(0, len(jobs), BS):
            batch = jobs[b:b + BS]
            enc = tok([x[2] for x in batch], return_tensors="pt", padding=True).to("cuda:0")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
            texts = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for (cid, i, _), t in zip(batch, texts):
                done[f"{cid}|{i}"] = t.strip().split("\n")[0]
            if (b // BS) % 20 == 0:
                json.dump(done, open(out_path, "w"))
                print(name, b + len(batch), "/", len(jobs), f"{time.time()-t0:.0f}s", flush=True)
        json.dump(done, open(out_path, "w"))


if __name__ == "__main__":
    main(sys.argv[1:])
