"""Normalise TopiOCQA / OR-QuAC into a common conversation format:
[{conv_id, turns:[{q, a, gold, topic, section, shift, rewrite}]}]
shift: 0 = same topic, 1 = new topic, 2 = return to a topic seen earlier in the conversation."""
import json, os, sys, ast, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")


def topiocqa(split):
    rows = [json.loads(l) for l in open(os.path.join(D, f"topiocqa_{split}.jsonl"), encoding="utf8")]
    convs = collections.OrderedDict()
    for r in rows:
        convs.setdefault(r["Conversation_no"], []).append(r)
    out = []
    for cid, turns in convs.items():
        turns.sort(key=lambda r: r["Turn_no"])
        seen, prev, T = set(), None, []
        for r in turns:
            topic = r["Topic"]
            shift = 0 if prev is None or topic == prev else (2 if topic in seen else 1)
            seen.add(topic); prev = topic
            T.append(dict(q=r["Question"], a=r["Answer"], gold=[int(r["Gold_passage"]["id"].split(":")[1])],
                          topic=topic, section=r["Topic_section"], shift=shift, rewrite=None))
        out.append(dict(conv_id=f"topiocqa-{split}-{cid}", turns=T))
    return out


def orquac(split):
    # preprocessed/{split}.txt: one json per line with qid "C_xxx_q#n", question history and rewrite
    rows = [json.loads(l) for l in open(os.path.join(D, "orquac", f"{split}.txt"), encoding="utf8")]
    qrels = {}
    import gzip
    if os.path.exists(os.path.join(D, "orquac", "qrels.txt.gz")):
        for k, v in json.load(gzip.open(os.path.join(D, "orquac", "qrels.txt.gz"), "rt")).items():
            qrels[k] = [p for p, s in v.items() if s > 0]
    convs = collections.OrderedDict()
    for r in rows:
        cid, n = r["qid"].rsplit("_q#", 1)
        convs.setdefault(cid, []).append((int(n), r))
    out = []
    for cid, turns in convs.items():
        turns.sort(key=lambda x: x[0])
        T = []
        for n, r in turns:
            ans = r.get("answer", {}).get("text", "") if isinstance(r.get("answer"), dict) else ""
            T.append(dict(q=r["question"], a=ans if ans != "CANNOTANSWER" else "",
                          gold=qrels.get(r["qid"], []), topic=None, section=None, shift=0,
                          rewrite=r.get("rewrite")))
        out.append(dict(conv_id=f"orquac-{split}-{cid}", turns=T))
    return out


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "topiocqa":
        for s, name in [("train", "train"), ("valid", "dev")]:
            c = topiocqa(s)
            json.dump(c, open(os.path.join(D, f"topiocqa_{name}.json"), "w"))
            sh = collections.Counter(t["shift"] for x in c for t in x["turns"])
            print(name, len(c), sum(len(x["turns"]) for x in c), sh)
    else:
        for s in ["train", "dev", "test"]:
            c = orquac(s)
            json.dump(c, open(os.path.join(D, f"orquac_{s}.json"), "w"))
            print(s, len(c), sum(len(x["turns"]) for x in c), sum(1 for x in c for t in x["turns"] if t["gold"]))
