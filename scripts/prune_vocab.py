"""Shrink a Moonshine checkpoint's word table: keep only the tokens our text uses.

Kept: every token that occurs when the given texts are tokenized, every single-character
token (so any word can still be spelled), and the special tokens. Writes three variants:

  OUT/untied    input and output tables both pruned, still separate (no other change)
  OUT/tied_in   one shared table: the pruned input table
  OUT/tied_out  one shared table: the pruned output table

Each has kept_ids.json (model id i = original id kept[i]), which engine.py reads to decode.
usage: prune_vocab.py MODEL OUT TEXT_FILE ... [--check MANIFEST ...]
TEXT_FILE: a manifest.jsonl (its "text") or a plain text file (one text per line)."""
import argparse, copy, json, os
import torch


def texts(path):
    for l in open(path):
        if not l.strip():
            continue
        yield json.loads(l)["text"] if path.endswith(".jsonl") else l.strip()


def counts(m):
    tot = sum(p.numel() for p in m.parameters())
    emb = {id(p): p.numel() for p in (m.get_input_embeddings().weight, m.proj_out.weight)}
    v = m.get_input_embeddings().weight.numel()
    return tot, tot - sum(emb.values()) + v, tot - sum(emb.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("text", nargs="+")
    ap.add_argument("--check", nargs="*", default=[])
    a = ap.parse_args()
    from transformers import AutoTokenizer, MoonshineStreamingForConditionalGeneration as M
    tok = AutoTokenizer.from_pretrained(a.model)
    base = M.from_pretrained(a.model).eval()
    V = base.get_input_embeddings().weight.shape[0]
    used = set()
    for f in a.text:
        n = 0
        for t in texts(f):
            used.update(tok(t, add_special_tokens=False)["input_ids"]); n += 1
        print(f"{f}: {n} texts, {len(used)} distinct tokens so far", flush=True)
    toks = tok.convert_ids_to_tokens(list(range(V)))
    chars = {i for i, s in enumerate(toks) if s is not None and len(s.replace("▁", "")) <= 1}
    cfg = base.config
    special = set(tok.all_special_ids) | {getattr(cfg, k) for k in ("decoder_start_token_id", "eos_token_id",
                                                                     "bos_token_id", "pad_token_id") if getattr(cfg, k, None) is not None}
    keep = sorted(i for i in used | chars | special if i < V)
    new = {o: i for i, o in enumerate(keep)}
    print(f"vocabulary {V} -> {len(keep)} kept ({len(used)} used in text, {len(chars)} single-character, {len(special)} special)")
    for f in a.check:   # how often a test reference needs a token we dropped (information only)
        tot = miss = 0
        for t in texts(f):
            ids = tok(t, add_special_tokens=False)["input_ids"]; tot += len(ids); miss += sum(i not in new for i in ids)
        print(f"check {f}: {miss} of {tot} reference tokens dropped ({100 * miss / max(1, tot):.2f}%)")
    E = base.get_input_embeddings().weight.detach(); P = base.proj_out.weight.detach()
    cos = torch.nn.functional.cosine_similarity(E[keep], P[keep], dim=-1)
    print(f"input vs output table, kept rows: cosine mean {cos.mean():.3f}, relative difference "
          f"{float((E[keep] - P[keep]).norm() / P[keep].norm()):.3f}")
    print(f"original: total {counts(base)[0]:,}; word table once {counts(base)[1]:,}; without word table {counts(base)[2]:,}")
    idx = torch.tensor(keep)
    for name, src in (("untied", None), ("tied_in", E), ("tied_out", P)):
        m = copy.deepcopy(base)
        c = m.config
        emb = m.get_input_embeddings()
        emb.weight = torch.nn.Parameter((src if src is not None else E)[idx].clone())
        emb.num_embeddings = len(keep)
        if emb.padding_idx is not None:
            emb.padding_idx = new[emb.padding_idx]
        m.proj_out.weight = torch.nn.Parameter((src if src is not None else P)[idx].clone())
        m.proj_out.out_features = len(keep)
        if getattr(m.proj_out, "bias", None) is not None:
            m.proj_out.bias = torch.nn.Parameter(m.proj_out.bias.detach()[idx].clone())
        c.vocab_size = len(keep)
        for obj in (c, m.generation_config):
            for k in ("decoder_start_token_id", "eos_token_id", "bos_token_id", "pad_token_id"):
                if getattr(obj, k, None) is not None:
                    setattr(obj, k, new[getattr(obj, k)])
        if src is not None:
            c.tie_word_embeddings = True
            m.proj_out.weight = m.get_input_embeddings().weight
        d = f"{a.out}/{name}"
        m.save_pretrained(d); tok.save_pretrained(d)
        json.dump(keep, open(f"{d}/kept_ids.json", "w"))
        r = M.from_pretrained(d)   # reload: the saved checkpoint is what gets scored
        assert r.get_input_embeddings().weight.shape[0] == len(keep)
        if src is not None:
            assert r.proj_out.weight.data_ptr() == r.get_input_embeddings().weight.data_ptr(), "tie lost on reload"
        tot, once, body = counts(r)
        print(f"params {name}: total {tot:,}; word table once {once:,}; without word table {body:,}")


if __name__ == "__main__":
    main()
