"""Cut a Moonshine streaming checkpoint down: fewer layers and a smaller word table, same width.

Every kept weight is the original's, so the result starts close to the original and only has
to recover what the dropped layers did (train.py --init OUT --teacher MODEL).

  layers      --enc 0,1,4,5 keeps those encoder layers (with their sliding windows), --dec 0,2,5
              those decoder layers. The default drops encoder layers 2 and 3, the two that look
              no frames ahead, so the encoder's total look-ahead (16 frames) is unchanged.
  word table  the --vocab most frequent tokens of the given texts as the teacher's tokenizer
              splits them, plus every single-character and byte token (so any word can still
              be spelled; single characters only if the texts use them or they are ASCII) and the special tokens. One shared table (the original input table;
              Tiny's input and output tables are identical).

Writes OUT/ (loadable by Engine and train.py) with kept_ids.json (model id i = original id
kept[i]). A word containing a dropped token is spelled with kept pieces instead
(vocab.Restrict); the share of training tokens that needs this is printed.
usage: shrink.py MODEL OUT TEXT_FILE ... [--enc 0,1,4,5] [--dec 0,2,5] [--vocab 8192]"""
import argparse, collections, copy, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from prune_vocab import texts, counts
from vocab import Restrict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("text", nargs="+")
    ap.add_argument("--enc", default="0,1,4,5"); ap.add_argument("--dec", default="0,2,5")
    ap.add_argument("--vocab", type=int, default=8192)
    a = ap.parse_args()
    from transformers import AutoTokenizer, MoonshineStreamingForConditionalGeneration as M
    tok = AutoTokenizer.from_pretrained(a.model)
    base = M.from_pretrained(a.model).eval()
    V = base.get_input_embeddings().weight.shape[0]
    freq = collections.Counter(); n = 0
    for f in a.text:
        for t in texts(f):
            freq.update(tok(t, add_special_tokens=False)["input_ids"]); n += 1
    print(f"{n} texts, {sum(freq.values()):,} tokens, {len(freq):,} distinct", flush=True)
    toks = tok.convert_ids_to_tokens(list(range(V)))
    # single characters: those our text uses or plain ASCII (Tiny has ~2,600, mostly CJK, that
    # would take a third of an 8k table); bytes spell anything else
    chars = {ch for f in a.text for t in texts(f) for ch in t} | {chr(c) for c in range(32, 127)}
    always = {i for i, s in enumerate(toks) if s is not None and (s.startswith("<0x") or (
        len(s.replace("▁", "")) <= 1 and s.replace("▁", "") in chars | {""}))}
    cfg = base.config
    always |= set(tok.all_special_ids) | {getattr(cfg, k) for k in ("decoder_start_token_id", "eos_token_id",
                                                                     "bos_token_id", "pad_token_id") if getattr(cfg, k, None) is not None}
    always = {i for i in always if i < V}
    keep = set(always)
    for i, _ in freq.most_common():
        if len(keep) >= a.vocab:
            break
        keep.add(i)
    keep = sorted(keep)
    covered = sum(c for i, c in freq.items() if i in set(keep)) / sum(freq.values())
    print(f"vocabulary {V} -> {len(keep)} ({len(always)} always kept); {100 * covered:.2f}% of training tokens kept as is")
    r = Restrict(tok, keep)
    words = collections.Counter(w for f in a.text for t in texts(f) for w in t.split())
    respelt = sum(c for w, c in words.items() if r.encode(w) != tok(w, add_special_tokens=False)["input_ids"])
    print(f"{100 * respelt / sum(words.values()):.2f}% of training words are spelled differently from the teacher's split")

    enc_keep = [int(x) for x in a.enc.split(",")]; dec_keep = [int(x) for x in a.dec.split(",")]
    new = {o: i for i, o in enumerate(keep)}
    c = copy.deepcopy(cfg)
    c.encoder_config.num_hidden_layers = len(enc_keep)
    c.encoder_config.sliding_windows = [cfg.encoder_config.sliding_windows[i] for i in enc_keep]
    c.num_hidden_layers = len(dec_keep)
    c.vocab_size = len(keep); c.tie_word_embeddings = True
    for k in ("decoder_start_token_id", "eos_token_id", "bos_token_id", "pad_token_id"):
        if getattr(c, k, None) is not None:
            setattr(c, k, new[getattr(c, k)])
    m = M(c).eval()
    src = base.state_dict(); dst = m.state_dict(); out = {}
    idx = torch.tensor(keep)
    for k in dst:
        s = k
        for part, kept in ((".encoder.layers.", enc_keep), (".decoder.layers.", dec_keep)):
            if part in k:
                head, rest = k.split(part); j, tail = rest.split(".", 1)
                s = f"{head}{part}{kept[int(j)]}.{tail}"
        w = src[s]
        if w.shape != dst[k].shape:   # the word table
            assert w.shape[0] == V and dst[k].shape[0] == len(keep), k
            w = (src["model.decoder.embed_tokens.weight"] if "proj_out" in k else w)[idx]
        out[k] = w.clone()
    m.load_state_dict(out)
    m.tie_weights()
    m.generation_config = copy.deepcopy(base.generation_config)
    for k in ("decoder_start_token_id", "eos_token_id", "bos_token_id", "pad_token_id"):
        if getattr(m.generation_config, k, None) is not None:
            setattr(m.generation_config, k, new[getattr(m.generation_config, k)])
    os.makedirs(a.out, exist_ok=True)
    m.save_pretrained(a.out); tok.save_pretrained(a.out)
    json.dump(keep, open(f"{a.out}/kept_ids.json", "w"))
    r2 = M.from_pretrained(a.out)
    assert r2.proj_out.weight.data_ptr() == r2.get_input_embeddings().weight.data_ptr(), "tie lost on reload"
    tot, once, body = counts(r2)
    print(f"encoder layers {enc_keep}, decoder layers {dec_keep}")
    print(f"params: total {tot:,}; word table once {once:,}; without word table {body:,}")


if __name__ == "__main__":
    main()
