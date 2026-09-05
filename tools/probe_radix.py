#!/usr/bin/env python3
"""Does an intervening other-session request kill this session's prefix cache,
and does it depend on context length?

Runs against a LIVE SGLang server. No restart, no config change.

    python3 probe_radix.py --url http://localhost:8010

Two sessions (A, B) that share no prefix, probed at a SHORT and a LONG
context. The mamba-pressure hypothesis predicts short survives alternation
and long does not; if both survive, or both die, the cause is elsewhere.

Watch the server log alongside:
    grep -E "Prefill batch|mamba evictable" <serverlog> | tail -40
"""
import argparse, json, sys, time, urllib.request


def gen(url, prompt, timeout=900):
    req = urllib.request.Request(
        url + "/generate",
        data=json.dumps(
            {"text": prompt,
             "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    dt = time.perf_counter() - t0
    meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
    return dt, meta.get("prompt_tokens"), meta.get("cached_tokens")


def filler(tag, n_words):
    """Deterministic and distinct per tag: same tag => identical prefix."""
    return " ".join(f"{tag}{i}" for i in range(n_words))


def run_round(url, words, label):
    A = filler("alpha", words)
    B = filler("bravo", words)   # shares nothing with A
    steps = [
        ("A1 cold A",                 A + " Q1"),
        ("A2 A again, no gap",        A + " Q2"),
        ("B1 other session",          B + " Q1"),
        ("A3 back to A  <-- KEY",     A + " Q3"),
    ]
    print(f"\n=== {label}: ~{words} words ===")
    rows = []
    for name, prompt in steps:
        try:
            dt, ptok, ctok = gen(url, prompt)
        except Exception as e:
            print(f"  {name:24s} FAILED: {e}", file=sys.stderr)
            rows.append((name, None, None, None))
            continue
        pct = (100.0 * ctok / ptok) if (ptok and ctok is not None) else None
        rows.append((name, dt, ptok, ctok))
        print(f"  {name:24s} {dt:7.2f}s  prompt={ptok:<7} cached={ctok:<7}"
              f" {'' if pct is None else f'({pct:.1f}%)'}")
    return rows


def verdict(rows):
    """A2 = same session repeat, A3 = repeat after an intervening session."""
    by = {r[0].split()[0]: r for r in rows}
    a2, a3 = by.get("A2"), by.get("A3")
    if not a2 or not a3 or a2[2] is None or a3[2] is None:
        return "incomplete"
    a2_hit = (a2[3] or 0) > 0.5 * (a2[2] or 1)
    a3_hit = (a3[3] or 0) > 0.5 * (a3[2] or 1)
    if a2_hit and a3_hit:
        return "survives alternation"
    if a2_hit and not a3_hit:
        return "KILLED by the other session"
    return "no hit even without a gap"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8010")
    ap.add_argument("--short", type=int, default=1500, help="words, short round")
    ap.add_argument("--long", type=int, default=20000, help="words, long round")
    a = ap.parse_args()

    short_rows = run_round(a.url, a.short, "SHORT")
    long_rows = run_round(a.url, a.long, "LONG")

    sv, lv = verdict(short_rows), verdict(long_rows)
    print("\n" + "=" * 62)
    print(f"  SHORT context : {sv}")
    print(f"  LONG  context : {lv}")
    print("=" * 62)
    print("""
  short survives, long killed -> length-dependent. Consistent with mamba
                                 checkpoints being evicted on the long path.
  both killed                 -> not length-dependent. Mamba pool pressure is
                                 NOT the cause; look elsewhere.
  both survive                -> alternation alone is innocent; the real
                                 trigger is something else (Stop, compaction,
                                 a changed system prompt).
  "no hit even without a gap" -> caching is broken outright, independent of
                                 sessions.
""")


if __name__ == "__main__":
    main()
