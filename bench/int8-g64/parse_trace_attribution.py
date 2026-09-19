"""Correct in-server attribution from a vLLM torch-profiler trace.

TWO BUGS IN THE FIRST ATTEMPT, both now avoided:
  1. it summed `gpu_user_annotation` events, which WRAP kernels -> double counting
     (shares summed to 77,705%);
  2. it aggregated the WHOLE trace, which includes the ~277 s prefill, so a "per pass"
     figure was really "per prefill + decode".

Method here:
  * locate the decode passes from the annotations: `generation_1(N)` marks a generation
    pass, `generation_0` marks prefill chunks;
  * take the decode window as [first decode ts, last decode ts+dur];
  * sum ONLY `cat == "kernel"` events inside that window;
  * divide by the number of decode passes actually seen.
"""
import gzip, json, glob, os, sys, collections

def load(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def main(trace_dir):
    files = [f for f in glob.glob(os.path.join(trace_dir, "**", "*.json*"), recursive=True)
             if os.path.getsize(f) > 100000]
    if not files:
        print("no substantial trace files"); return 1
    big = max(files, key=os.path.getsize)
    print("trace: %s (%.1f MB)" % (os.path.basename(big), os.path.getsize(big)/1e6))
    d = load(big)
    ev = d.get("traceEvents") or []

    ann = [e for e in ev if e.get("cat") == "gpu_user_annotation"]
    gen = [e for e in ann if "_generation_1(" in str(e.get("name"))]
    pre = [e for e in ann if "_generation_0(" in str(e.get("name"))]
    print("annotations: %d generation passes, %d prefill chunks" % (len(gen), len(pre)))
    if not gen:
        print("no generation annotations -> cannot isolate decode"); return 1

    lo = min(e["ts"] for e in gen)
    hi = max(e["ts"] + e.get("dur", 0) for e in gen)
    print("decode window: %.1f ms  (%d passes -> %.2f ms/pass)" %
          ((hi - lo)/1000.0, len(gen), (hi - lo)/1000.0/len(gen)))

    kern = [e for e in ev
            if e.get("cat") == "kernel" and lo <= e.get("ts", -1) <= hi and e.get("dur", 0) > 0]
    print("kernel events in window: %d of %d total" %
          (len(kern), sum(1 for e in ev if e.get("cat") == "kernel")))
    tot_us = sum(e["dur"] for e in kern)
    passes = len(gen)
    print("kernel time in window: %.1f ms total = %.3f ms/pass" % (tot_us/1000.0, tot_us/1000.0/passes))
    print()

    def bucket(n):
        import re
        for lab, pat in (
            ("verifier_partial", r"_spec_attn_partial"),
            ("verifier_combine", r"_spec_attn_combine|combine_kernel"),
            ("marlin_gemm", r"marlin|Marlin"),
            ("gdn", r"gdn|delta_rule|chunk_fwd|chunk_gated|causal_conv1d|ssm|mamba"),
            ("attention_flash", r"flash_fwd|flash::|paged_attention|attention"),
            ("rmsnorm_silu", r"rms_norm|silu_and_mul|act_fn"),
            ("elementwise", r"elementwise|copy|cat|index|embedding|rope|memset"),
        ):
            if re.search(pat, n, re.IGNORECASE):
                return lab
        return "unclassified"

    by = collections.defaultdict(lambda: [0.0, 0])
    for e in kern:
        b = bucket(e.get("name", ""))
        by[b][0] += e["dur"]; by[b][1] += 1
    print("%-20s %12s %10s %8s %8s" % ("bucket", "total_ms", "ms/pass", "share", "kernels"))
    for b, (us, c) in sorted(by.items(), key=lambda kv: -kv[1][0]):
        print("%-20s %12.1f %10.3f %7.2f%% %8d" %
              (b, us/1000.0, us/1000.0/passes, 100.0*us/tot_us, c))

    print()
    print("top kernels in the decode window:")
    per = collections.defaultdict(lambda: [0.0, 0])
    for e in kern:
        per[e.get("name", "")][0] += e["dur"]; per[e.get("name", "")][1] += 1
    for n, (us, c) in sorted(per.items(), key=lambda kv: -kv[1][0])[:14]:
        print("  %-56s %8.2f ms/pass  (%d calls/pass)" % (n[:56], us/1000.0/passes, round(c/passes)))
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ph2_traces"))
