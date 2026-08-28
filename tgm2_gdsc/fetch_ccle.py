import os, json, sys
os.environ.setdefault("https_proxy", "http://127.0.0.1:17895")
os.environ.setdefault("http_proxy", "http://127.0.0.1:17895")
import xenaPython as xena

HUB = "https://ucscpublic.xenahubs.net"
EXPR = "ccle/CCLE_DepMap_18Q2_RNAseq_RPKM_20180502"
INFO = "ccle/CCLE_sample_info_file_2012-10-18.txt"
SITE = "ccle/sample_info_primary_site"
DRUG = "ccle/CCLE_NP24.2009_Drug_data_2015.02.24"

samples = xena.dataset_samples(HUB, EXPR, None)
print("CCLE expr samples:", len(samples), samples[:4])

fields = xena.dataset_field(HUB, EXPR)
print("n genes:", len(fields))
tg = [f for f in fields if f.upper() in ("TGM2", "TGM2.1")]
print("TGM2 field:", tg)

vals = xena.dataset_fetch(HUB, EXPR, samples, tg[:1])[0]
out = {s: v for s, v in zip(samples, vals)}
json.dump(out, open("/store/zkyang/tgm2_gdsc/ccle_TGM2_rpkm.json", "w"))
nn = sum(1 for v in out.values() if v is not None and v == v)
print("non-null TGM2 values:", nn)

# CCLE 自带药敏（24 个药）
try:
    ds = xena.dataset_samples(HUB, DRUG, None)
    df = xena.dataset_field(HUB, DRUG)
    print("CCLE drug samples:", len(ds), "fields:", len(df))
    print("drug fields sample:", df[:40])
    json.dump({"samples": ds, "fields": df}, open("/store/zkyang/tgm2_gdsc/ccle_drug_meta.json", "w"))
except Exception as e:
    print("drug err", e)
