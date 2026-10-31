# Spot check — rationale / evidence / verifier triples

Random sample (seed=42); review by hand.

## test_031534 (arm S2, stratum fn)

**Evidence table**

```
1. handles.nmutant = 282 | typical | toward_benign | 47.8pct
2. malfind.ninjections = 78 | very_high | toward_benign | 99.0pct
3. handles.ntimer = 129 | typical | toward_benign | 48.0pct
4. pslist.avg_threads = 7.5484 | very_low | toward_ransomware | 0.8pct
5. handles.nfile = 681 | typical | toward_benign | 44.0pct
6. malfind.commitCharge = 9215 | very_high | toward_ransomware | 95.5pct
```

**Rationale** ({"verdict": "benign", "probability_stated": 0.054, "confidence_statement": "The verdict is benign but the evidence is mixed, so it carries real uncertainty.", "claims": [{"feature": "handles.nmutant", "direction": "toward_benign", "level": "typical", "value_stated": 282.0, "text": "282 mutant handles is typical and leans toward benign.", "extractor": "given"}, {"feature": "malfind.ninjections", "direction": "toward_benign", "level": "very_high", "value_stated": 78.0, "text": "78 malfind-flagged regions is very high but is treated here as benign-leaning.", "extractor": "given"}, {"feature": "handles.ntimer", "direction": "toward_benign", "level": "typical", "value_stated": 129.0, "text": "129 timer handles is typical and benign-leaning.", "extractor": "given"}, {"feature": "pslist.avg_threads", "direction": "toward_ransomware", "level": "very_low", "value_stated": 7.55, "text": "7.55 threads per process is very low and pushes toward ransomware.", "extractor": "given"}, {"feature": "handles.nfile", "direction": "toward_benign", "level": "typical", "value_stated": 681.0, "text": "681 file handles is typical and benign-leaning.", "extractor": "given"}, {"feature": "malfind.commitCharge)


**Verifier**

- stated verdict/prob: benign / 0.054
- FS=1.000 pass=True precision=1.00 coverage=1.00
- claims:
  - `handles.nmutant` -> supported (failed: none)
  - `malfind.ninjections` -> supported (failed: none)
  - `handles.ntimer` -> supported (failed: none)
  - `pslist.avg_threads` -> supported (failed: none)
  - `handles.nfile` -> supported (failed: none)
  - `malfind.commitCharge` -> supported (failed: none)

## test_051466 (arm S2, stratum tn)

**Evidence table**

```
1. handles.nfile = 695 | typical | toward_benign | 45.0pct
2. dlllist.avg_dlls_per_proc = 39.286 | typical | toward_ransomware | 40.2pct
3. handles.nkey = 729 | typical | toward_ransomware | 47.0pct
4. svcscan.nactive = 123 | typical | toward_ransomware | 75.0pct
5. handles.nsection = 189 | typical | toward_ransomware | 45.0pct
6. handles.nsemaphore = 657 | typical | toward_ransomware | 45.2pct
```

**Rationale** ({"verdict": "benign", "probability_stated": 0.073, "confidence_statement": "The verdict is benign but the evidence is mixed.", "claims": [{"feature": "handles.nfile", "direction": "toward_benign", "level": "typical", "value_stated": 695.0, "text": "695 file handles is typical and benign-leaning.", "extractor": "given"}, {"feature": "dlllist.avg_dlls_per_proc", "direction": "toward_ransomware", "level": "typical", "value_stated": 39.3, "text": "39.3 DLLs per process is typical but pushes toward ransomware.", "extractor": "given"}, {"feature": "handles.nkey", "direction": "toward_ransomware", "level": "typical", "value_stated": 729.0, "text": "729 registry-key handles is typical but pushes toward ransomware.", "extractor": "given"}, {"feature": "svcscan.nactive", "direction": "toward_ransomware", "level": "typical", "value_stated": 123.0, "text": "123 active services is typical but pushes toward ransomware.", "extractor": "given"}, {"feature": "handles.nsection", "direction": "toward_ransomware", "level": "typical", "value_stated": 189.0, "text": "189 section handles is typical but pushes toward ransomware.", "extractor": "given"}, {"feature": "handles.nsemaphore", "direction": "towa)


**Verifier**

- stated verdict/prob: benign / 0.073
- FS=1.000 pass=True precision=1.00 coverage=1.00
- claims:
  - `handles.nfile` -> supported (failed: none)
  - `dlllist.avg_dlls_per_proc` -> supported (failed: none)
  - `handles.nkey` -> supported (failed: none)
  - `svcscan.nactive` -> supported (failed: none)
  - `handles.nsection` -> supported (failed: none)
  - `handles.nsemaphore` -> supported (failed: none)

## test_034844 (arm S2, stratum tp)

**Evidence table**

```
1. malfind.ninjections = 1 | very_low | toward_ransomware | 0.0pct
2. dlllist.avg_dlls_per_proc = 30.054 | very_low | toward_ransomware | 0.8pct
3. handles.nthread = 705 | low | toward_benign | 23.5pct
4. malfind.commitCharge = 1 | very_low | toward_ransomware | 0.0pct
5. pslist.avg_threads = 9.5349 | low | toward_ransomware | 9.4pct
6. handles.nsection = 176 | low | toward_ransomware | 24.0pct
```

**Rationale** ({"verdict": "ransomware", "probability_stated": 0.739, "confidence_statement": "The evidence is mixed but the balance favours ransomware.", "claims": [{"feature": "malfind.ninjections", "direction": "toward_ransomware", "level": "very_low", "value_stated": 1.0, "text": "A single malfind-flagged region is very low and supports the verdict.", "extractor": "given"}, {"feature": "dlllist.avg_dlls_per_proc", "direction": "toward_ransomware", "level": "very_low", "value_stated": 30.1, "text": "30.1 DLLs per process is very low and strongly supports the verdict.", "extractor": "given"}, {"feature": "handles.nthread", "direction": "toward_benign", "level": "low", "value_stated": 705.0, "text": "705 thread handles is low and pushes toward benign.", "extractor": "given"}, {"feature": "malfind.commitCharge", "direction": "toward_ransomware", "level": "very_low", "value_stated": 1.0, "text": "1 committed byte across flagged regions is very low.", "extractor": "given"}, {"feature": "pslist.avg_threads", "direction": "toward_ransomware", "level": "low", "value_stated": 9.53, "text": "9.53 threads per process is low and supports the verdict.", "extractor": "given"}, {"feature": "handles.nsection")


**Verifier**

- stated verdict/prob: ransomware / 0.739
- FS=1.000 pass=True precision=1.00 coverage=1.00
- claims:
  - `malfind.ninjections` -> supported (failed: none)
  - `dlllist.avg_dlls_per_proc` -> supported (failed: none)
  - `handles.nthread` -> supported (failed: none)
  - `malfind.commitCharge` -> supported (failed: none)
  - `pslist.avg_threads` -> supported (failed: none)
  - `handles.nsection` -> supported (failed: none)

## test_055106 (arm S2, stratum tp)

**Evidence table**

```
1. dlllist.avg_dlls_per_proc = 39.081 | typical | toward_ransomware | 34.0pct
2. handles.nmutant = 230 | low | toward_ransomware | 11.0pct
3. handles.nhandles = 7966 | low | toward_ransomware | 9.9pct
4. pslist.avg_threads = 10.108 | typical | toward_ransomware | 31.3pct
5. pslist.avg_handlers = 215.3 | typical | toward_ransomware | 41.4pct
6. ldrmodules.not_in_mem = 43 | low | toward_ransomware | 12.0pct
```

**Rationale** ({"verdict": "ransomware", "probability_stated": 0.909, "confidence_statement": null, "claims": [{"feature": "dlllist.avg_dlls_per_proc", "direction": "toward_ransomware", "level": "typical", "value_stated": 39.1, "text": "39.1 DLLs per process is typical and leans toward ransomware.", "extractor": "given"}, {"feature": "handles.nmutant", "direction": "toward_ransomware", "level": "low", "value_stated": 230.0, "text": "230 mutant handles is low and supports the verdict.", "extractor": "given"}, {"feature": "handles.nhandles", "direction": "toward_ransomware", "level": "low", "value_stated": 7966, "text": "7,966 total handles is low and supports the verdict.", "extractor": "given"}, {"feature": "pslist.avg_threads", "direction": "toward_ransomware", "level": "typical", "value_stated": 10.1, "text": "10.1 threads per process is typical and leans toward ransomware.", "extractor": "given"}, {"feature": "pslist.avg_handlers", "direction": "toward_ransomware", "level": "typical", "value_stated": 215.3, "text": "215.3 handles per process is typical and leans toward ransomware.", "extractor": "given"}, {"feature": "ldrmodules.not_in_mem", "direction": "toward_ransomware", "level": "low", "v)


**Verifier**

- stated verdict/prob: ransomware / 0.909
- FS=1.000 pass=True precision=1.00 coverage=1.00
- claims:
  - `dlllist.avg_dlls_per_proc` -> supported (failed: none)
  - `handles.nmutant` -> supported (failed: none)
  - `handles.nhandles` -> supported (failed: none)
  - `pslist.avg_threads` -> supported (failed: none)
  - `pslist.avg_handlers` -> supported (failed: none)
  - `ldrmodules.not_in_mem` -> supported (failed: none)

## test_032402 (arm S2, stratum tp)

**Evidence table**

```
1. dlllist.avg_dlls_per_proc = 38.075 | low | toward_ransomware | 15.9pct
2. handles.nsemaphore = 605 | low | toward_ransomware | 18.5pct
3. handles.nkey = 695 | typical | toward_ransomware | 37.0pct
4. handles.nmutant = 233 | low | toward_ransomware | 14.0pct
5. handles.nsection = 161 | low | toward_ransomware | 11.0pct
6. dlllist.ndlls = 1523 | low | toward_ransomware | 19.1pct
```

**Rationale** ({"verdict": "ransomware", "probability_stated": 0.931, "confidence_statement": null, "claims": [{"feature": "dlllist.avg_dlls_per_proc", "direction": "toward_ransomware", "level": "low", "value_stated": 38.1, "text": "38.1 DLLs per process is low and supports the verdict.", "extractor": "given"}, {"feature": "handles.nsemaphore", "direction": "toward_ransomware", "level": "low", "value_stated": 605.0, "text": "605 semaphore handles is low and supports the verdict.", "extractor": "given"}, {"feature": "handles.nkey", "direction": "toward_ransomware", "level": "typical", "value_stated": 695.0, "text": "695 registry-key handles is typical and leans toward ransomware.", "extractor": "given"}, {"feature": "handles.nmutant", "direction": "toward_ransomware", "level": "low", "value_stated": 233.0, "text": "233 mutant handles is low and supports the verdict.", "extractor": "given"}, {"feature": "handles.nsection", "direction": "toward_ransomware", "level": "low", "value_stated": 161.0, "text": "161 section handles is low and supports the verdict.", "extractor": "given"}, {"feature": "dlllist.ndlls", "direction": "toward_ransomware", "level": "low", "value_stated": 1523, "text": "1,523 load)


**Verifier**

- stated verdict/prob: ransomware / 0.931
- FS=1.000 pass=True precision=1.00 coverage=1.00
- claims:
  - `dlllist.avg_dlls_per_proc` -> supported (failed: none)
  - `handles.nsemaphore` -> supported (failed: none)
  - `handles.nkey` -> supported (failed: none)
  - `handles.nmutant` -> supported (failed: none)
  - `handles.nsection` -> supported (failed: none)
  - `dlllist.ndlls` -> supported (failed: none)

## test_057797 (arm S2, stratum fn)

**Evidence table**

```
1. dlllist.avg_dlls_per_proc = 38.659 | low | toward_ransomware | 22.4pct
2. handles.nkey = 733 | typical | toward_ransomware | 47.4pct
3. psxview.not_in_deskthrd_false_avg = 0.14634 | typical | toward_benign | 26.0pct
4. svcscan.nactive = 120 | low | toward_ransomware | 24.0pct
5. handles.nthread = 697 | low | toward_benign | 20.0pct
6. ldrmodules.not_in_init_avg = 0.05725 | high | toward_benign | 78.3pct
```

**Rationale** ({"verdict": "benign", "probability_stated": 0.326, "confidence_statement": "The verdict is benign but the evidence is mixed, so it is not certain.", "claims": [{"feature": "dlllist.avg_dlls_per_proc", "direction": "toward_ransomware", "level": "low", "value_stated": 38.7, "text": "38.7 DLLs per process is low and pushes toward ransomware.", "extractor": "given"}, {"feature": "handles.nkey", "direction": "toward_ransomware", "level": "typical", "value_stated": 733.0, "text": "733 registry-key handles is typical but pushes toward ransomware.", "extractor": "given"}, {"feature": "psxview.not_in_deskthrd_false_avg", "direction": "toward_benign", "level": "typical", "value_stated": 0.1463, "text": "0.1463 benign deskthrd cross-view mismatches per process is typical and benign-leaning.", "extractor": "given"}, {"feature": "svcscan.nactive", "direction": "toward_ransomware", "level": "low", "value_stated": 120.0, "text": "120 active services is low and pushes toward ransomware.", "extractor": "given"}, {"feature": "handles.nthread", "direction": "toward_benign", "level": "low", "value_stated": 697.0, "text": "697 thread handles is low but benign-leaning here.", "extractor": "given"}, {"fe)


**Verifier**

- stated verdict/prob: benign / 0.326
- FS=1.000 pass=True precision=1.00 coverage=1.00
- claims:
  - `dlllist.avg_dlls_per_proc` -> supported (failed: none)
  - `handles.nkey` -> supported (failed: none)
  - `psxview.not_in_deskthrd_false_avg` -> supported (failed: none)
  - `svcscan.nactive` -> supported (failed: none)
  - `handles.nthread` -> supported (failed: none)
  - `ldrmodules.not_in_init_avg` -> supported (failed: none)

## test_033454 (arm S2, stratum tp)

**Evidence table**

```
1. pslist.avg_threads = 9.3256 | very_low | toward_ransomware | 3.5pct
2. ldrmodules.not_in_load_avg = 0.027578 | low | toward_ransomware | 7.4pct
3. handles.nsection = 179 | typical | toward_ransomware | 33.0pct
4. handles.nmutant = 264 | typical | toward_benign | 35.0pct
5. handles.nfile = 661 | typical | toward_ransomware | 34.0pct
6. ldrmodules.not_in_mem_avg = 0.027578 | low | toward_ransomware | 7.3pct
```

**Rationale** ({"verdict": "ransomware", "probability_stated": 0.636, "confidence_statement": "The evidence is mixed but the balance favours ransomware.", "claims": [{"feature": "pslist.avg_threads", "direction": "toward_ransomware", "level": "very_low", "value_stated": 9.33, "text": "9.33 threads per process is very low and supports the verdict.", "extractor": "given"}, {"feature": "ldrmodules.not_in_load_avg", "direction": "toward_ransomware", "level": "low", "value_stated": 0.02758, "text": "0.02758 modules per process absent from the load list is low.", "extractor": "given"}, {"feature": "handles.nsection", "direction": "toward_ransomware", "level": "typical", "value_stated": 179.0, "text": "179 section handles is typical and leans toward ransomware.", "extractor": "given"}, {"feature": "handles.nmutant", "direction": "toward_benign", "level": "typical", "value_stated": 264.0, "text": "264 mutant handles is typical and pushes toward benign.", "extractor": "given"}, {"feature": "handles.nfile", "direction": "toward_ransomware", "level": "typical", "value_stated": 661.0, "text": "661 file handles is typical and leans toward ransomware.", "extractor": "given"}, {"feature": "ldrmodules.not_in_mem)


**Verifier**

- stated verdict/prob: ransomware / 0.636
- FS=1.000 pass=True precision=1.00 coverage=1.00
- claims:
  - `pslist.avg_threads` -> supported (failed: none)
  - `ldrmodules.not_in_load_avg` -> supported (failed: none)
  - `handles.nsection` -> supported (failed: none)
  - `handles.nmutant` -> supported (failed: none)
  - `handles.nfile` -> supported (failed: none)
  - `ldrmodules.not_in_mem_avg` -> supported (failed: none)

## test_032951 (arm S2, stratum tp)

**Evidence table**

```
1. dlllist.avg_dlls_per_proc = 38.628 | low | toward_ransomware | 22.0pct
2. svcscan.nactive = 123 | typical | toward_ransomware | 75.0pct
3. ldrmodules.not_in_load_avg = 0.02755 | low | toward_ransomware | 7.2pct
4. handles.nsection = 183 | typical | toward_ransomware | 40.0pct
5. handles.nmutant = 269 | typical | toward_benign | 42.0pct
6. ldrmodules.not_in_mem_avg = 0.02755 | low | toward_ransomware | 7.1pct
```

**Rationale** ({"verdict": "ransomware", "probability_stated": 0.715, "confidence_statement": "The evidence is mixed but the balance favours ransomware.", "claims": [{"feature": "dlllist.avg_dlls_per_proc", "direction": "toward_ransomware", "level": "low", "value_stated": 38.6, "text": "38.6 DLLs per process is low and supports the verdict.", "extractor": "given"}, {"feature": "svcscan.nactive", "direction": "toward_ransomware", "level": "typical", "value_stated": 123.0, "text": "123 active services is typical and leans toward ransomware.", "extractor": "given"}, {"feature": "ldrmodules.not_in_load_avg", "direction": "toward_ransomware", "level": "low", "value_stated": 0.02755, "text": "0.02755 modules per process absent from the load list is low.", "extractor": "given"}, {"feature": "handles.nsection", "direction": "toward_ransomware", "level": "typical", "value_stated": 183.0, "text": "183 section handles is typical and leans toward ransomware.", "extractor": "given"}, {"feature": "handles.nmutant", "direction": "toward_benign", "level": "typical", "value_stated": 269.0, "text": "269 mutant handles is typical and pushes toward benign.", "extractor": "given"}, {"feature": "ldrmodules.not_in_mem_)


**Verifier**

- stated verdict/prob: ransomware / 0.715
- FS=1.000 pass=True precision=1.00 coverage=1.00
- claims:
  - `dlllist.avg_dlls_per_proc` -> supported (failed: none)
  - `svcscan.nactive` -> supported (failed: none)
  - `ldrmodules.not_in_load_avg` -> supported (failed: none)
  - `handles.nsection` -> supported (failed: none)
  - `handles.nmutant` -> supported (failed: none)
  - `ldrmodules.not_in_mem_avg` -> supported (failed: none)

## test_038036 (arm S2, stratum fp)

**Evidence table**

```
1. handles.nsection = 188 | typical | toward_ransomware | 44.0pct
2. callbacks.ncallbacks = 56 | very_low | toward_ransomware | 0.1pct
3. handles.nthread = 820 | typical | toward_ransomware | 44.3pct
4. handles.nmutant = 275 | typical | toward_benign | 46.0pct
5. pslist.avg_threads = 9.7174 | low | toward_ransomware | 15.3pct
6. handles.nsemaphore = 656 | typical | toward_ransomware | 45.0pct
```

**Rationale** ({"verdict": "ransomware", "probability_stated": 0.869, "confidence_statement": "The evidence is mixed but the balance favours ransomware.", "claims": [{"feature": "handles.nsection", "direction": "toward_ransomware", "level": "typical", "value_stated": 188.0, "text": "188 section handles is typical and leans toward ransomware.", "extractor": "given"}, {"feature": "callbacks.ncallbacks", "direction": "toward_ransomware", "level": "very_low", "value_stated": 56.0, "text": "56 kernel callbacks is very low and strongly supports the verdict.", "extractor": "given"}, {"feature": "handles.nthread", "direction": "toward_ransomware", "level": "typical", "value_stated": 820.0, "text": "820 thread handles is typical and leans toward ransomware.", "extractor": "given"}, {"feature": "handles.nmutant", "direction": "toward_benign", "level": "typical", "value_stated": 275.0, "text": "275 mutant handles is typical and pushes toward benign.", "extractor": "given"}, {"feature": "pslist.avg_threads", "direction": "toward_ransomware", "level": "low", "value_stated": 9.72, "text": "9.72 threads per process is low and supports the verdict.", "extractor": "given"}, {"feature": "handles.nsemaphore", "dire)


**Verifier**

- stated verdict/prob: ransomware / 0.869
- FS=1.000 pass=True precision=1.00 coverage=1.00
- claims:
  - `handles.nsection` -> supported (failed: none)
  - `callbacks.ncallbacks` -> supported (failed: none)
  - `handles.nthread` -> supported (failed: none)
  - `handles.nmutant` -> supported (failed: none)
  - `pslist.avg_threads` -> supported (failed: none)
  - `handles.nsemaphore` -> supported (failed: none)

## test_032517 (arm S2, stratum tp)

**Evidence table**

```
1. dlllist.avg_dlls_per_proc = 39.561 | typical | toward_ransomware | 44.1pct
2. ldrmodules.not_in_load_avg = 0.027594 | low | toward_ransomware | 7.5pct
3. handles.nthread = 859 | typical | toward_ransomware | 50.5pct
4. ldrmodules.not_in_mem_avg = 0.027594 | low | toward_ransomware | 7.4pct
5. handles.nevent = 3115 | typical | toward_ransomware | 48.3pct
6. pslist.avg_handlers = 215.88 | typical | toward_ransomware | 43.4pct
```

**Rationale** ({"verdict": "ransomware", "probability_stated": 0.917, "confidence_statement": null, "claims": [{"feature": "dlllist.avg_dlls_per_proc", "direction": "toward_ransomware", "level": "typical", "value_stated": 39.6, "text": "39.6 DLLs per process is typical but leans toward ransomware.", "extractor": "given"}, {"feature": "ldrmodules.not_in_load_avg", "direction": "toward_ransomware", "level": "low", "value_stated": 0.02759, "text": "0.02759 modules per process absent from the load list is low and supports the verdict.", "extractor": "given"}, {"feature": "handles.nthread", "direction": "toward_ransomware", "level": "typical", "value_stated": 859.0, "text": "859 thread handles is typical and leans toward ransomware.", "extractor": "given"}, {"feature": "ldrmodules.not_in_mem_avg", "direction": "toward_ransomware", "level": "low", "value_stated": 0.02759, "text": "0.02759 loader-listed modules missing from memory per process is low.", "extractor": "given"}, {"feature": "handles.nevent", "direction": "toward_ransomware", "level": "typical", "value_stated": 3115, "text": "3,115 event handles is typical and leans toward ransomware.", "extractor": "given"}, {"feature": "pslist.avg_handlers)


**Verifier**

- stated verdict/prob: ransomware / 0.917
- FS=1.000 pass=True precision=1.00 coverage=1.00
- claims:
  - `dlllist.avg_dlls_per_proc` -> supported (failed: none)
  - `ldrmodules.not_in_load_avg` -> supported (failed: none)
  - `handles.nthread` -> supported (failed: none)
  - `ldrmodules.not_in_mem_avg` -> supported (failed: none)
  - `handles.nevent` -> supported (failed: none)
  - `pslist.avg_handlers` -> supported (failed: none)
