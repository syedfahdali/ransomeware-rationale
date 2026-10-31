# Laya cross-check — disagreements worth a look

Cases where the independent decision layer disagrees with the deterministic verifier / classifier. Advisory only: nothing here changes FS (see `laya_fs_independence` in outputs/phase_checks.json).

## arm P0

- gate: 0 hard disagreement(s) of 50 samples; agreement rate 1.000
- claim cross-judge: LAR 0.931 (strict 0.903) over 180 claims

- **test_013878** `handles.nmutant`: verifier `contradicted` vs Laya `supported` (confidence 0.187).
- **test_031534** `malfind.commitCharge`: verifier `contradicted` vs Laya `supported` (confidence 0.313).
- **test_032517** `handles.avg_handles_per_proc`: verifier `out_of_topk` vs Laya `supported` (confidence 0.057).
- **test_032644** `ldrmodules.not_in_init_avg`: verifier `supported` vs Laya `not_in_evidence` (confidence 0.034).
- **test_033454** `handles.nfile`: verifier `contradicted` vs Laya `supported` (confidence 0.037).
- **test_034133** `ldrmodules.not_in_load_avg`: verifier `out_of_topk` vs Laya `supported` (confidence 0.038).
- **test_053435** `handles.nthread`: verifier `supported` vs Laya `not_in_evidence` (confidence 0.007).
- **test_053436** `pslist.avg_threads`: verifier `contradicted` vs Laya `supported` (confidence 0.096).
- **test_055867** `handles.nfile`: verifier `supported` vs Laya `not_in_evidence` (confidence 0.006).
- **test_057797** `svcscan.nactive`: verifier `contradicted` vs Laya `supported` (confidence 0.218).
- **test_057797** `handles.nthread`: verifier `supported` vs Laya `not_in_evidence` (confidence 0.005).

## arm P0R

- gate: 0 hard disagreement(s) of 50 samples; agreement rate 1.000
- claim cross-judge: LAR 0.990 (strict 0.985) over 234 claims

- **test_032644** `ldrmodules.not_in_init_avg`: verifier `supported` vs Laya `not_in_evidence` (confidence 0.034).
- **test_034133** `ldrmodules.not_in_load_avg`: verifier `out_of_topk` vs Laya `supported` (confidence 0.038).

## arm P2

- gate: 0 hard disagreement(s) of 50 samples; agreement rate 1.000
- claim cross-judge: LAR 0.953 (strict 0.953) over 300 claims

- **test_013878** `ldrmodules.not_in_load_avg`: verifier `supported` vs Laya `not_in_evidence` (confidence 0.033).
- **test_032287** `handles.nkey`: verifier `supported` vs Laya `not_in_evidence` (confidence 0.022).
- **test_032402** `handles.nkey`: verifier `supported` vs Laya `not_in_evidence` (confidence 0.007).
- **test_032517** `handles.nthread`: verifier `supported` vs Laya `not_in_evidence` (confidence 0.016).
- **test_032950** `svcscan.nactive`: verifier `supported` vs Laya `not_in_evidence` (confidence 0.011).
- **test_034844** `handles.nthread`: verifier `supported` vs Laya `not_in_evidence` (confidence 0.020).
- **test_038036** `handles.nsemaphore`: verifier `supported` vs Laya `not_in_evidence` (confidence 0.010).
- **test_039259** `handles.nfile`: verifier `supported` vs Laya `not_in_evidence` (confidence 0.019).
- **test_046734** `handles.nthread`: verifier `supported` vs Laya `not_in_evidence` (confidence 0.021).
- **test_051466** `handles.nsemaphore`: verifier `supported` vs Laya `not_in_evidence` (confidence 0.009).

## arm S2

- gate: 0 hard disagreement(s) of 50 samples; agreement rate 1.000
- claim cross-judge: LAR 1.000 (strict 1.000) over 300 claims


## arm S2R

- gate: 0 hard disagreement(s) of 50 samples; agreement rate 1.000
- claim cross-judge: LAR 1.000 (strict 1.000) over 300 claims


## arm T

- gate: 0 hard disagreement(s) of 50 samples; agreement rate 1.000
- claim cross-judge: LAR 0.997 (strict 0.997) over 300 claims

- **test_032644** `ldrmodules.not_in_init_avg`: verifier `supported` vs Laya `not_in_evidence` (confidence 0.017).

## arm X

- gate: 0 hard disagreement(s) of 50 samples; agreement rate 1.000
- claim cross-judge: LAR 0.387 (strict 0.133) over 300 claims

- **test_003028** `svcscan.nservices`: verifier `contradicted` vs Laya `supported` (confidence 0.209).
- **test_003028** `handles.nsection`: verifier `contradicted` vs Laya `not_in_evidence` (confidence 0.001).
- **test_003028** `dlllist.avg_dlls_per_proc`: verifier `contradicted` vs Laya `supported` (confidence 0.017).
- **test_006079** `svcscan.nservices`: verifier `contradicted` vs Laya `supported` (confidence 0.064).
- **test_006079** `handles.nsection`: verifier `contradicted` vs Laya `supported` (confidence 0.073).
- **test_006079** `handles.nmutant`: verifier `contradicted` vs Laya `supported` (confidence 0.013).
- **test_012875** `svcscan.nservices`: verifier `contradicted` vs Laya `supported` (confidence 0.176).
- **test_012875** `handles.nsection`: verifier `contradicted` vs Laya `not_in_evidence` (confidence 0.002).
- **test_012875** `dlllist.avg_dlls_per_proc`: verifier `contradicted` vs Laya `supported` (confidence 0.002).
- **test_013221** `svcscan.nservices`: verifier `contradicted` vs Laya `supported` (confidence 0.006).
- **test_013221** `handles.nsection`: verifier `contradicted` vs Laya `not_in_evidence` (confidence 0.013).
- **test_013221** `dlllist.avg_dlls_per_proc`: verifier `contradicted` vs Laya `supported` (confidence 0.037).
- **test_013878** `handles.nsection`: verifier `supported` vs Laya `not_in_evidence` (confidence 0.017).
- **test_016376** `dlllist.avg_dlls_per_proc`: verifier `contradicted` vs Laya `supported` (confidence 0.417).
- **test_016376** `handles.nsection`: verifier `contradicted` vs Laya `not_in_evidence` (confidence 0.017).
- **test_016376** `handles.nmutant`: verifier `contradicted` vs Laya `supported` (confidence 0.017).
- **test_020120** `pslist.avg_threads`: verifier `contradicted` vs Laya `supported` (confidence 0.010).
- **test_020120** `ldrmodules.not_in_load_avg`: verifier `out_of_topk` vs Laya `supported` (confidence 0.091).
- **test_025235** `dlllist.avg_dlls_per_proc`: verifier `contradicted` vs Laya `supported` (confidence 0.006).
- **test_025235** `handles.nsection`: verifier `contradicted` vs Laya `not_in_evidence` (confidence 0.002).
- **test_025235** `handles.nmutant`: verifier `contradicted` vs Laya `supported` (confidence 0.024).
- **test_028179** `malfind.ninjections`: verifier `out_of_topk` vs Laya `supported` (confidence 0.064).
- **test_028179** `handles.ntimer`: verifier `out_of_topk` vs Laya `supported` (confidence 0.101).
- **test_028179** `pslist.avg_threads`: verifier `contradicted` vs Laya `supported` (confidence 0.075).
- **test_030109** `handles.nthread`: verifier `out_of_topk` vs Laya `supported` (confidence 0.108).
- **test_030109** `handles.nsection`: verifier `out_of_topk` vs Laya `supported` (confidence 0.045).
- **test_030109** `handles.nhandles`: verifier `out_of_topk` vs Laya `supported` (confidence 0.062).
