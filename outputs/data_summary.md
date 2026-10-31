# Data summary — CIC-MalMem-2022

- Raw file: `/home/faddy/Desktop/githubrepos/ransomware-rationale/data/raw/CIC-MalMem-2022.csv`
- Raw rows: 58596
- Rows dropped for NaN features: 0
- Exact duplicate rows dropped: 573
- Final rows: 58023
- Features: 55
- Generated: 2026-10-10T16:52:51+00:00

## Leakage structure (important)

- The Category column encodes the source memory capture of every malware row (e.g. `Ransomware-Shade-<hash>-7.raw`).
- Distinct malware captures (hashes): 2844; rows per capture: mean 10.0, median 10, max 20.
- Malware rows without a capture hash: 429.
- Malware rows from the same capture are kept in ONE split via group-aware splitting (`group_id`).
- Benign rows carry NO capture ids in this mirror; they are split row-level. Near-duplicate benign dumps may therefore span splits — documented limitation.

## `Class` column

```
Class
Benign     29227
Malware    28796
```

## Binary task (`y_binary`: ransomware vs rest)

```
y_binary
not_ransomware    48500
ransomware         9523
```

## 3-class task

```
y_3class
benign           29227
other_malware    19273
ransomware        9523
```

## Malware classes (non-benign rows)

```
malware_class
Spyware       9797
Ransomware    9523
Trojan        9476
```

## Ransomware families

```
family
Shade    2125
Ako      1999
Conti    1987
Maze     1754
Pysa     1658
```

## Other-malware families

```
family
Transponder     2403
180solutions    1999
CWS             1999
Scar            1999
Refroso         1996
Gator           1990
Emotet          1967
Zeus            1946
Reconyc         1568
TIBS            1406
```
