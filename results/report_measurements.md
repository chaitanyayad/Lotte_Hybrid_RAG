# LoTTE RAG Comprehensive Experimental Report Measurements

## 1. System Engineering & Resource Footprint
- **Vocabulary Size (stemmed tokens)**: `149,046`
- **Stop-words List Size**: `198`
- **Stemmer**: `PorterStemmer`
- **Total Startup Latency**: `11.24 seconds`
- **RAM Allocated at Startup**: `3536.7 MB` (Total Process RSS: `3727.5 MB`)
- **FAISS Index Vector Size**: `393.89 MB` (`268,893` vectors $\times 384$ dim, float32)

## 2. Latency Benchmarks (417 Queries)

| System | Mean (ms) | Median (ms) | P95 (ms) |
| :--- | :---: | :---: | :---: |
| **TF-IDF** | 45.85 | 40.57 | 120.59 |
| **BM25** | 45.0 | 47.69 | 94.66 |
| **Dense (MiniLM)** | 24.44 | 22.09 | 31.17 |
| **Hybrid RRF** | 0.17 | 0.16 | 0.21 |
| **End-to-End API RAG (top-5 + synthesis)** | 73.67 | 50.07 | 146.8 |

## 3. Pairwise Win / Loss / Tie (nDCG@10)

| Comparison | Wins | Losses | Ties |
| :--- | :---: | :---: | :---: |
| **Dense vs BM25** | 235 | 76 | 106 |
| **Hybrid vs Dense** | 126 | 155 | 136 |
| **Hybrid vs BM25** | 239 | 44 | 134 |

## 4. Statistical Significance Tests (nDCG@10)

| Comparison | Mean Diff | Paired t-test ($t, p$) | Wilcoxon ($W, p$) | Bootstrap 95% CI |
| :--- | :---: | :---: | :---: | :---: |
| **Dense vs BM25** | +0.1525 | $t=10.5558, p=3.08e-23$ | $W=9048.5, p=9.38e-22$ | [0.1246, 0.1813] |
| **Hybrid vs Dense** | -0.0302 | $t=-2.8685, p=4.33e-03$ | $W=16452.5, p=1.38e-02$ | [-0.0513, -0.0099] |

## 5. BM25 Hyperparameter Grid Sweep

| $k_1$ | $b$ | P@10 | Recall@100 | MAP | nDCG@10 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.9 | 0.5 | 0.1048 | 0.6312 | 0.2724 | 0.3292 |
| 0.9 | 0.75 | 0.1084 | 0.6350 | 0.2703 | 0.3313 |
| 0.9 | 0.9 | 0.1067 | 0.6315 | 0.2563 | 0.3170 |
| 1.2 | 0.5 | 0.1065 | 0.6307 | 0.2717 | 0.3309 |
| 1.2 | 0.75 | 0.1077 | 0.6399 | 0.2672 | 0.3282 |
| 1.2 | 0.9 | 0.1062 | 0.6303 | 0.2527 | 0.3143 |
| 1.5 | 0.5 | 0.1053 | 0.6324 | 0.2694 | 0.3275 |
| 1.5 | 0.75 | 0.1082 | 0.6381 | 0.2668 | 0.3286 |
| 1.5 | 0.9 | 0.1046 | 0.6320 | 0.2512 | 0.3103 |
| 2.0 | 0.5 | 0.1036 | 0.6279 | 0.2684 | 0.3246 |
| 2.0 | 0.75 | 0.1055 | 0.6339 | 0.2635 | 0.3221 |
| 2.0 | 0.9 | 0.1034 | 0.6290 | 0.2458 | 0.3047 |

## 6. RRF and Score Fusion Sweep

| Fusion Strategy | $k$ | Dense Weight ($w$) | P@10 | Recall@100 | MAP | nDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Standard RRF | 10 | - | 0.1516 | 0.7903 | 0.3918 | 0.4635 |
| Standard RRF | 30 | - | 0.1494 | 0.7903 | 0.3864 | 0.4562 |
| Standard RRF | 60 | - | 0.1463 | 0.7903 | 0.3830 | 0.4508 |
| Standard RRF | 100 | - | 0.1451 | 0.7903 | 0.3814 | 0.4483 |
| Weighted RRF (k=60) | 60 | 0.5 | 0.1463 | 0.7903 | 0.3830 | 0.4508 |
| Weighted RRF (k=60) | 60 | 0.7 | 0.1561 | 0.8017 | 0.3944 | 0.4681 |
| Weighted RRF (k=60) | 60 | 0.9 | 0.1600 | 0.7980 | 0.4056 | 0.4830 |
| Linear Score Fusion | - | 0.5 | 0.1511 | 0.7880 | 0.3958 | 0.4665 |
| Linear Score Fusion | - | 0.7 | 0.1590 | 0.8004 | 0.4188 | 0.4924 |
| Linear Score Fusion | - | 0.9 | 0.1595 | 0.8063 | 0.4110 | 0.4863 |

## 7. Recall@k Multi-Cutoff Evaluation

| System | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF** | 0.0140 | 0.0711 | 0.1016 | 0.1619 | 0.2817 | 0.3821 |
| **BM25** | 0.1359 | 0.3102 | 0.3973 | 0.4616 | 0.5715 | 0.6381 |
| **Dense (MiniLM)** | 0.2131 | 0.4426 | 0.5482 | 0.6248 | 0.7191 | 0.7980 |
| **Hybrid RRF** | 0.1986 | 0.4172 | 0.5148 | 0.6088 | 0.7183 | 0.7903 |

## 8. Performance by Query Length (BM25 vs Dense)

| Query Length Bucket | System | P@10 | Recall@100 | MAP | nDCG@10 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Short (<= 6 words)** ($N=210$) | BM25 | 0.1119 | 0.6544 | 0.2842 | 0.3405 |
| | Dense | 0.1619 | 0.8111 | 0.4200 | 0.4919 |
| | Hybrid RRF | 0.1495 | 0.7976 | 0.3923 | 0.4570 |
| **Medium (7-10 words)** ($N=181$) | BM25 | 0.1017 | 0.6373 | 0.2509 | 0.3139 |
| | Dense | 0.1530 | 0.7867 | 0.4055 | 0.4793 |
| | Hybrid RRF | 0.1436 | 0.8001 | 0.3784 | 0.4476 |
| **Long (>= 11 words)** ($N=26$) | BM25 | 0.1231 | 0.5128 | 0.2370 | 0.3336 |
| | Dense | 0.1385 | 0.7715 | 0.3148 | 0.4051 |
| | Hybrid RRF | 0.1385 | 0.6625 | 0.3389 | 0.4230 |

## 9. Hard Queries Analysis

- Total queries scoring 0 across all 4 systems: `57` out of `417` (`13.7%`)

### Example 1: QID 20 - *"does sugar make lemonade flat?"*
- **Gold Target**: `PID 16555: As children, we used to put a strip of clean folded tissue/paper towel in the glass. You might have to repeat a few times....`
- **BM25 Top Match**: PID 56508 — *"I dont think this would be essentially different from a backyard lemonade stand. Make sure its something she does for fun, and not something she feels forced to continue with. Thats the biggest downsi..."*
- **Dense Top Match**: PID 10199 — *"You can make a little heap with the sugar, and then with a sheet of paper you try to push the sheet under the heap to get the sugar. Its not a perfect solution, there is probably sugar remaining but n..."*
- **Hybrid Top Match**: PID 10199 — *"You can make a little heap with the sugar, and then with a sheet of paper you try to push the sheet under the heap to get the sugar. Its not a perfect solution, there is probably sugar remaining but n..."*

### Example 2: QID 21 - *"how to open a garage door without making noise?"*
- **Gold Target**: `PID 14617: Use a lipsalve aka chapstick on the hinges, push it into the gaps between the two hinge leaves with your fingers. It also works on swinging shower screens. Oil runs out, lipsalve just stays put....`
- **BM25 Top Match**: PID 146856 — *"I think it all depends on yourself. Just remember that if someone is intruding in your garage already they can pretty much break that door without to much disturbance form outside. If your concern is ..."*
- **Dense Top Match**: PID 160203 — *"Any door should open easily by hand. Its a common misconception that garage doors are heavy and need motor drives to lift them. Nope. You dont have to lift the door at all. They have a spring mechanis..."*
- **Hybrid Top Match**: PID 146857 — *"It mostly depends on the type of garage door (composite slider, thin aluminium, or solid wood flip up etc... ) but generally the automated motors connected to your door are extremely tough at preventi..."*

### Example 3: QID 79 - *"are all synths the same?"*
- **Gold Target**: `PID 90950: Most analog VCOs generate almost perfect mathematical waveforms (almost because of minor instabilities/noise but its usually below -60db). But you dont sample VCO, there are many elements in the signal path.. Like high pass filters used to kill DC, u...`
- **BM25 Top Match**: PID 115394 — *"What I hear is not a 2 over 3, but a 4 over 3 pattern going on. I transcribed what I hear and explain it progressively so it is clear. I used chrome song maker to make the graphics. You can go there a..."*
- **Dense Top Match**: PID 120549 — *"It may be true that every synth is just some oscillators, filters and envelopes, but like @Tetsujuin said in the comments, thats the same thing with cars, they each have 2 or 3 pedals. So what is diff..."*
- **Hybrid Top Match**: PID 120549 — *"It may be true that every synth is just some oscillators, filters and envelopes, but like @Tetsujuin said in the comments, thats the same thing with cars, they each have 2 or 3 pedals. So what is diff..."*

## 10. Cross-Encoder Reranker Benchmark (Top-50 Hybrid Candidate Reranking)

- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Inference Hardware**: `cuda` (91.98s)
| Metric | Cross-Encoder Reranked | Dense (MiniLM) | Hybrid RRF |
| :--- | :---: | :---: | :---: |
| **P@10** | **0.1552** | 0.1566 | 0.1463 |
| **Recall@100** | **0.7903** | 0.7980 | 0.7903 |
| **MAP** | **0.3997** | 0.4072 | 0.3830 |
| **nDCG@10** | **0.4788** | 0.4810 | 0.4508 |

## 11. RAG Evaluation over 30 Random Queries

- **Queries Evaluated**: `30`
- **Citation Validity Rate (Gold Present)**: `100.0%` (30/30)
- **Abstention Rate (Gold Present)**: `0.0%` (0/30)
- **Abstention Rate ('not found' when Gold Missing)**: `0.0%` (0/30)

