# MAESTRO — Tier Re-balancing Worksheet (entity counts under Phase 0 contract)
*Generated 2026-06-11. Updated 2026-06-12 after rebalance.*
*entity = inline-drawn node; group = subgraph (pool/lane/boundary/expanded sub-process).*
*Goal: each category needs 5 diagrams in each tier. Bands (fixed): t1 `<10`, t2 `10–25`, t3 `>25` entities.*

## BPMN

| file | assigned tier | entities (new) | groups | entities+groups | tier by entities | in band? | note |
|---|---|---|---|---|---|---|---|
| 01_bpmn_1 | t1 (<10) | **5** | 0 | 5 | t1 | ✅ |  |
| 02_bpmn_1 | t1 (<10) | **8** | 0 | 8 | t1 | ✅ |  |
| 03_bpmn_1 | t1 (<10) | **8** | 0 | 8 | t1 | ✅ |  |
| 04_bpmn_1 | t1 (<10) | **9** | 0 | 9 | t1 | ✅ | rebalanced: dropped task_2, rewired sub_process via boundary events only |
| 05_bpmn_1 | t1 (<10) | **9** | 0 | 9 | t1 | ✅ | rebalanced: dropped employee-not-found branch, notify tasks, update tasks; merged approved ends |
| 11_bpmn_2 | t2 (10–25) | **15** | 6 | 21 | t2 | ✅ |  |
| 12_bpmn_2 | t2 (10–25) | **21** | 5 | 26 | t2 | ✅ |  |
| 13_bpmn_2 | t2 (10–25) | **23** | 3 | 26 | t2 | ✅ | rebalanced: dropped IT/Payroll/Facilities pools + 3 icatch events + parallel split/merge in Responsible Dept |
| 14_bpmn_2 | t2 (10–25) | **24** | 4 | 28 | t2 | ✅ | rebalanced: dropped connected-clients subprocess + call activity, replaced corporate rework with B2B referral, dropped document_risk + degenerate merge gateways |
| 15_bpmn_2 | t2 (10–25) | **23** | 3 | 26 | t2 | ✅ |  |
| 21_bpmn_3 | t3 (>25) | **29** | 5 | 34 | t3 | ✅ |  |
| 22_bpmn_3 | t3 (>25) | **30** | 6 | 36 | t3 | ✅ |  |
| 23_bpmn_3 | t3 (>25) | **38** | 2 | 40 | t3 | ✅ |  |
| 24_bpmn_3 | t3 (>25) | **26** | 1 | 27 | t3 | ✅ |  |
| 25_bpmn_3 | t3 (>25) | **27** | 1 | 28 | t3 | ✅ | rebalanced: added HR-input notify send task + interrupting timer boundary + HR timeout end event |

*Assigned per tier: t1=5 t2=5 t3=5 (target 5/5/5). By new entity count: t1=5 t2=5 t3=5.* ✅

## IT

| file | assigned tier | entities (new) | groups | entities+groups | tier by entities | in band? | note |
|---|---|---|---|---|---|---|---|
| 06_it_1 | t1 (<10) | **4** | 1 | 5 | t1 | ✅ |  |
| 07_it_1 | t1 (<10) | **6** | 1 | 7 | t1 | ✅ |  |
| 08_it_1 | t1 (<10) | **4** | 3 | 7 | t1 | ✅ |  |
| 09_it_1 | t1 (<10) | **6** | 2 | 8 | t1 | ✅ |  |
| 10_it_1 | t1 (<10) | **8** | 1 | 9 | t1 | ✅ |  |
| 16_it_2 | t2 (10–25) | **11** | 1 | 12 | t2 | ✅ |  |
| 17_it_2 | t2 (10–25) | **13** | 1 | 14 | t2 | ✅ |  |
| 18_it_2 | t2 (10–25) | **11** | 2 | 13 | t2 | ✅ |  |
| 19_it_2 | t2 (10–25) | **12** | 6 | 18 | t2 | ✅ |  |
| 20_it_2 | t2 (10–25) | **12** | 2 | 14 | t2 | ✅ |  |
| 26_it_3 | t3 (>25) | **28** | 10 | 38 | t3 | ✅ | rebalanced: added CDN edge, per-DC SIEMs in new monitoring zones, bidirectional SIEM replication |
| 27_it_3 | t3 (>25) | **28** | 3 | 31 | t3 | ✅ |  |
| 28_it_3 | t3 (>25) | **26** | 2 | 28 | t3 | ✅ | rebalanced: added BigQuery, Pub/Sub DLQ, Error Reporting |
| 29_it_3 | t3 (>25) | **27** | 5 | 32 | t3 | ✅ | rebalanced: split PostHog into Analytics + Error Tracking sub-systems, completed staging mirror (redis + monitoring + log server + worker), added production background worker |
| 30_it_3 | t3 (>25) | **26** | 3 | 29 | t3 | ✅ | rebalanced: added OTA service, schema registry, secret manager (Vault), IAM service, relational DB backup |

*Assigned per tier: t1=5 t2=5 t3=5 (target 5/5/5). By new entity count: t1=5 t2=5 t3=5.* ✅

## Out-of-band diagrams to adjust

*All previously out-of-band diagrams have been rebalanced. No action required.*

*Reference — if tier were defined on entities+groups ("structural size") instead of entities-only:*
- BPMN by size: recompute after rebalance
- IT by size: recompute after rebalance