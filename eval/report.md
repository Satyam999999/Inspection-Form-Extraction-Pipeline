# Inspection-Form Extraction — Evaluation Report

**Forms evaluated**: 15

## Per-Form Results

| form_id     | quality        |   scalar_f1 |   checklist_f1 |   nc_f1 |   overall_f1 |
|:------------|:---------------|------------:|---------------:|--------:|-------------:|
| SW-2024-001 | clean          |       1.000 |          1.000 |   1.000 |        1.000 |
| EQ-2024-002 | clean          |       1.000 |          1.000 |   1.000 |        1.000 |
| FS-2024-003 | clean          |       1.000 |          1.000 |   1.000 |        1.000 |
| SC-2024-004 | clean          |       1.000 |          1.000 |   1.000 |        1.000 |
| EL-2024-005 | clean          |       1.000 |          1.000 |   1.000 |        1.000 |
| HA-2024-006 | clean          |       1.000 |          1.000 |   1.000 |        1.000 |
| SW-2024-007 | blurry         |       0.875 |          0.800 |   1.000 |        0.892 |
| EQ-2024-008 | lowres         |       1.000 |          0.750 |   1.000 |        0.917 |
| FS-2024-009 | rotated        |       1.000 |          1.000 |   1.000 |        1.000 |
| EL-2024-010 | handwritten    |       0.875 |          0.800 |   1.000 |        0.892 |
| SC-2024-011 | noisy          |       1.000 |          0.889 |   0.000 |        0.630 |
| HA-2024-012 | dark           |       1.000 |          0.750 |   1.000 |        0.917 |
| SW-2024-013 | clean          |       1.000 |          1.000 |   1.000 |        1.000 |
| FS-2024-014 | blurry_rotated |       1.000 |          0.750 |   0.667 |        0.806 |
| EQ-2024-015 | overexposed    |       1.000 |          0.889 |   1.000 |        0.963 |


## Aggregate Metrics

| Metric | Mean F1 |
|--------|---------|
| scalar_f1 | 0.983 |
| checklist_f1 | 0.908 |
| nc_f1 | 0.911 |
| overall_f1 | 0.934 |


## Performance by Scan Quality

| quality        |   mean_f1 |
|:---------------|----------:|
| blurry         |     0.892 |
| blurry_rotated |     0.806 |
| clean          |     1.000 |
| dark           |     0.917 |
| handwritten    |     0.892 |
| lowres         |     0.917 |
| noisy          |     0.630 |
| overexposed    |     0.963 |
| rotated        |     1.000 |


## Failure Analysis


### Blurry

- [inspector.id_number] pred='TB-11Z' ≠ gt='TB-112'
- Checklist item not found: 'VENTILATIONDUCTSINPLACE'
- Spurious checklist item: 'VENTILATIONDUCTINPLACE'

### Blurry Rotated

- Checklist item not found: 'ELECTRICALPANELLOCKED'
- Checklist item not found: 'EMERGENCYLIGHTINGTESTED'
- Missing NC: 'EMERGENCYLIGHTINGFAILURE'

### Dark

- Result mismatch for 'GROUNDINGCABLESONTANKERS': pred=N/A gt=FAIL
- Result mismatch for 'NOSMOKINGSIGNSPOSTED': pred=N/A gt=PASS

### Handwritten

- [inspector.name] pred='Fatima Al-Rashid [illegible]' ≠ gt='Fatima Al-Rashid'
- Checklist item not found: 'EARTHINGPITSRESISTANCE<10Ω'
- Spurious checklist item: 'EARTHINGPITSRESISTANCE<100'

### Lowres

- Result mismatch for 'CONVEYORBELTALIGNMENT': pred=PASS gt=FAIL
- Checklist item not found: 'OPERATORHEARINGPROTECTION'

### Noisy

- Result mismatch for 'TRAPDOORHATCHSECURED': pred=PASS gt=FAIL
- Missing NC: 'MISSINGHATCHPIN'

### Overexposed

- Result mismatch for 'TOWERBOLTTORQUEVERIFIED': pred=N/A gt=PASS


## Worst-Performing Forms

| form_id     | quality        |   overall_f1 |
|:------------|:---------------|-------------:|
| SC-2024-011 | noisy          |        0.630 |
| FS-2024-014 | blurry_rotated |        0.806 |
| SW-2024-007 | blurry         |        0.892 |
| EL-2024-010 | handwritten    |        0.892 |
| EQ-2024-008 | lowres         |        0.917 |

