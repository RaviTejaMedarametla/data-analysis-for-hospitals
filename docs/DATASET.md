# Dataset Notes

The repository uses small tabular CSV snapshots from three synthetic hospital contexts (`general`, `prenatal`, `sports`).

## Schema
- Numeric: `age`, `height`, `weight`, `bmi`, `children`, `months`
- Categorical: `gender`, `blood_test`, `ecg`, `ultrasound`, `mri`, `xray`, `diagnosis`, `hospital`

## Preprocessing assumptions
- Missing numerics are filled with `0`
- Missing tests and diagnosis are filled with `unknown`
- Gender labels are normalized to `m` / `f`
