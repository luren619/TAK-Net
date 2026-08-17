# Split files

Create one directory per fold:

```text
splits/
  fold_1/
    train_files.txt
    val_files.txt
    test_files.txt
  ...
  fold_5/
    train_files.txt
    val_files.txt
    test_files.txt
```

Each text file contains one image filename per line. The same filename must exist
under `data/imgs`, `data/ceus`, and `data/masks`. Keep all frames from the same
case in the same fold to prevent case-level leakage.


