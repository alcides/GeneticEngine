This dataset will be used to test geneticEngine to classify exon groups based on motif ocurrences features.

Two groups (classes) exist: KD and CTRL:
    - KD group refers to a list of exons that were affected by knockdown of a specific RNA binding protein. Can be thought as the positive set.
    - CTRL group refers to a list of exons (that match the KD group on genome architecture features) that were not affected by knockdown of a specific RNA binding protein. Can be thought as the negative set.

This data refers to the specific knockdown of an important splicing regulator, SRSF1.

To generate data in an easy format for geneticEngine, just ran:
python split_train_test.py

Then to test genetic engine, in the examples folder:
python genomics_classification.py
