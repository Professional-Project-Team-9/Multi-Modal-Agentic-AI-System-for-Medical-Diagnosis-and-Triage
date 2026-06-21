# Code Review - Team 9, Week 5

Team reviewed: Team 9
Repo: https://github.com/Professional-Project-Team-9/Week-5-Team-9

Files looked at:
- chest_xray_preprocessing.ipynb
- efficientnet_fine_tuning.ipynb
- CNN_model_training.ipynb
- hybrid_ehr_model.ipynb
- resnet_and_vit_fine_tuning.ipynb

I have gone through all of the files above. The small individual errors are commented inline, directly above the lines they relate to in each file (look for the # Review: comments). This document is a summary of the main points across the whole codebase.


## Overall

Code is in good shape. The pipeline is easy to follow, the naming is sensible, and there are plenty of inline comments explaining the choices (augmentation scale, imbalance handling, patient leakage). The harder data problems have clearly been thought about. The points below are mostly small and none of them are blockers.

## Functionality

The train/val/test split is done patient-wise, so there's no patient overlap between sets, which matches the test plan. One thing to check: the test plan also asks for stratified splits within 2% on class frequency, but the split is currently random with no stratification. On an imbalanced dataset, some of the rarer findings could end up unevenly distributed, so it may be worth printing the per-split class frequencies to confirm they're within the target range.

## Logic and readability

The main thing here is duplication. ChestXrayDataset and the transform block are copied almost identically between efficientnet_fine_tuning and CNN_model_training. If one copy is changed later the others will drift. Pulling the shared parts into a small .py module that both notebooks import would make this easier to maintain.

Also a couple of small cleanup items: filter_chestxray appears to keep every row (the condition always evaluates to true), and in the hybrid notebook the PATIENT column is renamed to itself. Neither causes problems, but both could be removed.

## Security and performance

ChestXrayDataset caches every image in RAM up front for train, val and test together. On the full chest X-ray set (~112k images) that's a large memory footprint and could run the machine out of memory. A capped/lazy cache, or just reading from disk with a
few workers, would scale more safely.

On the security side, PII columns (SSN, name, address etc.) are dropped before modelling, which is good. The hybrid notebook uses an API key via dotenv - fine as long as the .env file is gitignored and the key is never printed in cell output.

## Coding standards

Measured against the team's own standard in the test plan, a few small gaps: a few public functions are missing docstrings/type hints (e.g. train_one_epoch, evaluate), and a lot of the print statements use emojis where the standard points to plain logging. Quick to tidy up.

## Summary

Functionality is sound and the data handling is solid. Main suggestions are to verify the class distribution across splits, reduce the duplicated Dataset code, and consider a more memory-efficient caching strategy. Adding docstrings to the remaining functions would bring it fully in line with the team's standard. Good work overall.
