# Language Identification in Code-Mixed Text using Native Sample Mixing

## Overview

This repository implements a language identification (LID) system for code-mixed text using transformer-based models with native sample mixing strategies. The approach combines code-mixed data with native language samples (English/Hindi) to improve model performance on the challenging task of identifying languages at token level in mixed-language text.

Key Features:

- Supports multiple transformer architectures (XLMR, mBERT, IndicBERT, MuRIL)
- Implements 12 different training configurations for sample mixing strategies
- Early stopping with configurable patience (3-5 epochs)
- Cross-dataset evaluation on Lince and ComiLingua datasets
- Comprehensive metrics reporting (F1 score, precision, recall)


## Directory Structure

```
LID_Native_Sampling/
├── main.py              # Lince dataset handler (configs 0-5)
├── main2.py             # Extended Lince handler (configs 6-11)
├── main3.py             # ComiLingua dataset handler (configs 0-5)
├── main4.py             # Extended ComiLingua handler (configs 6-11)
├── english_dataset_cleaned.csv    # Native English samples
├── hindi_dataset_cleaned.csv      # Native Hindi samples
├── lince_train_hi_eng.csv         # Lince training data
├── lince_test_hi_eng.csv          # Lince evaluation data
├── LID_train_ComiLingua.csv
├── LID_test_ComiLingua.csv      # ComiLingua training
└── requirements.txt        # ComiLingua evaluation
```


## Installation

1. Clone repository:
```bash
git clone https://github.com/Arpan2307/LID_Native_Sampling.git
cd LID_Native_Sampling
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```


## Dataset Preparation

Datasets should be CSV files with:

- `Tags` column containing token-label pairs in JSON format:

```json
[{"key":"token1","value":"lang1"}, {"key":"token2","value":"lang2"}]
```

- Native language datasets must follow same format with pure language samples


## Configurations

The system supports 12 experimental configurations:


| Config | Code-Mixed | English | Hindi | Total Samples |
| :-- | :-- | :-- | :-- | :-- |
| 0 | 4k | - | - | 4k |
| 1 | 4k | 4k | - | 8k |
| 2 | 4k | - | 4k | 8k |
| 3 | 4k | 4k | 4k | 12k |
| 4 | 4k | 2k | 2k | 8k |
| 5 | 2k | 1k | 1k | 4k |
| 6 | 2k | 2k | - | 4k |
| 7 | 2k | - | 2k | 4k |
| 8 | - | 2k | - | 2k |
| 9 | - | - | 2k | 2k |
| 10 | - | 4k | 4k | 8k |
| 11 | - | 2k | 2k | 4k |


## Usage

### Training on Lince Dataset

```bash
# Example: XLMR model with Config 3 (CM+EN+HI)
python main.py --model XLMR \
  --train lince_train_hi_eng.csv \
  --test lince_test_hi_eng.csv \
  --output XLMR_config3.txt \
  --train_config 3 \
  --sample_size_en 4000 \
  --sample_size_hi 4000 \
  --train_sample_size 12000 \
  --patience 3
```


### Evaluating on ComiLingua

```bash
python main3.py --model MuRIL \
  --train LID_train_ComiLingua.csv \
  --test LID_test_ComiLingua.csv \
  --output MuRIL_comilingua.txt \
  --train_config 2 \
  --sample_size_hi 2000
```


### Key Arguments

| Parameter | Description |
| :-- | :-- |
| `--model` | Model architecture (XLMR/mBERT/etc) |
| `--train_config` | Mixing configuration (0-11) |
| `--sample_size_en` | Number of English samples to include |
| `--sample_size_hi` | Number of Hindi samples to include |
| `--train_sample_size` | Total training samples after mixing |
| `--patience` | Early stopping patience (epochs) |

## Results

Example output format showing comprehensive metrics:

```
Test Loss: 0.1428
Test Report:
              precision    recall  f1-score   support

          en       0.91      0.89      0.90      4231
          hi       0.87      0.85      0.86      3872
          un       0.79      0.82      0.80      2011

    accuracy                           0.87     10114
   macro avg       0.86      0.85      0.85     10114
weighted avg       0.87      0.87      0.87     10114
```


