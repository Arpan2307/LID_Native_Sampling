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
├── main.py              # Lince dataset handler (configs 0-3)
├── main2.py             # Extended Lince handler (configs 0-4)
├── main3.py             # ComiLingua dataset handler (configs 0-3)
├── main4.py             # Extended ComiLingua handler (configs 0-4)
├── english_dataset_cleaned.csv    # Native English samples
├── hindi_dataset_cleaned.csv      # Native Hindi samples
├── lince_train_hi_eng.csv         # Lince training data
├── lince_test_hi_eng.csv          # Lince evaluation data
├── LID_train_ComiLingua.csv       # ComiLingua training
└── LID_test_ComiLingua.csv        # ComiLingua evaluation
```


## Installation

1. Clone repository:
```bash
git clone https://github.com/Arpan2307/LID_Native_Sampling.git
cd LID_Native_Sampling
```

2. Install dependencies:
```bash
pip install torch transformers pandas numpy scikit-learn
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
| ... | ... | ... | ... | ... |

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


## Citation

If using this work, please cite:

```latex
@misc{LID_Native_Sampling,
  author = {Arpan},
  title = {Code-Mixed LID with Native Sample Mixing},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Arpan2307/LID_Native_Sampling}}
}
```

<div style="text-align: center">⁂</div>

[^1]: Sequence-Classification-tasks-with-native-sample-mixing.pdf

[^2]: LID_Native_Sampling

[^3]: https://stackoverflow.com/questions/39065921/what-do-raw-githubusercontent-com-urls-represent

[^4]: https://gist.github.com/Tensorfengsheng1926/cd7f4924329bddb27b4f8bda108284c2/revisions

[^5]: https://stackoverflow.com/questions/51524687/dont-know-how-to-generate-sampling-locations

[^6]: https://huggingface.co/datasets/Denm/lch_codebase

[^7]: https://github.com/StarlangSoftware/Sampling-Py

[^8]: https://stackoverflow.com/questions/24721751/github-raw-githubusercontent-returning-invalid-request

[^9]: https://pypi.org/project/samplics/

[^10]: https://huggingface.co/datasets/pe-nlp/ov-kit-files-filtered-dedup-py/viewer/default/train?p=4

[^11]: https://rentry.co/wf7pv

[^12]: https://gist.github.com/junhe/806c57ce629e1d7035a1

[^13]: https://gist.github.com/rjlutz/5fb448c6f6b9d2663a745c833bbf162f

[^14]: https://gist.github.com/lukaspiatkowski/f1e2e076758354641a74428036d58635

[^15]: https://stackoverflow.com/questions/24721575/github-url-to-raw-files

[^16]: https://huggingface.co/spaces/hysts/BLIP2/blob/main/app.py

[^17]: https://github.com/orgs/community/discussions/17023

[^18]: https://cocalc.com/github/automatic1111/stable-diffusion-webui/blob/master/modules/launch_utils.py

[^19]: https://gist.github.com/marcobellaccini/29fcb4e3b34923d3be6ad2599f655181

[^20]: https://stackoverflow.com/questions/59208760/python-tagui-trying-to-connect-to-https-raw-githubusercontent-com

[^21]: https://stackoverflow.com/questions/77054807/how-do-you-use-raw-githubusercontent-com-in-a-github-action

[^22]: https://gist.github.com/fedarko/4fc177cff9084b9e325dcbe954547edc

[^23]: https://github.com/orgs/community/discussions/41519

[^24]: https://gist.github.com/driazati/e009f09ff44c6bc91c4d95a8e17fd6f1

[^25]: https://gist.github.com/albertlai431?direction=asc\&sort=created

[^26]: https://gist.github.com/cristiancristea00/408e9cec37e4a441bdae6009faa32e49

[^27]: https://huggingface.co/datasets/AlignmentLab-AI/validated-python-instruct/viewer

[^28]: https://gist.github.com/snmishra

[^29]: https://gist.github.com/EvaMart/a2f087fe9eaf58de3d85f8e72a21893e

[^30]: https://gist.github.com/nirshlezinger1?direction=asc\&sort=created

[^31]: https://gist.github.com/DSDanielPark/starred?direction=desc\&sort=created

[^32]: https://gist.github.com/l4u/c219ead4c686bd38bc70a8e83978b941

[^33]: https://huggingface.co/datasets/ngocuong/Ghepmat/commit/420a3aa889f04645e9e6a849ac74b3470ef0eedb

[^34]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/b5180223

[^35]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/35443277

[^36]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/8cef523c.sample

[^37]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/f10b35a0.sample

[^38]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/8a3870c4.sample

[^39]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/f6669cb8.sample

[^40]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/49c1c609.sample

[^41]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/34cb30cc.sample

[^42]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/be1df996.sample

[^43]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/1f0f93bc.sample

[^44]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/12370859.sample

[^45]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/f0d0ba0f.sample

[^46]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/684a7dee.sample

[^47]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/56be02b9.sample

[^48]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/eeffd61b.sample

[^49]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/edecc321

[^50]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/1bc04b52

[^51]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/c9046f7a

[^52]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/b79606fb

[^53]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/020b54e7.py

[^54]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/edf1fe3e.py

[^55]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/da4ba56e.py

[^56]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/b10564ab.py

[^57]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/1096fe2a.csv

[^58]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/eb25f847.csv

[^59]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/db05833c.csv

[^60]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/2c5d782e.csv

[^61]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/1d84cf94.csv

[^62]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/7a80d3a1f685890bd31baa4f8af9a7a0/eec1b48a-41fa-4e8e-9085-5b89b38d3f9d/37a2136a.csv
