# NLP Coursework — PCL Detection (SemEval 2022 Task 4)

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data
Download the data from the following sources and place in `data/`:

1. Clone https://github.com/CRLala/NLPLabs-2024
   - Dont_Patronize_Me_Trainingset/dontpatronizeme_pcl.tsv
   - Dont_Patronize_Me_Trainingset/dontpatronizeme_categories.tsv

2. Clone https://github.com/Perez-AlmendrosC/dontpatronizeme
   - semeval-2022/practice splits/train_semeval_parids-labels.csv
   - semeval-2022/practice splits/dev_semeval_parids-labels.csv
   - semeval-2022/TEST/task4_test.tsv

## Structure
- `eda/` — exploratory data analysis notebook
- `BestModel/` — final model training code and saved model
- `predictions/` — dev.txt and test.txt for submission
- `dont_patronize_me.py` — data loading helper
- `evaluation.py` — official evaluation script