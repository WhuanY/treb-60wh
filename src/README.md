# Linear Probing Experiment 
This repo contains the source code for carrying out the linear probing experiment for using `bert-base-uncased` on `IMDB` dataset.

##  Code Structure
```
HW06-BERT/
├─ model/
│  ├─ bert-base-uncased/ # 
├─ report.md
├─ README.md
├─ data/
│  ├─ IMDB Dataset.csv
├─ src/
│  ├─ data.py
│  ├─ main.py
│  ├─ model.py
│  ├─ utils.py

```

## 🏃Quickstart
To quickly run to get experiment table result, simply:
```
python src/main.py
```

## 🚧Experiment Result Table

| Exp ID | Representation Layer   | Training Strategy    | Data Size | Accuracy (%) | F1-score | Train Time | # Params Updated |
|--------|------------------------|----------------------|-----------|--------------|----------|------------|------------------|
| 1      | Last layer CLS         | Full fine-tuning     | 40k       | xx.xx        | xx.xx    | xx mins    | xx M             |
| 2      | Last layer CLS         | Linear probing       | 40k       | xx.xx        | xx.xx    | xx mins    | xx M             |
| 3      | 8th layer CLS          | Linear probing       | 40k       | xx.xx        | xx.xx    | xx mins    | xx M             |
| 4      | 8th + 12th avg pooling | Linear probing       | 40k       | xx.xx        | xx.xx    | xx mins    | xx M             |
| 5      | Last layer CLS         | Unfreeze last 2      | 40k       | xx.xx        | xx.xx    | xx mins    | xx M             |
| 6      | Last layer CLS         | Unfreeze last 4      | 40k       | xx.xx        | xx.xx    | xx mins    | xx M             |
| 7      | Last layer CLS         | Full fine-tuning     | 5k        | xx.xx        | xx.xx    | xx mins    | xx M             |
| 8      | Last layer CLS         | Linear probing       | 5k        | xx.xx        | xx.xx    | xx mins    | xx M             |
| 9      | 8th + 12th avg pooling | Linear probing       | 5k        | xx.xx        | xx.xx    | xx mins    | xx M             |
| 10     | Last layer CLS         | Full fine-tuning     | 40k       | xx.xx        | xx.xx    | xx mins    | xx M             |

