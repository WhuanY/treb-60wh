| Exp ID | Representation Layer   | Training Strategy    | Data Size | Accuracy (%) | F1-score | Train Time | # Params Updated |
|--------|------------------------|----------------------|-----------|--------------|----------|------------|------------------|
| 1      | Last layer CLS         | Full fine-tuning     | 40k       | 94.33        | 94.34    | 42.12 mins    | 109,483,778   |
| 2      | Last layer CLS         | Linear probing       | 40k       | 66.83        | 69.79    | 30.12 mins    | 1538          |
| 3      | 8th layer CLS          | Linear probing       | 40k       | 57.71        | 61.73    | 31.98 mins    | 1538          |
| 4      | 8th + 12th avg pooling | Linear probing       | 40k       | 59.22        | 66.33    | 32.73 mins    | 1538          |
| 5      | Last layer CLS         | Unfreeze last 2      | 40k       | 92.48        | 92.41    | 41.48 mins    | 14,177,282    |
| 6      | Last layer CLS         | Unfreeze last 4      | 40k       | 93.52        | 93.50    | 46.60 mins    | 28,353,026    |
| 7      | Last layer CLS         | Full fine-tuning     | 5k        | 91.76        | 91.89    | 26.08 mins    | 109,483,778   |
| 8      | Last layer CLS         | Linear probing       | 5k        | 57.82        | 66.61    | 24.65 mins    | 1538          |
| 9      | 8th + 12th avg pooling | Linear probing       | 5k        | 54.51        | 0.64     | 18.05 mins    | 1538 M        |
| 10     | Last layer CLS         | Full fine-tuning     | 40k       | xx.xx        | xx.xx    | xx mins    | xx M             |