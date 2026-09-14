# Predicting Pokémon primary type from battle statistics

A multiclass classification problem: seven outcome classes, eight predictors,
800 observations. Completed for PSTAT 131, Introduction to Machine Learning, at
UC Santa Barbara, summer 2026.

## The problem

Each Pokémon has a primary type. The question is how far that type can be
predicted from six battle statistics plus generation and legendary status. The
eighteen original types are badly unbalanced — the rarest has four members — so
the twelve least frequent are pooled into a single `Other` category, leaving
seven classes.

## Approach

- Stratified 80/20 train/test split, and 5-fold cross-validation stratified on
  the outcome. Stratification matters here: without it, rare classes can be
  absent from a fold entirely and the fold-level estimates become unstable.
- Two models tuned over regular grids and compared on ROC AUC: an elastic net
  multinomial regression (`glmnet`, 100 configurations) and a random forest
  (`ranger`, 512 configurations).

## Result

The random forest wins on cross-validated ROC AUC, 0.720 against 0.687, a margin
of about 2.6 standard errors. On the held-out set it reaches an AUC of 0.682 and
an accuracy of 0.410.

The gap between those two test-set numbers is the interesting part. By accuracy
the model looks close to useless outside the largest class, because the argmax
rule sends nearly everything there. By per-class ROC, which assesses ranking
rather than the decision rule, it separates some classes well and others not at
all — it can rank cases by their probability of belonging to a class it almost
never predicts outright. Which of the two summaries is the honest one depends
entirely on what the model would be used for.

The test AUC also falls about three standard errors below the cross-validated
figure. Two things account for that: the test set holds only 161 observations,
and taking the maximum over 512 tuning configurations is optimistically biased,
so part of the winning fold average was luck.

## Files

| File | |
|---|---|
| `analysis.md` | Full write-up with output and figures — **start here** |
| `analysis.Rmd` | Source |
| `tune_rf.R` | Random forest tuning, run separately (~15 minutes) |
| `data/Pokemon.csv` | Data, from [Kaggle](https://www.kaggle.com/abcsds/pokemon) |

## Reproducing

Requires R with `tidyverse`, `tidymodels`, `janitor`, `corrplot`, `glmnet` and
`ranger`.

```r
Rscript tune_rf.R            # writes cache/rf_tune.rds, about 15 minutes
rmarkdown::render("analysis.Rmd")
```

`set.seed(123)` is set identically in both scripts, so the split and the folds
match between them.
