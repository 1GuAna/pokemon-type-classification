# ---------------------------------------------------------------------------
# Random forest tuning for the Pokemon primary-type classification problem.
#
# The grid is 8 x 8 x 8 = 512 configurations across 5 folds, which is too slow
# to run at knit time (roughly 15 minutes on a laptop). It is run once here and
# the result is cached for analysis.Rmd to read back in.
#
# Run from the project root:  Rscript tune_rf.R
# ---------------------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(janitor)
library(ranger)
tidymodels_prefer()

set.seed(123)   # must match the seed in analysis.Rmd so the split is identical

dir.create("cache", showWarnings = FALSE)

# --- data ------------------------------------------------------------------

pokemon <- read_csv("data/Pokemon.csv") %>%
  clean_names() %>%
  mutate(type_1     = fct_lump_n(factor(type_1), n = 6),
         legendary  = factor(legendary),
         generation = factor(generation))

pokemon_split <- initial_split(pokemon, prop = 0.80, strata = type_1)
pokemon_train <- training(pokemon_split)
pokemon_folds <- vfold_cv(pokemon_train, v = 5, strata = type_1)

# --- recipe ----------------------------------------------------------------

pokemon_recipe <- recipe(type_1 ~ legendary + generation + sp_atk + attack +
                           speed + defense + hp + sp_def,
                         data = pokemon_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())

# --- model and grid --------------------------------------------------------

rf_model <- rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(pokemon_recipe)

rf_grid <- grid_regular(
  mtry(range  = c(1, 8)),      # 8 predictors, so mtry cannot exceed 8
  trees(range = c(200, 800)),
  min_n(range = c(2, 20)),
  levels = 8
)

# --- tune ------------------------------------------------------------------

rf_tune <- tune_grid(
  rf_wf,
  resamples = pokemon_folds,
  grid      = rf_grid,
  metrics   = metric_set(roc_auc)
)

write_rds(rf_tune, "cache/rf_tune.rds")

message("Done. Best configuration:")
print(select_best(rf_tune, metric = "roc_auc"))
