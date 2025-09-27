library(tidyverse)
library(tidymodels)
library(vroom)
library(DataExplorer)
library(patchwork)
library(glmnet)
library(rpart)
library(ranger)
library(bonsai)
library(lightgbm)
library(dbarts)
library(agua)

# read in data
bike_data <- vroom("/Users/eliseclark/Documents/FALL 2025/STAT 348/BikeShareComp/bike-sharing-demand/train.csv")

# change variables into factors 
bike_data <- bike_data %>%
  mutate(workingday = as.factor(workingday), 
         holiday = as.factor(holiday))

# look at variable types & num of obs
glimpse(bike_data)


# read in test data
testData <- vroom("/Users/eliseclark/Documents/Fall 2025/Stat 348/BikeShareComp/bike-sharing-demand/test.csv")

# set factors in test data 
testData <- testData %>%
  mutate(workingday = as.factor(workingday), 
         holiday = as.factor(holiday))


######## EDA ##########

# bar plot of holiday and working day (good or no?)
plot_bar(bike_data)

# histogram of double variables
plot_histogram(bike_data)

# looking at correlations
plot_correlation(bike_data)

# scatterplots
ggplot(data = bike_data, aes(x = humidity, y = windspeed)) +
  geom_point()

ggplot(data = bike_data, aes(x = casual, y = temp)) +
  geom_point()

ggplot(data = bike_data, aes(x = registered, y = temp)) +
  geom_point()


# ggplots of key features (one must be barplot of weather)
weather_plot <- ggplot(data = bike_data, aes(x = weather)) +  # clear, cloudy + mist, light precip, heavy precip
  geom_bar(color = "black", fill = "lightblue") +
  labs(title = "Weather and Bike Rentals", x = "Weather", y = "Frequency")

temp_plot <- ggplot(data = bike_data, aes(x = temp)) +
  geom_histogram(color = "black", bins = 15, fill = "steelblue") +
  labs(title = "Temperature and Bike Rentals", x = "Temp (C)", y = "Frequency")

humid_plot <- ggplot(data = bike_data, aes(x = humidity)) +
  geom_histogram(color = "black", fill = "navy", bins = 15) +
  labs(title = "Humidity and Bike Rentals", x = "Humidity Levels", y = "Frequency")

wind_plot <- ggplot(data = bike_data, aes(x = windspeed)) +
  geom_histogram(color = "black", bins = 20, fill = "skyblue") +
  labs(title = "Windspeed and Bike Rentals", x = "Windspeed", y = "Frequency")

(weather_plot + temp_plot) / (humid_plot + wind_plot)


# note that test dataset has 1 category 4 while training dataset does not 

###### CLEANING DATA ######

# drop casual & registered from training dataset 
bike_data <- bike_data %>% select(-casual, -registered)

# change cont to log (count)
log_data <- bike_data %>% 
                mutate(log_count = log(count))
# drop count
log_data <- log_data %>% select(-count)

# recipe
bike_recipe <- recipe(log_count ~ . , data = log_data) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = as.factor(weather)) %>%
  step_time(datetime, features= "hour") %>%
  step_mutate(season = as.factor(season)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) 

prepped_recipe <- prep(bike_recipe)
baked_data <- bake(prepped_recipe, new_data = log_data)

glimpse(baked_data)

###### LINEAR REGRESSION ######


# linear model (1st set of predictions)
my_linear_model <- linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression") %>% 
  fit(formula = count ~ . - datetime, data = bike_data)


# linear model (2nd set of predictions)
lin_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")


###### WORKFLOW ######
bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(lin_model) %>%
  fit(data = log_data)


###### PREDICTIONS ######

# 1st set of predictions
bike_predictions <- predict(my_linear_model, new_data = testData)


# 2nd set using workflow 
lin_preds <- predict(bike_workflow, new_data = testData)

# backtransform 2nd set of predictions
lin_preds <- lin_preds %>%
  mutate(count_pred = exp(.pred))



###### PENALIZED REGRESSION MODEL ######


# Fix grid code 


# recipe
pen_recipe <- recipe(log_count ~ . , data = log_data) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = as.factor(weather)) %>%
  step_time(datetime, features= "hour") %>%
  step_rm(datetime) %>%
  step_mutate(season = as.factor(season)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) 

prepped_pen_recipe <- prep(pen_recipe)
pen_data <- bake(prepped_pen_recipe, new_data = log_data)

# model 
# for the 5 different sets of predictions, I just changed the parameters and re-ran my code 
pen_model <- linear_reg(penalty = tune(), 
                        mixture = tune()) %>% #Set model and tuning
                      set_engine("glmnet")  # Function to fit in R
  
# workflow  
pen_workflow <- workflow() %>%
    add_recipe(pen_recipe) %>%
    add_model(pen_model)

# grid of values to tune 
grid_of_tuning_params <- grid_regular(penalty(),
                              mixture(),
                              levels = 5)

# split data for cv & run it 
folds <- vfold_cv(log_data, v = 5, repeats=1)

CV_results <- pen_workflow %>%
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(rmse, mae))

## plot results
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric == "rmse") %>%
  ggplot(data =., aes(x = penalty, y = mean, color = factor(mixture))) +
  geom_line()
  
# find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "rmse")

## finalize the workflow & fit 
final_wf <- pen_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = log_data) 

## predictions
pen_preds <- final_wf %>% predict(new_data = testData)

# back transform
pen_preds <- pen_preds %>%
  mutate(.pred = exp(.pred))





###### TREE MODEL ######

tree_mod <- decision_tree(tree_depth = tune(),
                          cost_complexity = tune(),
                          min_n=tune()) %>% #Type of model
  set_engine("rpart") %>% # What R function to use
  set_mode("regression")


tree_recipe <- recipe(log_count ~ . , data = log_data) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = as.factor(weather)) %>%
  step_time(datetime, features= "hour") %>%
  step_rm(datetime) %>%
  step_mutate(season = as.factor(season)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) 

prepped_tree_recipe <- prep(tree_recipe)
tree_data <- bake(prepped_tree_recipe, new_data = log_data)


# workflow  
tree_workflow <- workflow() %>%
  add_recipe(tree_recipe) %>%
  add_model(tree_mod)

# grid of values to tune 
grid_of_tuning_params <- grid_regular(cost_complexity(),
                                      tree_depth(),
                                      min_n(),
                                      levels = 3)

# split data for cv & run it 
folds <- vfold_cv(log_data, v = 5, repeats=1)

CV_results <- tree_workflow %>%
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(rmse, mae))



# find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "rmse")

## finalize the workflow & fit 
final_wf <- tree_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = log_data) 

## predictions
tree_preds <- final_wf %>% predict(new_data = testData)

# back transform
tree_preds <- tree_preds %>%
  mutate(.pred = exp(.pred))




###### RANDOM FOREST MODEL ######

# model using random forest
forest_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>% #Type of model
                  set_engine("ranger") %>% # What R function to use
                  set_mode("regression")


# forest recipe
forest_recipe <- recipe(log_count ~ . , data = log_data) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = as.factor(weather)) %>%
  step_time(datetime, features= "hour") %>%
  step_rm(datetime) %>%
  step_mutate(season = as.factor(season)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) 

prepped_forest_recipe <- prep(forest_recipe)
forest_data <- bake(prepped_forest_recipe, new_data = log_data)


# workflow  
forest_workflow <- workflow() %>%
  add_recipe(forest_recipe) %>%
  add_model(forest_mod)

# grid of values to tune 
grid_of_tuning_params <- grid_regular(mtry(range = c(1,10)),
                              min_n(),
                              levels = 3) 

# split data for cv & run it 
folds <- vfold_cv(log_data, v = 5, repeats=1)

CV_results <- forest_workflow %>%
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(rmse, mae))



# find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "rmse")

## finalize the workflow & fit 
final_wf <- forest_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = log_data) 

## predictions
forest_preds <- final_wf %>% predict(new_data = testData)

# back transform
forest_preds <- forest_preds %>%
  mutate(.pred = exp(.pred))



###### BART MODEL #######

# BART model
bart_mod <- parsnip::bart(mode = "regression") %>%
  set_engine("dbarts") %>%
  set_args(
    trees = tune())  


# bart recipe
bart_recipe <- recipe(log_count ~ . , data = log_data) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = as.factor(weather)) %>%
  step_time(datetime, features = ("hour")) %>% #create time variable
  step_date(datetime, features = "dow") %>% # gets day of week
  step_rm(datetime) %>%
  step_mutate(season = as.factor(season)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) 


prepped_bart_recipe <- prep(bart_recipe)
bart_data <- bake(prepped_bart_recipe, new_data = log_data)


# workflow  
bart_workflow <- workflow() %>%
  add_recipe(bart_recipe) %>%
  add_model(bart_mod)

# grid of values to tune 
grid_of_tuning_params <- grid_regular(
  trees(range = c(50, 100)),
  levels = 5)


# split data for cv & run it 
folds <- vfold_cv(log_data, v = 5, repeats=1)

CV_results <- bart_workflow %>%
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(rmse, mae))



# find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "rmse")

## finalize the workflow & fit 
final_wf <- bart_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = log_data) 

## predictions
bart_preds <- final_wf %>% predict(new_data = testData)

# back transform
bart_preds <- bart_preds %>%
  mutate(.pred = exp(.pred))



###### META LEARNER MODEL ######
h2o::h2o.init()

auto_model <- auto_ml() %>%
  set_engine("h2o", max_runtime_secs = 180, max_models = 25) %>%
  set_mode("regression")


meta_recipe <- recipe(log_count ~ . , data = log_data) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = as.factor(weather)) %>%
  step_time(datetime, features= "hour") %>%
  step_rm(datetime) %>%
  step_mutate(season = as.factor(season)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) 

prepped_meta_recipe <- prep(meta_recipe)
meta_data <- bake(prepped_meta_recipe, new_data = log_data)

automl_wf <- workflow() %>%
  add_recipe(meta_recipe) %>%
  add_model(auto_model) %>%
  fit(data = log_data)


## predictions
meta_preds <- automl_wf %>% predict(new_data = testData)

# back transform
meta_preds <- meta_preds %>%
  mutate(.pred = exp(.pred))





# submit to kaggle (change code accordingly for prediction set)
kaggle_submission <- bart_preds %>%
  bind_cols(., testData) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

# write up file for kaggle
vroom_write(x = kaggle_submission, file = "./BARTPredsCV.csv", delim=",")











