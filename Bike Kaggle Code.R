library(tidyverse)
library(tidymodels)
library(vroom)
library(DataExplorer)
library(patchwork)

# read in data
bike_data <- vroom("/Users/eliseclark/Documents/FALL 2025/STAT 348/BikeShareComp/bike-sharing-demand/train.csv")

# change weather to a factor 
bike_data <- bike_data %>%
  mutate(weather = as.factor(weather), 
         season = as.factor(season),
         workingday = as.factor(workingday), 
         holiday = as.factor(holiday))

# look at variable types & num of obs
glimpse(bike_data)



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




###### LINEAR REGRESSION ######

# drop casual & registered from training dataset 
bike_data <- bike_data %>% select(-casual, -registered)

# read in test data
testData <- vroom("/Users/eliseclark/Documents/Fall 2025/Stat 348/BikeShareComp/bike-sharing-demand/test.csv")

# set factors in test data 
testData <- testData %>%
  mutate(weather = as.factor(weather), 
         season = as.factor(season),
         workingday = as.factor(workingday), 
         holiday = as.factor(holiday))

# linear model
my_linear_model <- linear_reg() %>% #Type of model
  set_engine("lm") %>% # Engine = What R function to use
  set_mode("regression") %>% # Regression just means quantitative response
  fit(formula = count ~ . - datetime, data = bike_data)


# predictions
bike_predictions <- predict(my_linear_model, new_data = testData)


kaggle_submission <- bike_predictions %>%
  bind_cols(., testData) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count = .pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count = pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

# write up file for kaggle
vroom_write(x = kaggle_submission, file = "./LinearPreds.csv", delim=",")













