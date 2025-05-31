# 1. Load necessary libraries
library(dplyr)
library(caret)         # for ML training
library(ggplot2)       # for visualization

# 2. Load your dataset
clients <- read.csv("dq_recsys_challenge_2025(in).csv")

# 3. Clean / select important features
# Keep only relevant columns: idcol, segment, tod, active_ind
clients_clean <- clients %>%
  select(idcol, segment, tod, active_ind) %>%
  filter(!is.na(segment), !is.na(tod), !is.na(active_ind))  # remove missing values

# 4. Convert target variable to factor (classification)
clients_clean$active_ind <- as.factor(clients_clean$active_ind)

# 5. Split into training and testing sets
set.seed(42)  # for reproducibility
trainIndex <- createDataPartition(clients_clean$active_ind, p = 0.8, list = FALSE)
train_data <- clients_clean[trainIndex, ]
test_data <- clients_clean[-trainIndex, ]

#Sample of 10000 
clients_sample <- clients_clean %>% sample_n(100000)

# 6. Train a model (Random Forest for classification)
#model <- train(active_ind ~ segment + tod, data = train_data, method = "rf",  trControl = trainControl(method = "cv", number = 5), importance = TRUE)

#Sample data testing 
trainIndex <- createDataPartition(clients_sample$active_ind, p = 0.8, list = FALSE)
train_data <- clients_sample[trainIndex, ]
test_data <- clients_sample[-trainIndex, ]


#Train model for sample size decision tree
model <- train(
  active_ind ~ segment + tod,
  data = train_data,
  method = "rpart",  # fast decision tree
  trControl = trainControl(method = "cv", number = 3)
)


# 7. View model results
print(model)

# 8. Make predictions on test set
predictions <- predict(model, newdata = test_data)

# 9. Evaluate accuracy
conf_matrix <- confusionMatrix(predictions, test_data$active_ind)
print(conf_matrix)

# 10. Optional: Visualize predicted vs actual
ggplot(test_data, aes(x = active_ind, fill = predictions)) +
  geom_bar(position = "dodge") +
  labs(title = "Actual vs Predicted Activity Status", x = "Actual", fill = "Predicted") +
  theme_minimal()


# Predict for a new user
new_user <- data.frame(segment = "segment2", tod = "Afternoon")
predict(model, new_user)

