library(tidyverse)
library(dslabs)
data("movielens")

movielens %>% 
  summarise(n_users = n_distinct(userId), 
            n_movies = n_distinct(movieId))
#Output shows they are 671 unique users and 9066 unique movies rated.

library(caret)
set.seed(755)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.2, list = FALSE)
train_set <- movielens[-test_index,]
test_set <- movielens[test_index,]

#We use the semi_join function so that we ensure user and movies not in the training set are not in the test set
test_set <- test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")


RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Using a linear model - model_1
mu <- mean(train_set$rating)
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

user_avgs <- train_set %>% 
  left_join(movie_avgs, by = 'movieId') %>% 
  group_by(userId) %>% 
  summarise(b_u = mean(rating - mu - b_i))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by = 'movieId') %>% 
  left_join(user_avgs, by = 'userId') %>% 
  mutate(pred = mu + b_i + b_u) %>% 
  pull(pred)

model_1_rmse <- RMSE(test_set$rating, predicted_ratings)
rmse_results <- data_frame(method = "Linear Model: Movie + User Effects", RMSE = model_1_rmse )
rmse_results


#Regularization
#We use cross-validation to choose the tuning parameter
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <-
    test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})
qplot(lambdas, rmses)
lambdas[which.min(rmses)]

rmse_results <- bind_rows(rmse_results, 
                          data_frame(method = "Regularized Movie + User Effect Model", 
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()

##############################################################################

#now train a decision tree, while transforming the ratings into categorical data 
#so that the decision tree gives us categorical data as well.


# train_rpart <- train(as.character(rating) ~ ., 
#                      method = "rpart", data = train_set)



