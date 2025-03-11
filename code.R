#loading libraries
library(data.table) 
library(Rtsne)
library(ggplot2)
library(gridExtra)
library(caret) 
library(glmnet)
library(dplyr)
library(arules)
library(arulesViz)
library(recommenderlab)
library(biglm)


#read the data set
data <- fread("D:/Downloads/Study/data mining project/archive/Spotify_Youtube.csv")

#Data preprocessing
#check if there have any missing value
sum(is.na(data))

#drop the row with missing value
data <- data[complete.cases(data), ]

#drop the variable is not useful for this project
data <- subset(data, select = c(Artist, Track, Album, Album_type, Danceability, Energy, Key, Loudness, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, Duration_ms, Stream, Title, Channel, Views, Likes, Comments, Licensed, official_video ))

#Convert categorical variables to numerical format
data$Licensed <- ifelse(data$Licensed == "No", 0, 1)
data$official_video <- as.integer(data$official_video)

#Normalize the numerical variables (for k-means clustering)
num_cols <- c("Danceability", "Energy", "Key", "Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo", "Duration_ms")
data[, (num_cols) := lapply(.SD, scale), .SDcols = num_cols]

# Fit t-SNE to the selected columns
tsne_obj <- Rtsne(as.matrix(data[, ..num_cols]), dims = 2, perplexity = 30, verbose = TRUE, check_duplicates = FALSE)

# Convert the t-SNE result to a data table and add it to the original data table
tsne_data <- data.table(tsne_obj$Y)
setnames(tsne_data, c("tsne_x", "tsne_y"))
data <- cbind(data, tsne_data)

# Define colors for each album type
colors <- c("album" = "#1f77b4", "single" = "#ff7f0e", "compilation" = "#2ca02c")

# Create a scatterplot of the preprocessed data before t-SNE (using the first two numerical columns)
p1 <- ggplot(data, aes(x = Danceability, y = Energy, color = Album_type)) +
  geom_point(size = 2) +
  scale_color_manual(values = colors) +
  ggtitle("Relationship Between Danceability and Energy, Colored by Album Type (Before t-SNE)") +
  xlab("Danceability (scaled)") +
  ylab("Energy (scaled)")

# Create a scatterplot of the t-SNE result
p2 <- ggplot(data, aes(x = tsne_x, y = tsne_y, color = Album_type)) +
  geom_point(size = 2) +
  scale_color_manual(values = colors) +
  ggtitle("Relationship Between t-SNE Components and Album Type (After t-SNE)") +
  xlab("t-SNE Component 1") +
  ylab("t-SNE Component 2")

# Combine the two scatterplots into a single figure using the gridExtra package
grid.arrange(p1, p2, ncol = 2)

# Split the data into training and testing sets
set.seed(123)
trainIndex <- sample(nrow(data), floor(0.8 * nrow(data)))
training <- data[trainIndex, c("Danceability", "Energy", "Key", "Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo", "Stream", "Views", "Likes", "Licensed", "official_video", "Artist", "Album", "Album_type", "Duration_ms")]
testing <- data[-trainIndex, c("Danceability", "Energy", "Key", "Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo", "Stream", "Views", "Likes", "Licensed", "official_video", "Artist", "Album", "Album_type", "Duration_ms")]

# Convert integer64 data to numeric
training$Stream <- as.numeric(training$Stream)
training$Views <- as.numeric(training$Views)
training$Likes <- as.numeric(training$Likes)
testing$Stream <- as.numeric(testing$Stream)
testing$Views <- as.numeric(testing$Views)
testing$Likes <- as.numeric(testing$Likes)

# Fit linear regression models separately
model1_views <- lm(Views ~ Danceability + Energy + Key + Loudness + Speechiness + Acousticness + Instrumentalness + Liveness + Valence + Tempo, data = training)
model1_stream <- lm(Stream ~ Danceability + Energy + Key + Loudness + Speechiness + Acousticness + Instrumentalness + Liveness + Valence + Tempo, data = training)

# Use models to make predictions on testing data
pred1_views <- predict(model1_views, newdata = testing)
pred1_stream <- predict(model1_stream, newdata = testing)

# Compute R-squared values for Model 1
r_squared1_views <- summary(model1_views)$r.squared
r_squared1_stream <- summary(model1_stream)$r.squared

# Print results for Model 1
cat("Model 1 (song features):\n")
cat("Coefficients for Views:\n")
print(coef(model1_views))
cat("R-squared value for Views:", r_squared1_views, "\n\n")
cat("Coefficients for Stream:\n")
print(coef(model1_stream))
cat("R-squared value for Stream:", r_squared1_stream, "\n\n")

new_song <- data.frame(Danceability = 0.8,
                       Energy = 0.1,
                       Key = 7,
                       Loudness = 5.2,
                       Speechiness = 0.01,
                       Acousticness = 0.05,
                       Instrumentalness = 0.1,
                       Liveness = 0.1,
                       Valence = 0.6,
                       Tempo = 60)

# Use Model 1 to make a prediction for the new song
new_song_pred <- predict(model1_views, newdata = new_song)

# Print the predicted number of views for the new song
cat("Predicted number of views for the new song:", new_song_pred, "\n")


# Create a scatter plot of predicted versus actual values for Views
plot(x = pred1_views, y = testing$Views, xlab = "Predicted Views", ylab = "Actual Views", main = "Predicted vs. Actual Views")

# Add a diagonal line to the plot to show perfect correlation
abline(a = 0, b = 1, col = "red")

# Clustering Analysis: K-means clustering to group similar songs together based on their musical features

# Select the numerical columns for clustering
num_cols <- c("Danceability", "Energy", "Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Liveness", "Valence")

# Create a scatterplot of the t-SNE result before clustering
p0 <- ggplot(data, aes(x = tsne_x, y = tsne_y, color = Album_type)) + 
  geom_point(size = 2) +
  scale_color_manual(values = colors) +
  ggtitle("Before Clustering")

# Print the scatterplot
print(p0)

# Perform k-means clustering with k = 4, using 2 iterations with different starting centroids
set.seed(123)
k <- 4
nstart <- 2
kmeans_objs <- list()
for (i in 1:nstart) {
  kmeans_objs[[i]] <- kmeans(data[, ..num_cols], centers = k, nstart = nstart)
}

# Add the cluster assignments to the original data table for each iteration
for (i in 1:nstart) {
  data[, paste0("cluster_", i) := kmeans_objs[[i]]$cluster]
}

# Define colors and symbols for each cluster
colors <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
symbols <- c(15, 16, 17, 18)

# Create a scatterplot of the t-SNE result, colored by cluster for each iteration
p_list <- list()
for (i in 1:nstart) {
  p_list[[i]] <- ggplot(data, aes(x = tsne_x, y = tsne_y, color = factor(get(paste0("cluster_", i))), shape = factor(get(paste0("cluster_", i))))) + 
    geom_point(size = 3) +
    scale_color_manual(values = colors) +
    scale_shape_manual(values = symbols) +
    ggtitle(paste0("Iteration ", i, " - K-means Clustering")) +
    theme(legend.position = "bottom")
}

# Print the scatterplots for each iteration
for (p in p_list) {
  print(p)
}

# Convert the relevant variables to factors
data$Artist <- as.factor(data$Artist)
data$Album_type <- as.factor(data$Album_type)
data$Genre <- as.factor(data$Genre)

# Create a transaction dataset
trans_data <- as(data[, c("Artist", "Album_type", "Genre"), with = FALSE], "transactions")

# Explore the transaction dataset
summary(trans_data)

# Mine association rules
rules <- apriori(trans_data, parameter = list(support = 0.0002, confidence = 0.3))

# Explore the mined rules
summary(rules)
inspect(rules)

# Visualize the association rules using a scatter plot
plot(rules, method = "scatterplot")

# Convert the data to a sparse matrix format
rating_matrix <- as(data, "realRatingMatrix")

# Split the data into training and testing sets
set.seed(123)
sampled_ratings <- sample(x = c(TRUE, FALSE), size = nrow(rating_matrix), replace = TRUE, prob = c(0.8, 0.2))
train_ratings <- rating_matrix[sampled_ratings]
test_ratings <- rating_matrix[!sampled_ratings]

# Build the recommendation model using collaborative filtering
model <- Recommender(train_ratings, method = "UBCF")

# Generate recommendations for the test set
recommendations <- predict(model, test_ratings, n = 5)

# Print the top 5 recommended items for each user in the test set
print(recommendations@items)

ggplot(training_df, aes(x = Danceability, y = Views)) + 
  geom_point() +
  labs(title = "Relationship between Danceability and Views", x = "Danceability", y = "Views")

cluster_sizes <- data[, .N, by = cluster_1]
ggplot(cluster_sizes, aes(x = factor(cluster_1), y = N)) + 
  geom_bar(stat = "identity", fill = colors) +
  labs(title = "Cluster Sizes", x = "Cluster", y = "Size")

