#Install all the libraries
# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(gdata)) install.packages("gdata", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot", repos = "http://cran.us.r-project.org")
if(!require(PET)) install.packages("PET", repos = "http://cran.us.r-project.org")
if(!require(ggfortify)) install.packages("ggfortify", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")

#UCI HAR DATASET
#https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

#Download and Unzip the files
dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip", dl)
unzip (zipfile=dl)
unlink(dl)


#PreProcess the data to extract the train set, test set, the subject ID and the activity labels
train_x <-read.table("UCI HAR Dataset/train/X_train.txt")
train_y <-read.table("UCI HAR Dataset/train/Y_train.txt")
subjectId_train<-readData("UCI HAR Dataset/train/subject_train.txt")
test_x<-read.table("UCI HAR Dataset/test/X_test.txt")
test_y<-read.table("UCI HAR Dataset/test/Y_test.txt")
subjectId_test<-readData("UCI HAR Dataset/test/subject_test.txt")

column_names <- readLines("UCI HAR Dataset/features.txt")
colnames(train_x) <- make.names(column_names)
colnames(test_x)<-make.names(column_names)
colnames(train_y)<-"activity"
colnames(test_y)<-"activity"
train_lab<-train_y$activity - 1
test_lab<-test_y$activity - 1
train_y$activity <- as.factor(make.names(train_y$activity))
test_y$activity <- as.factor(make.names(test_y$activity))

# Keep x and y matrices and the lab vectors for classification, but also get a composite dataset for test and train
train_set<-transform(train_x, subjectID = factor(subjectId_train), activity = factor(train_y$activity))

test_set <- transform(test_x, subjectID = factor(subjectId_test), activity = factor(test_y$activity))

all_data<-dplyr::bind_rows(train_set, test_set, .id = "source")
all_data$activity<-factor(all_data$activity)


# Plot the data
qplot(data = all_data, x = subjectID, fill = activity)

qplot(data = all_data, x = subjectID, fill = source)

#Calculate and plot the SVD components
svd1 = svd(scale(train_set[, -which(names(train_set) %in% c("activity","subjectID"))]))
par(mfrow = c(1, 2))
plot(svd1$u[, 1], col = train_set$activity, pch = 19)
plot(svd1$u[, 2], col = train_set$activity, pch = 19)



#Calculate and plot the PCA components
a <- autoplot(prcomp(train_set[, -which(names(train_set) %in% c("activity","subjectID"))]), colour = as.factor(train_lab+1))
a + scale_color_manual(values = c("#FF1BB3","#A7FF5B","#99554D", "#e6daa6",
                                 "#ff474c", "#b0dd16"))

pca<-prcomp(train_set[, -which(names(train_set) %in% c("activity","subjectID"))])
PCi<-data.frame(pca$x,activity=train_set$activity)

ggplot(PCi,aes(x=PC1,y=PC2,col=as.factor(activity)))+
  geom_point(size=3,alpha=0.5) + 
  scale_color_manual(values = c("#FF1BB3","#A7FF5B","#99554D", "#e6daa6",
                                "#ff474c", "#b0dd16"))+
  theme_classic()


new_d <- cbind(train_set, pca$x)
ggplot(new_d, aes(x=activity, y=PC1)) + 
       geom_boxplot()
ggplot(new_d, aes(x=activity, y=PC3)) + 
            geom_boxplot()

var_exp = cumsum(pca$sdev^2 / sum(pca$sdev^2))

# plot percentage of variance explained for each principal component    
barplot(100*var_exp[1:10], las=2, xlab='', ylab='% Variance Explained')


#Set the control parameters for cross validation
control_parameters <- trainControl(method = "cv",
                                  number = 5
)

#Train the slda model
model_slda <- train(train_x, as.factor(train_lab), method="slda", tuneControl = control_parameters)
prediction_slda<-predict(model_slda, train_x)
acc_slda <- confusionMatrix(table(prediction_slda, train_lab))$overall["Accuracy"]
table(prediction_slda, train_lab)

acc <- tibble(method = "Stabilized linear Discriminant Analysis",
                  ACC = acc_slda)


#Train the MLP model
model_mlp <- train(train_x, as.factor(train_lab), method="mlp", tuneControl = control_parameters, tuneGrid = data.frame(size=c(3,4)))
prediction_mlp <- predict(model_mlp, train_x)
acc_mlp <- confusionMatrix(table(prediction_mlp, train_lab))$overall["Accuracy"]
table(prediction_mlp, train_lab)
acc <- bind_rows(acc,
          data_frame(method="Multi Layer Perceptron",  
                     ACC = acc_mlp))


#Train the svm model
model_svm <- train(train_x, as.factor(train_lab), method="svmLinear2", tuneControl = control_parameters, tuneGrid = expand.grid(cost=2), allowParallel=TRUE)
prediction_svm <- predict(model_svm, train_x)
acc_svm<- confusionMatrix(table(prediction_svm, train_lab))$overall["Accuracy"]
table(prediction_svm, train_lab)
acc <- bind_rows(acc,
                 data_frame(method="Support Vector Machine",  
                            ACC = acc_svm))

#Set up parameter grid for training the xgboost model
parametersGrid <-  expand.grid(eta = 0.1, 
                               colsample_bytree=c(0.5,0.7),
                               max_depth=c(3,6),
                               nrounds=100,
                               gamma=1,
                               min_child_weight=2,
                               subsample=0.5
)


modelxgboost <- train(train_set[, -which(names(train_set) %in% c("activity","subjectID"))],
                      as.factor(train_set$activity),
                      method = "xgbTree",
                      trControl = control_parameters,
                      tuneGrid=parametersGrid)

prediction_xg <- predict(modelxgboost, train_x)
acc_xgb<-confusionMatrix(table(prediction_xg, train_set$activity))$overall["Accuracy"]
table(prediction_xg, train_lab)
acc <- bind_rows(acc,
                 data_frame(method="Gradient Boosted Trees",  
                            ACC = acc_xgb))

modelxgboost

#Train the final xgboost model since it was the best based on the paramters found above.
xgb <- xgboost(data.matrix(train_x), train_lab,
               eta = 0.1,
               max_depth = 6, 
               min_child_weight=2,
               nround=100, 
               gamma=1,
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               nthread = 3,
               objective='multi:softmax',
               num_class=6
)

model <- xgb.dump(xgb, with_stats = T)
model[1:10]
y_pred <- predict(xgb, data.matrix(test_x))

# Lets start with finding what the actual tree looks like
model <- xgb.dump(xgb, with.stats = T)
model[1:10] #This statement prints top 10 nodes of the model
# Get the feature real names
names <- dimnames(data.matrix(train_x))[[2]]
# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = xgb)
# Nice graph
xgb.plot.importance(importance_matrix[1:10,])

#Print and save the final accuracy and save the worspace
xgb_confusion <- confusionMatrix(table(y_pred, test_lab))
xgb_confusion$table
xgb_confusion$overall["Accuracy"]

save.image(file = "workspaceImage.RData")
