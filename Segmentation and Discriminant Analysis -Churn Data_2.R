getwd()
library (tree)
library (ISLR)
library(rpart)
library(caret)
library(dplyr)

ChurnData <- read.csv("Churn_Modelling.csv", header = T)
head(ChurnData)
glimpse(ChurnData)

sapply(ChurnData, function(x) sum(is.na(x)))
# RowNumber      CustomerId         Surname     CreditScore       Geography          Gender 
# 0               0               0               0               0               0 
# Age          Tenure         Balance   NumOfProducts       HasCrCard  IsActiveMember 
# 0               0               0               0               0               0 
# EstimatedSalary          Exited 
# 0               0 

table(ChurnData$HasCrCard) # 1 or 0
table(ChurnData$IsActiveMember) # 1 or 0
table(ChurnData$NumOfProducts) # 1,2,3,4

# Create factor variables 
ChurnData2 <- ChurnData[c("CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary","Exited")]
glimpse(ChurnData2)

ChurnData3 <- ChurnData2 %>%
  mutate(Exited = factor(Exited, levels = c(1,0), labels = c("Yes","No")),
         IsActiveMember = factor(IsActiveMember, levels = c(1,0), labels = c("Yes","No")),
         HasCrCard = factor(HasCrCard, levels = c(1,0), labels = c("Yes","No")),
         Gender = factor(Gender),
         Geography = factor(Geography))

glimpse(ChurnData3)

# Test and train datasets
trainrows = sample(1:nrow(ChurnData3), 5000)
traindata = ChurnData3[trainrows,]
testdata = ChurnData3[-trainrows,]
dim(traindata)
# 5000  11
dim(testdata)
# 5000  11

####################### MODEL #############################
ModelChurnTree = tree(formula= Exited~ . , data = traindata )
summary(ModelChurnTree)
# Classification tree:
#  tree(formula = Exited ~ ., data = traindata)
# Variables actually used in tree construction:
#   [1] "Age"            "NumOfProducts"  "IsActiveMember"
# Number of terminal nodes:  8 
# Residual mean deviance:  0.7607 = 3798 / 4992 
# Misclassification error rate: 0.1532 = 766 / 5000

# Deviance means here the mean squared error.

plot(ModelChurnTree)
text(ModelChurnTree, pretty = 0)

#Test the predictions of the trained model
testpredictions = predict(ModelChurnTree, testdata, type ="class")
# 'class': for classification			
# 'prob': to compute the probability of each class			
# 'vector': Predict the mean response at the node level

#Find the accuracy of the model
cm =table(testpredictions, testdata$Exited)
print(cm)
# testpredictions  Yes   No
#             Yes  313  101
#             No   690 3896

#Display the accuracy of the model in test data
accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
print(accuracy)
# 0.8418


# Cross validate to see whether pruning the tree will improve performance
cv.ModelChurnTree <- cv.tree(ModelChurnTree)
plot(cv.ModelChurnTree)
# It seems like the 7th sized trees result in the lowest deviance. We can prune the tree. 

####################### 1 #############################
prune.trees <- prune.tree(ModelChurnTree, best=4)
summary(prune.trees)
# Classification tree:
#  snip.tree(tree = ModelChurnTree, nodes = c(6L, 7L, 4L))
# Variables actually used in tree construction:
#   [1] "Age"            "NumOfProducts"  "IsActiveMember"
# Number of terminal nodes:  4
# Residual mean deviance:  0.8461 = 4227 / 4996 
# Misclassification error rate: 0.1768 = 884 / 5000 

# Residual mean deviance increased. I started pruning with 7 trees. And decreased to 4. 
# Because Residual mean deviance is increasing when I decrease the tree level but Misclassification error rate increases. Optimum level is 4.

plot(prune.trees)
text(prune.trees, pretty=0)

#Test the predictions of the trained model
testpredictions2 = predict(prune.trees, testdata, type ="class")
# 'class': for classification			
# 'prob': to compute the probability of each class			
# 'vector': Predict the mean response at the node level

#Find the accuracy of the model
cm2 =table(testpredictions2, testdata$Exited)
print(cm2)
# testpredictions2  Yes   No
#               Yes  448  347
#               No   555 3650

#Display the accuracy of the model in test data
accuracy2 = (cm2[1,1] + cm2[2,2]) / (cm2[1,1] + cm2[2,2] + cm2[1,2] + cm2[2,1])
print(accuracy2)
# 0.8196
# Accuracy decreased compared to first model. "Yes" and "No" decreased


####################### 2 #############################
prune.trees2 <- prune.tree(ModelChurnTree, best=6)
summary(prune.trees2)
# Classification tree:
#  snip.tree(tree = ModelChurnTree, nodes = 6L)
# Variables actually used in tree construction:
#   [1] "Age"            "NumOfProducts"  "IsActiveMember"
# Number of terminal nodes:  6
# Residual mean deviance:  0.7929 = 3960 / 4994 
# Misclassification error rate: 0.1608 = 804 / 5000 

#Test the predictions of the trained model
testpredictions3 = predict(prune.trees2, testdata, type ="class")
# 'class': for classification			
# 'prob': to compute the probability of each class			
# 'vector': Predict the mean response at the node level

#Find the accuracy of the model
cm3 =table(testpredictions3, testdata$Exited)
print(cm3)
# testpredictions2  Yes   No
#               Yes  283   96
#               No   720 3901

#Display the accuracy of the model in test data
accuracy3 = (cm3[1,1] + cm3[2,2]) / (cm3[1,1] + cm3[2,2] + cm3[1,2] + cm3[2,1])
print(accuracy3)
# 0.8368
# Accuracy decreased compared to first model. "Yes" decreased




