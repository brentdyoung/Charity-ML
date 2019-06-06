# PREDICT 422 Practical Machine Learning

# Course Project

# OBJECTIVE: A charitable organization wishes to develop a machine learning
# model to improve the cost-effectiveness of their direct marketing campaigns
# to previous donors.

# 1) Develop a classification model using data from the most recent campaign that
# can effectively capture likely donors so that the expected net profit is maximized.

# 2) Develop a prediction model to predict donation amounts for donors - the data
# for this will consist of the records for donors only.

#Load Libraries

library(Hmisc)
library(pROC) #ROC Curve
library(ROCR) #ROC Curve, AUC
library(AUC)
library(Deducer) #ROC Curve
library(InformationValue) #ROC Curve
library(corrplot) #Data Visualization
library(ggcorrplot) #Data Visualization
library (ggplot2) #Data Visualization
library(GGally) #Data Visualization
library(lattice)#Data Visualization
library(gridExtra)
library(gmodels)#Cross-tabs
library(dplyr)
library(reshape2)
library(readr)
library(glm2)
library(aod)
library(rcompanion) 
library(leaps) #Best subsets
library(MASS) #Linear Discriminant Analysis
library(car)

library(boot) #bootstrap
library(leaps) #Best subset selection; stepwise
library(glmnet) #Ridge Regression & the Lasso
library(pls) #Principal components Regression
library(splines) #regression splines
library(gam) #Generalized Additive Models
library(akima)
library(caret) #Machine Learning
library(yardstick) #Machine Learning
library(class) #KNN
library(neuralnet)  # Neural Network
library(nnet) # Neural Network
library(tree) # Decision Trees 
library(randomForest) # Bagging and Random Forest
library(gbm) # Gradient Boosting Machines
library(xgboost) # Gradient Boosting Machines
library(e1071) #Support Vector Machines, Naive Bayes Classifier

#Load the data

setwd("~/R/MSDS 422/Course Project")
charity <- read.csv("charity.csv") # load the "charity.csv" file

################################# Quick High Level EDA for Overall Dataset #################################

#Descriptive Statistics
str(charity)
summary(charity)
describe(charity)
dim(charity)

#Factor Variables
charity$reg1 <- as.factor(charity$reg1)
charity$reg2 <- as.factor(charity$reg2)
charity$reg3 <- as.factor(charity$reg3)
charity$reg4 <- as.factor(charity$reg4)
charity$home <- as.factor(charity$home)
charity$hinc <- as.factor(charity$hinc)
charity$genf <- as.factor(charity$genf)
charity$wrat <- as.factor(charity$wrat)
charity$donr <- as.factor(charity$donr)

#Descriptive Statistics
str(charity)
summary(charity)
describe(charity)
dim(charity)

######EDA for Numeric Variables#####

par(mfrow=c(2,2))
hist(charity$chld, col = "#09ADAD", xlab = "chld", main = "Histogram of chld")
hist(charity$avhv, col = "#DBCEAC", xlab = "avhv", main = "Histogram of avhv")
boxplot(charity$chld, col = "#09ADAD", main = "Boxplot of chld")
boxplot(charity$avhv, col = "#DBCEAC", main = "Boxplot of avhv")

par(mfrow=c(1,1))

par(mfrow=c(3,3))
hist(charity$incm, col = "#A71930", xlab = "incm", main = "Histogram of incm")
hist(charity$inca, col = "#09ADAD", xlab = "inca", main = "Histogram of inca")
hist(charity$plow, col = "#DBCEAC", xlab = "plow", main = "Histogram of plow")
boxplot(charity$incm, col = "#A71930", main = "Boxplot of incm")
boxplot(charity$inca, col = "#09ADAD", main = "Boxplot of inca")
boxplot(charity$plow, col = "#DBCEAC", main = "Boxplot of plow")
par(mfrow=c(1,1))

par(mfrow=c(3,3))
hist(charity$npro, col = "#A71930", xlab = "npro", main = "Histogram of npro")
hist(charity$tgif, col = "#09ADAD", xlab = "tgif", main = "Histogram of tgif")
hist(charity$lgif, col = "#DBCEAC", xlab = "lgif", main = "Histogram of lgif")

boxplot(charity$npro, col = "#A71930", main = "Boxplot of npro")
boxplot(charity$tgif, col = "#09ADAD", main = "Boxplot of tgif")
boxplot(charity$lgif, col = "#DBCEAC", main = "Boxplot of lgif")
par(mfrow=c(1,1))

par(mfrow=c(2,2))
hist(charity$rgif, col = "#A71930", xlab = "rgif", main = "Histogram of rgif")
hist(charity$tdon, col = "#09ADAD", xlab = " tdon", main = "Histogram of tdon")
boxplot(charity$rgif, col = "#A71930", main = "Boxplot of rgif")
boxplot(charity$tdon, col = "#09ADAD", main = "Boxplot of tdon")
par(mfrow=c(1,1))

par(mfrow=c(2,2))
hist(charity$tlag, col = "#DBCEAC", xlab = "tlag ", main = "Histogram of tlag")
hist(charity$agif, col = "#DBCEAC", xlab = "agif", main = "Histogram of agif")
boxplot(charity$tlag, col = "#DBCEAC", main = "Boxplot of tlag")
boxplot(charity$agif, col = "#DBCEAC", main = "Boxplot of agif")
par(mfrow=c(1,1))

###### EDA for Qualitative Variables#####
library(ggplot2)
#donr
require(ggplot2)
ggplot(charity) +
  geom_bar( aes(donr) ) +
  ggtitle("donr") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#reg1
require(ggplot2)
ggplot(charity) +
  geom_bar( aes(reg1) ) +
  ggtitle("reg1") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#reg2
require(ggplot2)
ggplot(charity) +
  geom_bar( aes(reg2) ) +
  ggtitle("reg2") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#reg3
require(ggplot2)
ggplot(charity) +
  geom_bar( aes(reg3) ) +
  ggtitle("reg3") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#reg4
require(ggplot2)
ggplot(charity) +
  geom_bar( aes(reg4) ) +
  ggtitle("reg4") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#home
require(ggplot2)
ggplot(charity) +
  geom_bar( aes(home) ) +
  ggtitle("home") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#hinc
require(ggplot2)
ggplot(charity) +
  geom_bar( aes(hinc) ) +
  ggtitle("hinc") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#genf
require(ggplot2)
ggplot(charity) +
  geom_bar( aes(genf) ) +
  ggtitle("genf") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#wrat
require(ggplot2)
ggplot(charity) +
  geom_bar( aes(wrat) ) +
  ggtitle("wrat") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#Factor Variables
charity$reg1 <- as.integer(charity$reg1)
charity$reg2 <- as.integer(charity$reg2)
charity$reg3 <- as.integer(charity$reg3)
charity$reg4 <- as.integer(charity$reg4)
charity$home <- as.integer(charity$home)
charity$hinc <- as.integer(charity$hinc)
charity$genf <- as.integer(charity$genf)
charity$wrat <- as.integer(charity$wrat)
charity$donr <- as.integer(charity$donr)

#######CLEAR WORKSPACE######

###################################### Predictor Transformations ######################################

#Load the data

setwd("~/R/MSDS 422/Course Project")
charity <- read.csv("charity.csv") # load the "charity.csv" file

# Add further transformations if desired. 
#For example, some statistical methods can struggle when predictors are highly skewed
charity.t <- charity
charity.t$avhv <- log(charity.t$avhv) #Original

####################################### Set up Data for Analysis ######################################

#Training Dataset
data.train <- charity.t[charity$part=="train",] #splits data into training dataset
x.train <- data.train[,2:21] #includes variables only; removes ID, donr, damt, and part

c.train <- data.train[,22] # donr column (1=Donor, 0=Non-donor)
n.train.c <- length(c.train) # shows 3984 records in training dataset

y.train <- data.train[c.train==1,23] #damt (donation amount) for observations with donr=1
n.train.y <- length(y.train) # shows 1995 donors in training dataset

#Validation Dataset
data.valid <- charity.t[charity$part=="valid",] #splits data into validation dataset
x.valid <- data.valid[,2:21]#includes variables only; removes ID, donr, damt, and part

c.valid <- data.valid[,22] # donr column (1=Donor, 0=Non-donor)
n.valid.c <- length(c.valid) # shows 2018 records in validation dataset 

y.valid <- data.valid[c.valid==1,23] #damt (donation amount) for observations with donr=1
n.valid.y <- length(y.valid) # shows 999 donors in validation dataset

#Test Dataset
data.test <- charity.t[charity$part=="test",] #splits data into test dataset
n.test <- dim(data.test)[1] # shows 2007 records in test dataset 
x.test <- data.test[,2:21] #includes variables only; removes ID, donr, damt, and part

#Standardize
#Training Dataset
x.train.mean <- apply(x.train, 2, mean) #mean of each variable
x.train.sd <- apply(x.train, 2, sd) #standard deviation of each variable

x.train.std <- t((t(x.train)-x.train.mean)/x.train.sd) # standardize to have zero mean and unit sd

apply(x.train.std, 2, mean) # check zero mean
apply(x.train.std, 2, sd) # check unit sd

data.train.std.c <- data.frame(x.train.std, donr=c.train) # used to classify donr
data.train.std.y <- data.frame(x.train.std[c.train==1,], damt=y.train) # used to predict damt when donr=1

#Validation Dataset
x.valid.std <- t((t(x.valid)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.valid.std.c <- data.frame(x.valid.std, donr=c.valid) # used to classify donr
data.valid.std.y <- data.frame(x.valid.std[c.valid==1,], damt=y.valid) # used to predict damt when donr=1

#Test Dataset
x.test.std <- t((t(x.test)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.test.std <- data.frame(x.test.std)

############################## EDA for Classification Models - Training Data ##############################

#Factor Variables
data.train$reg1 <- as.factor(data.train$reg1)
data.train$reg2 <- as.factor(data.train$reg2)
data.train$reg3 <- as.factor(data.train$reg3)
data.train$reg4 <- as.factor(data.train$reg4)
data.train$home <- as.factor(data.train$home)
data.train$hinc <- as.factor(data.train$hinc)
data.train$genf <- as.factor(data.train$genf)
data.train$wrat <- as.factor(data.train$wrat)
data.train$donr <- as.factor(data.train$donr)

#Descriptive Statistics
str(data.train)
summary(data.train)
describe(data.train)
dim(data.train)

######EDA for Numeric Variables#####

par(mfrow=c(2,2))
hist(data.train$chld, col = "#09ADAD", xlab = "chld", main = "Histogram of chld")
hist(data.train$avhv, col = "#DBCEAC", xlab = "avhv", main = "Histogram of avhv")
boxplot(data.train$chld, col = "#09ADAD", main = "Boxplot of chld")
boxplot(data.train$avhv, col = "#DBCEAC", main = "Boxplot of avhv")
par(mfrow=c(1,1))

par(mfrow=c(3,3))
hist(data.train$incm, col = "#A71930", xlab = "incm", main = "Histogram of incm")
hist(data.train$inca, col = "#09ADAD", xlab = "inca", main = "Histogram of inca")
hist(data.train$plow, col = "#DBCEAC", xlab = "plow", main = "Histogram of plow")
boxplot(data.train$incm, col = "#A71930", main = "Boxplot of incm")
boxplot(data.train$inca, col = "#09ADAD", main = "Boxplot of inca")
boxplot(data.train$plow, col = "#DBCEAC", main = "Boxplot of plow")

par(mfrow=c(1,1))

par(mfrow=c(3,3))
hist(data.train$npro, col = "#A71930", xlab = "npro", main = "Histogram of npro")
hist(data.train$tgif, col = "#09ADAD", xlab = "tgif", main = "Histogram of tgif")
hist(data.train$lgif, col = "#DBCEAC", xlab = "lgif", main = "Histogram of lgif")

boxplot(data.train$npro, col = "#A71930", main = "Boxplot of npro")
boxplot(data.train$tgif, col = "#09ADAD", main = "Boxplot of tgif")
boxplot(data.train$lgif, col = "#DBCEAC", main = "Boxplot of lgif")
par(mfrow=c(1,1))

par(mfrow=c(2,2))
hist(data.train$rgif, col = "#A71930", xlab = "rgif", main = "Histogram of rgif")
hist(data.train$tdon, col = "#09ADAD", xlab = " tdon", main = "Histogram of tdon")
boxplot(data.train$rgif, col = "#A71930", main = "Boxplot of rgif")
boxplot(data.train$tdon, col = "#09ADAD", main = "Boxplot of tdon")
par(mfrow=c(1,1))

par(mfrow=c(2,2))
hist(data.train$tlag, col = "#DBCEAC", xlab = "tlag ", main = "Histogram of tlag")
hist(data.train$agif, col = "#DBCEAC", xlab = "agif", main = "Histogram of agif")
boxplot(data.train$tlag, col = "#DBCEAC", main = "Boxplot of tlag")
boxplot(data.train$agif, col = "#DBCEAC", main = "Boxplot of agif")
par(mfrow=c(1,1))

#Outlier Analysis

quantile(data.train$avhv, c(.01, .05, .95, .99))
quantile(data.train$incm, c(.01, .05, .95, .99))
quantile(data.train$inca, c(.01, .05, .95, .99))
quantile(data.train$plow, c(.01, .05, .95, .99))
quantile(data.train$npro, c(.01, .05, .95, .99))
quantile(data.train$tgif, c(.01, .05, .95, .99))
quantile(data.train$lgif, c(.01, .05, .95, .99))
quantile(data.train$rgif, c(.01, .05, .95, .99))
quantile(data.train$tdon, c(.01, .05, .95, .99))
quantile(data.train$tlag, c(.01, .05, .95, .99))
quantile(data.train$agif, c(.01, .05, .95, .99))

#Boxplots for Numeric Variables
ggplot(data.train, aes(x=donr, y= chld)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of chld") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(data.train, aes(x=donr, y= avhv)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of avhv") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(data.train, aes(x=donr, y= incm)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of incm") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(data.train, aes(x=donr, y= inca)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of inca") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(data.train, aes(x=donr, y= plow)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of plow") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(data.train, aes(x=donr, y= npro)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of npro") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(data.train, aes(x=donr, y= tgif)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of tgif") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(data.train, aes(x=donr, y= lgif)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of lgif") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(data.train, aes(x=donr, y= rgif)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of rgif") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(data.train, aes(x=donr, y= tdon)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of tdon") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(data.train, aes(x=donr, y= tlag)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of tlag") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(data.train, aes(x=donr, y= agif)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of agif") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#Correlation Matrix
subdatnumcor <- subset(data.train, select=c("chld","avhv","incm","inca","plow","npro","tgif","lgif","rgif","tdon","tlag","agif","damt"))

par(mfrow=c(1,1))  
corr <- round(cor(subdatnumcor),2)
ggcorrplot(corr, outline.col = "white", ggtheme = ggplot2::theme_gray, colors = c("#6D9EC1", "white", "#E46726"),lab = TRUE)
par(mfrow=c(1,1))  

#Scatterplot Matrix 
require(lattice)
pairs(subdatnumcor, pch = 21)

###### EDA for Qualitative Variables#####
library(ggplot2)
#donr
require(ggplot2)
ggplot(data.train) +
  geom_bar( aes(donr) ) +
  ggtitle("donr") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#reg1
require(ggplot2)
ggplot(data.train) +
  geom_bar( aes(reg1) ) +
  ggtitle("reg1") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#reg2
require(ggplot2)
ggplot(data.train) +
  geom_bar( aes(reg2) ) +
  ggtitle("reg2") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#reg3
require(ggplot2)
ggplot(data.train) +
  geom_bar( aes(reg3) ) +
  ggtitle("reg3") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#reg4
require(ggplot2)
ggplot(data.train) +
  geom_bar( aes(reg4) ) +
  ggtitle("reg4") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#home
require(ggplot2)
ggplot(data.train) +
  geom_bar( aes(home) ) +
  ggtitle("home") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#hinc
require(ggplot2)
ggplot(data.train) +
  geom_bar( aes(hinc) ) +
  ggtitle("hinc") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#genf
require(ggplot2)
ggplot(data.train) +
  geom_bar( aes(genf) ) +
  ggtitle("genf") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#wrat
require(ggplot2)
ggplot(data.train) +
  geom_bar( aes(wrat) ) +
  ggtitle("wrat") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

###Crosstabs
attach(data.train)
library(gmodels)

#reg1
CrossTable(donr, reg1, prop.r=FALSE, prop.c=TRUE, prop.t=TRUE, prop.chisq=FALSE)
l <- ggplot(data.train, aes(reg1,fill = donr))
l <- l + geom_histogram(stat="count")
tapply(as.numeric(data.train$donr) - 1 , data.train$ reg1,mean)

A <- ggplot(data.train, aes(x = reg1, fill = donr)) + geom_bar(position = 'fill')+
  theme_minimal() + coord_flip()
grid.arrange(l,A, nrow = 2)

#reg2
CrossTable(donr, reg2, prop.r=FALSE, prop.c=TRUE, prop.t=TRUE, prop.chisq=FALSE)
l <- ggplot(data.train, aes(reg2,fill = donr))
l <- l + geom_histogram(stat="count")
tapply(as.numeric(data.train$donr) - 1 , data.train$ reg2,mean)

A <- ggplot(data.train, aes(x = reg2, fill = donr)) + geom_bar(position = 'fill')+
  theme_minimal() + coord_flip()
grid.arrange(l,A, nrow = 2)

#reg3
CrossTable(donr, reg3, prop.r=FALSE, prop.c=TRUE, prop.t=TRUE, prop.chisq=FALSE)
l <- ggplot(data.train, aes(reg3,fill = donr))
l <- l + geom_histogram(stat="count")
tapply(as.numeric(data.train$donr) - 1 , data.train$ reg3,mean)

A <- ggplot(data.train, aes(x = reg3, fill = donr)) + geom_bar(position = 'fill')+
  theme_minimal() + coord_flip()
grid.arrange(l,A, nrow = 2)

#reg4
CrossTable(donr, reg4, prop.r=FALSE, prop.c=TRUE, prop.t=TRUE, prop.chisq=FALSE)
l <- ggplot(data.train, aes(reg4,fill = donr))
l <- l + geom_histogram(stat="count")
tapply(as.numeric(data.train$donr) - 1 , data.train$ reg4,mean)

A <- ggplot(data.train, aes(x = reg4, fill = donr)) + geom_bar(position = 'fill')+
  theme_minimal() + coord_flip()
grid.arrange(l,A, nrow = 2)

#hinc
CrossTable(donr, hinc, prop.r=FALSE, prop.c=TRUE, prop.t=TRUE, prop.chisq=FALSE)
l <- ggplot(data.train, aes(hinc,fill = donr))
l <- l + geom_histogram(stat="count")
tapply(as.numeric(data.train$donr) - 1 , data.train$ hinc,mean)

A <- ggplot(data.train, aes(x = hinc, fill = donr)) + geom_bar(position = 'fill')+
  theme_minimal() + coord_flip()
grid.arrange(l,A, nrow = 2)

#home
CrossTable(donr, home, prop.r=FALSE, prop.c=TRUE, prop.t=TRUE, prop.chisq=FALSE)
l <- ggplot(data.train, aes(home,fill = donr))
l <- l + geom_histogram(stat="count")
tapply(as.numeric(data.train$donr) - 1 , data.train$ home,mean)

A <- ggplot(data.train, aes(x = home, fill = donr)) + geom_bar(position = 'fill')+
  theme_minimal() + coord_flip()
grid.arrange(l,A, nrow = 2)

#genf
CrossTable(donr, genf, prop.r=FALSE, prop.c=TRUE, prop.t=TRUE, prop.chisq=FALSE)
l <- ggplot(data.train, aes(genf,fill = donr))
l <- l + geom_histogram(stat="count")
tapply(as.numeric(data.train$donr) - 1 , data.train$ genf,mean)

A <- ggplot(data.train, aes(x = genf, fill = donr)) + geom_bar(position = 'fill')+
  theme_minimal() + coord_flip()
grid.arrange(l,A, nrow = 2)

#wrat
CrossTable(donr, wrat, prop.r=FALSE, prop.c=TRUE, prop.t=TRUE, prop.chisq=FALSE)
l <- ggplot(data.train, aes(wrat,fill = donr))
l <- l + geom_histogram(stat="count")
tapply(as.numeric(data.train$ donr) - 1 , data.train$ wrat,mean)

A <- ggplot(data.train, aes(x = wrat, fill = donr)) + geom_bar(position = 'fill')+
  theme_minimal() + coord_flip()
grid.arrange(l,A, nrow = 2)

data.train$reg1 <- as.integer(data.train$reg1)
data.train$reg2 <- as.integer(data.train$reg2)
data.train$reg3 <- as.integer(data.train$reg3)
data.train$reg4 <- as.integer(data.train$reg4)
data.train$home <- as.integer(data.train$home)
data.train$hinc <- as.integer(data.train$hinc)
data.train$genf <- as.integer(data.train$genf)
data.train$wrat <- as.integer(data.train$wrat)
data.train$donr <- as.integer(data.train$donr)

############################## EDA for Prediction Models - Training Data ##############################

prediction_damt <-filter(data.train, damt >0)

#Factor Variables
prediction_damt$reg1 <- as.factor(prediction_damt$reg1)
prediction_damt$reg2 <- as.factor(prediction_damt$reg2)
prediction_damt$reg3 <- as.factor(prediction_damt$reg3)
prediction_damt$reg4 <- as.factor(prediction_damt$reg4)
prediction_damt$home <- as.factor(prediction_damt$home)
prediction_damt$hinc <- as.factor(prediction_damt$hinc)
prediction_damt$genf <- as.factor(prediction_damt$genf)
prediction_damt$wrat <- as.factor(prediction_damt$wrat)
prediction_damt$donr <- as.factor(prediction_damt$donr)

#Descriptive Statistics
str(prediction_damt)
summary(prediction_damt)
describe(prediction_damt)
dim(prediction_damt)

#####EDA for Numeric Variables#####

par(mfrow=c(2,2))
hist(prediction_damt $chld, col = "#09ADAD", xlab = "chld", main = "Histogram of chld")
hist(prediction_damt $avhv, col = "#DBCEAC", xlab = "avhv", main = "Histogram of avhv")

boxplot(prediction_damt $chld, col = "#09ADAD", main = "Boxplot of chld")
boxplot(prediction_damt $avhv, col = "#DBCEAC", main = "Boxplot of avhv")

par(mfrow=c(1,1))

par(mfrow=c(3,3))
hist(prediction_damt $incm, col = "#A71930", xlab = "incm", main = "Histogram of incm")
hist(prediction_damt $inca, col = "#09ADAD", xlab = "inca", main = "Histogram of inca")
hist(prediction_damt $plow, col = "#DBCEAC", xlab = "plow", main = "Histogram of plow")
boxplot(prediction_damt $incm, col = "#A71930", main = "Boxplot of incm")
boxplot(prediction_damt $inca, col = "#09ADAD", main = "Boxplot of inca")
boxplot(prediction_damt $plow, col = "#DBCEAC", main = "Boxplot of plow")
par(mfrow=c(1,1))

par(mfrow=c(3,3))
hist(prediction_damt $npro, col = "#A71930", xlab = "npro", main = "Histogram of npro")
hist(prediction_damt $tgif, col = "#09ADAD", xlab = "tgif", main = "Histogram of tgif")
hist(prediction_damt $lgif, col = "#DBCEAC", xlab = "lgif", main = "Histogram of lgif")

boxplot(prediction_damt $npro, col = "#A71930", main = "Boxplot of npro")
boxplot(prediction_damt $tgif, col = "#09ADAD", main = "Boxplot of tgif")
boxplot(prediction_damt $lgif, col = "#DBCEAC", main = "Boxplot of lgif")
par(mfrow=c(1,1))

par(mfrow=c(2,2))
hist(prediction_damt $rgif, col = "#A71930", xlab = "rgif", main = "Histogram of rgif")
hist(prediction_damt $tdon, col = "#09ADAD", xlab = " tdon", main = "Histogram of tdon")
boxplot(prediction_damt $rgif, col = "#A71930", main = "Boxplot of rgif")
boxplot(prediction_damt $tdon, col = "#09ADAD", main = "Boxplot of tdon")
par(mfrow=c(1,1))

par(mfrow=c(2,2))
hist(prediction_damt $tlag, col = "#DBCEAC", xlab = "tlag ", main = "Histogram of tlag")
hist(prediction_damt $agif, col = "#DBCEAC", xlab = "agif", main = "Histogram of agif")
boxplot(prediction_damt $tlag, col = "#DBCEAC", main = "Boxplot of tlag")
boxplot(prediction_damt $agif, col = "#DBCEAC", main = "Boxplot of agif")
par(mfrow=c(1,1))

#Correlation Matrix
subdatnumcor <- subset(prediction_damt , select=c("chld","avhv","incm","inca","plow","npro","tgif","lgif","rgif","tdon","tlag","agif","damt"))

par(mfrow=c(1,1))  
corr <- round(cor(subdatnumcor),2)
ggcorrplot(corr, outline.col = "white", ggtheme = ggplot2::theme_gray, colors = c("#6D9EC1", "white", "#E46726"),lab = TRUE)
par(mfrow=c(1,1))  

#Scatterplot Matrix 
require(lattice)
pairs(subdatnumcor, pch = 21)

#####EDA for Qualitative Variables#####

library(ggplot2)
#donr
require(ggplot2)
ggplot(prediction_damt ) +
  geom_bar( aes(donr) ) +
  ggtitle("donr") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#reg1
require(ggplot2)
ggplot(prediction_damt ) +
  geom_bar( aes(reg1) ) +
  ggtitle("reg1") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#reg2
require(ggplot2)
ggplot(prediction_damt ) +
  geom_bar( aes(reg2) ) +
  ggtitle("reg2") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#reg3
require(ggplot2)
ggplot(prediction_damt ) +
  geom_bar( aes(reg3) ) +
  ggtitle("reg3") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#reg4
require(ggplot2)
ggplot(prediction_damt ) +
  geom_bar( aes(reg4) ) +
  ggtitle("reg4") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#home
require(ggplot2)
ggplot(prediction_damt ) +
  geom_bar( aes(home) ) +
  ggtitle("home") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#hinc
require(ggplot2)
ggplot(prediction_damt ) +
  geom_bar( aes(hinc) ) +
  ggtitle("hinc") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#genf
require(ggplot2)
ggplot(prediction_damt ) +
  geom_bar( aes(genf) ) +
  ggtitle("genf") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#wrat
require(ggplot2)
ggplot(prediction_damt ) +
  geom_bar( aes(wrat) ) +
  ggtitle("wrat") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#Boxplots for Qualitative Variables
ggplot(prediction_damt , aes(x=reg1, y= damt)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of reg1") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(prediction_damt , aes(x= reg2, y= damt)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of reg2") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(prediction_damt , aes(x= reg3, y= damt)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of reg3") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(prediction_damt , aes(x= reg4, y= damt)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of reg4") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(prediction_damt , aes(x=home, y= damt)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of home ") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(prediction_damt , aes(x=hinc, y= damt)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of hinc ") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(prediction_damt , aes(x=genf, y= damt)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of genf ") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(prediction_damt , aes(x=wrat, y= damt)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of wrat ") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

########################################### CLASSIFICATION MODELING ############################################

##### Logistic Regression #####

###Logistic Regression Model 1 (Baseline)###
model.log1<- glm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                   avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                 data.train.std.c, family=binomial("logit"))

summary(model.log1)
Anova(model.log1, type="II", test="Wald")
varImp(model.log1)
nagelkerke(model.log1) #Baseline McFadden: 0.609345

#Performance Metrics
AIC(model.log1) #Baseline 2201.584

BIC(model.log1) #Baseline 2339.965

post.valid.log1 <- predict(model.log1, data.valid.std.c, type="response") # n.valid post probs

# Calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.log1 <- cumsum(14.5*c.valid[order(post.valid.log1, decreasing=T)]-2)
plot(profit.log1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.log1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.log1)) # report number of mailings and maximum profit
# Baseline: 1291.0 11642.5

cutoff.log1 <- sort(post.valid.log1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.log1 <- ifelse(post.valid.log1>cutoff.log1, 1, 0) # mail to everyone above the cutoff

#Performance
table(chat.valid.log1, c.valid) # classification table
mean(chat.valid.log1== c.valid) #Accuracy #Baseline: 0.8375  
xtab.log1=table(chat.valid.log1, c.valid)
confusionMatrix(xtab.log1, positive = "1")

#               c.valid
#chat.valid.log1   0   1
#              0 709  18
#              1 310 981
# check n.mail.valid = 310+981 = 1291
# check profit = 14.5*981-2*1291 = 11642.5

#ROC Curve
detach(package:ROCR)
library(ROCR)

prob <- predict(model.log1, newdata=data.valid.std.c, type="response")
pred <- prediction(prob, data.valid.std.c$donr)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf,colorize=TRUE)

#AUC
perf_auc <- performance(pred, measure = "auc")
perf_auc <- perf_auc@y.values[[1]]
perf_auc #Baseline 0.9422

###Logistic Regression Model 2### 

#Full Model for Variable Selection & Baseline
model.logfull <- glm(donr ~ ., data.train.std.c, family=binomial("logit"))
varImp(model.logfull)

#Stepwise Regression for Variable Selection
model.lower = glm(donr ~ 1, data.train.std.c, family = binomial(link="logit"))
model.logfull <- glm(donr ~ ., data.train.std.c, family=binomial("logit"))
step(model.lower, scope = list(upper=model.logfull), direction="both", test="Chisq", data=data.train.std.c)

model.logstep <- glm(donr ~ chld + reg2 + home + wrat + reg1 + incm + npro + tlag + tdon + plow + tgif + hinc, 
                     data.train.std.c, family=binomial("logit"))

model.log2 <- glm(donr ~ reg1 + reg2 + home + chld + I(hinc^2) + wrat + 
                    incm  + plow + npro + tdon + tlag, 
                  data.train.std.c, family=binomial("logit"))
summary(model.log2)
Anova(model.log2, type="II", test="Wald")
varImp(model.log2)
nagelkerke(model.log2) #Baseline McFadden: 0.609345

#Performance Metrics
AIC(model.log2) #2191.221
BIC(model.log2) #2266.702

post.valid.log1 <- predict(model.log2, data.valid.std.c, type="response") # n.valid post probs

# Calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.log1 <- cumsum(14.5*c.valid[order(post.valid.log1, decreasing=T)]-2)
plot(profit.log1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.log1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.log1)) # report number of mailings and maximum profit
# Baseline: 1271.0 11653.5

cutoff.log1 <- sort(post.valid.log1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.log1 <- ifelse(post.valid.log1>cutoff.log1, 1, 0) # mail to everyone above the cutoff

#Performance
table(chat.valid.log1, c.valid) # classification table
mean(chat.valid.log1== c.valid) #Accuracy 0.8454
xtab.log1=table(chat.valid.log1, c.valid)
confusionMatrix(xtab.log1, positive = "1")

#ROC Curve
detach(package:ROCR)
library(ROCR)

prob <- predict(model.log2, newdata=data.valid.std.c, type="response")
pred <- prediction(prob, data.valid.std.c$donr)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf,colorize=TRUE)

#AUC
perf_auc <- performance(pred, measure = "auc")
perf_auc <- perf_auc@y.values[[1]]
perf_auc #0.9425972

##### Logistic Regression GAM #####

#Logistic Regression Baseline for ANOVA
model.gam1a <- gam(donr ~ reg1 + reg2 + home + chld + I(hinc^2) + wrat + 
                     incm  + plow + npro + tdon + tlag, 
                   data.train.std.c, family=binomial("logit"))

#Note: reg1, reg2, and home failed..."Error in gam.match(data) : 
#A smoothing variable encountered with 3 or less unique values; at least 4 needed

#Test Chld variable for Splines using Anova
model.gam1b <-gam(donr~reg1 + reg2 + home + s(chld,4) + I(hinc^2) + wrat + 
                    incm  + plow + npro + tdon + tlag, data=data.train.std.c, family=binomial)

anova(model.gam1a,model.gam1b,test="Chisq") #Shows significant so keep it

#Test wrat variable for Splines using Anova
model.gam1c <-gam(donr~reg1 + reg2 + home + chld + I(hinc^2) + s(wrat,4) + 
                    incm  + plow + npro + tdon + tlag, data=data.train.std.c, family=binomial)

anova(model.gam1a,model.gam1c,test="Chisq") #Shows significant so keep it

#Test incm variable for Splines using Anova
model.gam1d <-gam(donr~reg1 + reg2 + home + chld + I(hinc^2) + wrat + 
                    s(incm,4)  + plow + npro + tdon + tlag, data=data.train.std.c, family=binomial)

anova(model.gam1a,model.gam1d,test="Chisq") #Not significant so do not apply spline

#Test plow variable for Splines using Anova
model.gam1e <-gam(donr~reg1 + reg2 + home + chld + I(hinc^2) + wrat + 
                    incm  + s(plow,4) + npro + tdon + tlag, data=data.train.std.c, family=binomial)

anova(model.gam1a,model.gam1e,test="Chisq") #Not significant so do not apply spline

#Test npro variable for Splines using Anova
model.gam1f <-gam(donr~reg1 + reg2 + home + chld + I(hinc^2) + wrat + 
                    incm  + plow + s(npro,4) + tdon + tlag, data=data.train.std.c, family=binomial)

anova(model.gam1a,model.gam1f,test="Chisq") #Not significant so do not apply spline

#Test tdon variable for Splines using Anova
model.gam1g <-gam(donr~reg1 + reg2 + home + chld + I(hinc^2) + wrat + 
                    incm  + plow +npro + s(tdon,4) + tlag, data=data.train.std.c, family=binomial)

anova(model.gam1a,model.gam1g,test="Chisq") #Shows significant so keep it

#Test tlag variable for Splines using Anova
model.gam1h <-gam(donr~reg1 + reg2 + home + chld + I(hinc^2) + wrat + 
                    incm  + plow +npro + tdon + s(tlag,4), data=data.train.std.c, family=binomial)

anova(model.gam1a,model.gam1h,test="Chisq") #Not significant so do not apply spline

###Logistic Regression GAM Model###

model.gam1 <-gam(donr~reg1 + reg2 + home + s(chld,4) + I(hinc^2) + s(wrat,9) + 
                   incm  + plow + npro + s(tdon,4) + tlag, data=data.train.std.c, family=binomial)

plot(model.gam1,se=T)
summary(model.gam1)
Anova(model.gam1, type="II", test="Wald")

#Performance Metrics
AIC(model.gam1) #1791.098

BIC(model.gam1) #1866.579

post.valid.log1 <- predict(model.gam1, data.valid.std.c, type="response") # n.valid post probs

# Calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.log1 <- cumsum(14.5*c.valid[order(post.valid.log1, decreasing=T)]-2)
plot(profit.log1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.log1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.log1)) # report number of mailings and maximum profit
#1234 11829

cutoff.log1 <- sort(post.valid.log1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.log1 <- ifelse(post.valid.log1>cutoff.log1, 1, 0) # mail to everyone above the cutoff

#Performance
table(chat.valid.log1, c.valid) # classification table
mean(chat.valid.log1== c.valid) #Accuracy 0.8707    
xtab.log1=table(chat.valid.log1, c.valid)
confusionMatrix(xtab.log1, positive = "1")

#ROC Curve
detach(package:ROCR)
library(ROCR)

prob <- predict(model.gam1, newdata=data.valid.std.c, type="response")
pred <- prediction(prob, data.valid.std.c$donr)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf,colorize=TRUE)

#AUC
perf_auc <- performance(pred, measure = "auc")
perf_auc <- perf_auc@y.values[[1]]
perf_auc #0.9643883

##### Linear Discriminant Analysis #####
# Include additional terms on the fly using I()
# Note: strictly speaking, LDA should not be used with qualitative predictors,
# but in practice it often is if the goal is simply to find a good predictive model

###Linear Discriminant Analysis (Baseline)###
model.lda1 <- lda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                    avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                  data.train.std.c) 

# Generate n.valid.c posterior probabilities
post.valid.lda1 <- predict(model.lda1, data.valid.std.c)$posterior[,2] 

# Calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.lda1 <- cumsum(14.5*c.valid[order(post.valid.lda1, decreasing=T)]-2)
plot(profit.lda1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.lda1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.lda1)) # report number of mailings and maximum profit; # 1329.0 11624.5

cutoff.lda1 <- sort(post.valid.lda1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.lda1 <- ifelse(post.valid.lda1>cutoff.lda1, 1, 0) # mail to everyone above the cutoff

#Performance
table(chat.valid.lda1, c.valid) # classification table
mean(chat.valid.lda1== c.valid) #Baseline Accuracy 0.8226    
xtab.lda1=table(chat.valid.lda1, c.valid)
confusionMatrix(xtab.lda1, positive = "1")

#               c.valid
#chat.valid.lda1   0   1
#              0 675  14
#              1 344 985
# check n.mail.valid = 344+985 = 1329
# check profit = 14.5*985-2*1329 = 11624.5

#ROC Curve
detach(package:ROCR)
library(ROCR)

test <-  predict(model.lda1, data.valid.std.c)$posterior
pred <- prediction(test[,2], data.valid.std.c$donr)
perf <- performance(pred, "tpr", "fpr") 

#AUC
perf_auc <- performance(pred, measure = "auc")
perf_auc <- perf_auc@y.values[[1]]
perf_auc #0.9413575

plot(perf,colorize=TRUE)

###Linear Discriminant Analysis Model 2###
model.lda2 <- lda(donr ~ reg1 + reg2 + home + chld + I(hinc^2) + wrat + 
                    incm  + plow + npro  + tdon + tlag, 
                  data.train.std.c) 

# Generate n.valid.c posterior probabilities
post.valid.lda1 <- predict(model.lda2, data.valid.std.c)$posterior[,2] 

# Calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.lda1 <- cumsum(14.5*c.valid[order(post.valid.lda1, decreasing=T)]-2)
plot(profit.lda1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.lda1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.lda1)) # report number of mailings and maximum profit
#1334.0 11643.5

cutoff.lda1 <- sort(post.valid.lda1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.lda1 <- ifelse(post.valid.lda1>cutoff.lda1, 1, 0) # mail to everyone above the cutoff

#Performance
table(chat.valid.lda1, c.valid) # classification table
mean(chat.valid.lda1== c.valid) #Accuracy 0.8221  
xtab.lda1=table(chat.valid.lda1, c.valid)
confusionMatrix(xtab.lda1, positive = "1")

#ROC Curve
detach(package:ROCR)
library(ROCR)

test <-  predict(model.lda2, data.valid.std.c)$posterior
pred <- prediction(test[,2], data.valid.std.c$donr)
perf <- performance(pred, "tpr", "fpr")

#AUC
perf_auc <- performance(pred, measure = "auc")
perf_auc <- perf_auc@y.values[[1]]
perf_auc #0.9417543

plot(perf,colorize=TRUE)

##### Quadratic Discriminant Analysis #####

model.qda1 <- qda(donr ~ reg1 + reg2 + home + chld + I(hinc^2) + wrat + 
                    incm  + plow + npro  + tdon + tlag, 
                  data.train.std.c) 

# Generate n.valid.c posterior probabilities
post.valid.qda1 <- predict(model.qda1, data.valid.std.c)$posterior[,2] 

# Calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.qda1 <- cumsum(14.5*c.valid[order(post.valid.qda1, decreasing=T)]-2)
plot(profit.qda1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.qda1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.qda1)) # report number of mailings and maximum profit
#1439 11274

cutoff.qda1 <- sort(post.valid.qda1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.qda1 <- ifelse(post.valid.qda1>cutoff.qda1, 1, 0) # mail to everyone above the cutoff

#Performance
table(chat.valid.qda1, c.valid) # classification table
mean(chat.valid.qda1== c.valid) #Accuracy 0.7592  
xtab.qda1=table(chat.valid.qda1, c.valid)
confusionMatrix(xtab.qda1, positive = "1")

#ROC Curve
detach(package:ROCR)
library(ROCR)

test <-  predict(model.qda1, data.valid.std.c)$posterior
pred <- prediction(test[,2], data.valid.std.c$donr)
perf <- performance(pred, "tpr", "fpr")

#AUC
perf_auc <- performance(pred, measure = "auc")
perf_auc <- perf_auc@y.values[[1]]
perf_auc #0.9213541

plot(perf,colorize=TRUE)

##### K-Nearest Neighbors #####

#Find K using CV Methods
data.train.std.c.knn= data.train.std.c
data.train.std.c.knn$donr = as.factor(data.train.std.c.knn$donr)

#Find K using CV Methods manual grid search
set.seed(7)
library(caret)
control <- trainControl(method="cv", number=5)
grid <- expand.grid(k=c(1:25))
model <- train(donr~., data=data.train.std.c.knn, method="knn", trControl=control, tuneGrid=grid,preProcess = c("center","scale"))
plot(model)

#Find K using CV Methods automatic grid search
set.seed(400)
ctrl <- trainControl(method="repeatedcv",repeats = 5) #,classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(donr ~ ., data = data.train.std.c.knn, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)
knnFit
plot(knnFit)

#Remove donr from train/validation sets since they should not be used
knn.train <- data.train.std.c[-21]
knn.valid <- data.valid.std.c[-21]

## Fit KNN model with K=9 ##
set.seed(1)
model.knn1=knn(knn.train,knn.valid,c.train,k=9)

## Fit KNN model with K=9 ##
set.seed(1)
model.knn1=knn(knn.train,knn.valid,c.train,k=9)

#Performance
table(model.knn1,c.valid) # classification table
mean(model.knn1== c.valid) #Accuracy 0.814668

xtab.model.knn1=table(model.knn1, c.valid)
confusionMatrix(xtab.model.knn1, positive = "1")

# Profit Calculation
14.5*931-2*1237   # 11025.5

#Performance
table(model.knn1,c.valid) # classification table
mean(model.knn1== c.valid) #Accuracy 0.814668

xtab.model.knn1=table(model.knn1, c.valid)
confusionMatrix(xtab.model.knn1, positive = "1")

# Profit Calculation
14.5*931-2*1237   # 11025.5

##### Neural Network #####
library(caret)
#Use CV to find ideal parameters
data.train.std.c.nnet= data.train.std.c
data.train.std.c.nnet$donr = as.factor(data.train.std.c.nnet$donr)

fitControl <- trainControl(method="cv", number=5)

set.seed(1)
gbmFit <- train(donr ~ reg1 + reg2 + home + chld + I(hinc^2) + wrat + 
                  incm  + plow + npro + tdon + tlag, data.train.std.c.nnet,
                method = "nnet", trControl = fitControl, verbose = FALSE)
gbmFit
plot(gbmFit)
#The final values used for the model were size = 5 and decay = 0.1.

set.seed(800)
model.nnet1 <- avNNet(donr ~ reg1 + reg2 + home + chld + I(hinc^2) + wrat + 
                      incm  + plow + npro + tdon + tlag, data.train.std.c, size = 5, decay=.10, maxit=2000) 

model.nnet1 

post.valid.nnet1 <- predict(model.nnet1, data.valid.std.c)

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.nnet1 <- cumsum(14.5*c.valid[order(post.valid.nnet1, decreasing = T)] - 2)
plot(profit.nnet1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.nnet1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.nnet1)) # report number of mailings and maximum profit
# 1242 11871

cutoff.nnet1 <- sort(post.valid.nnet1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.nnet1 <- ifelse(post.valid.nnet1 > cutoff.nnet1, 1, 0) # mail to everyone above the cutoff

#Performance
table(chat.valid.nnet1, c.valid) # classification table
mean(chat.valid.nnet1== c.valid) #Accuracy 0.870664   
xtab.nnet1=table(chat.valid.nnet1, c.valid)
confusionMatrix(xtab.nnet1, positive = "1")

#ROC Curve
detach(package:ROCR)
library(ROCR)

prob <- predict(model.nnet1, newdata=data.valid.std.c, type="raw")
pred <- prediction(prob, data.valid.std.c$donr)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf,colorize=TRUE)

#AUC
perf_auc <- performance(pred, measure = "auc")
perf_auc <- perf_auc@y.values[[1]]
perf_auc #0.9637606

##### Decision Tree #####

# Create datasets with donr coded as factor variable for decision tree
data.train.std.c.tree= data.train.std.c
data.train.std.c.tree$donr = as.factor(data.train.std.c.tree$donr)

data.valid.std.c.tree= data.valid.std.c
data.valid.std.c.tree$donr = as.factor(data.valid.std.c.tree$donr)

#Fit Full Decision Tree
model.treefull<- tree(donr ~ ., data.train.std.c.tree)
summary(model.treefull)
model.treefull

#Plot Full Tree
plot(model.treefull)
text(model.treefull,pretty=0)

# Evaluate Full Tree on Validation Set
post.valid.treefull <- predict(model.treefull, data.valid.std.c.tree, type="class") # n.valid post probs

# Calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.treefull <- cumsum(14.5*c.valid[order(post.valid.treefull, decreasing=T)]-2)
plot(profit.treefull) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.treefull) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.treefull)) # report number of mailings and maximum profit
# 1168 11149

##Performance for Full Tree on Validation Set
table(post.valid.treefull, c.valid) # classification table
mean(post.valid.treefull== c.valid) #Accuracy 0.8484
treeError <- mean(post.valid.treefull!= c.valid) #Error 0.1516
treeError
xtab.treefull=table(post.valid.treefull, c.valid)
confusionMatrix(xtab.treefull, positive = "1")

### Use Cross-Validation to Prune Tree ###
set.seed(3)
cv.treeprune=cv.tree(model.treefull,FUN=prune.misclass)
names(cv.treeprune)
cv.treeprune

#We plot the error rate as a function of both size and k.

par(mfrow=c(1,2))
plot(cv.treeprune$size, cv.treeprune$dev,type="b")
plot(cv.treeprune$k, cv.treeprune$dev,type="b")

#Apply Pruned Tree
model.treeprune=prune.misclass(model.treefull,best=15)
plot(model.treeprune)
text(model.treeprune,pretty=0)

##Evaluate Pruned Tree on Validation Set
post.valid.treeprune <- predict(model.treeprune, data.valid.std.c.tree, type="class") # n.valid post probs

# Calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.treeprune <- cumsum(14.5*c.valid[order(post.valid.treeprune, decreasing=T)]-2)
plot(profit.treeprune) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.treeprune) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.treeprune)) # report number of mailings and maximum profit
# 1168 11149

#Performance for Pruned Tree on Validation Set
table(post.valid.treeprune, c.valid) # classification table
mean(post.valid.treeprune== c.valid) #Accuracy 0.8484
treeError <- mean(post.valid.treeprune!= c.valid) #Error 0.1516
treeError
xtab.treeprune=table(post.valid.treeprune, c.valid)
confusionMatrix(xtab.treeprune, positive = "1")

#AUC
library(pROC)
prob <- predict(model.treeprune, newdata=data.valid.std.c.tree, type='vector')
auc<-auc(data.valid.std.c$donr,prob[,2])
auc #0.9092581

##### Bagging #####

# Create datasets with donr coded as factor variable for decision tree
data.train.std.c.tree= data.train.std.c
data.train.std.c.tree$donr = as.factor(data.train.std.c.tree$donr)

data.valid.std.c.tree= data.valid.std.c
data.valid.std.c.tree$donr = as.factor(data.valid.std.c.tree$donr)

library(randomForest)
set.seed(1)
bag.donr=randomForest(donr~.,data= data.train.std.c.tree,mtry=20,importance=TRUE, ntree=100)
bag.donr

# Evaluate Full Tree on Validation Set
post.valid.bag<- predict(bag.donr, data.valid.std.c.tree, type="class") # n.valid post probs

# Calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.bag <- cumsum(14.5*c.valid[order(post.valid.bag, decreasing=T)]-2)
plot(profit.bag) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.bag) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.bag)) # report number of mailings and maximum profit
# 1031 10959

##Performance on Validation Set
table(post.valid.bag, c.valid) # classification table
mean(post.valid.bag== c.valid) #Accuracy 0.8840
treeError <- mean(post.valid.bag!= c.valid) #Error 0.1160
treeError
xtab.bag=table(post.valid.bag, c.valid)
confusionMatrix(xtab.bag, positive = "1")

#AUC
library(pROC)
prob <- predict(bag.donr, newdata=data.valid.std.c.tree, type='prob')
auc<-auc(data.valid.std.c$donr,prob[,2])
auc #0.9500167

##### Random Forest ##### 

# Create datasets with donr coded as factor variable for decision tree
data.train.std.c.tree= data.train.std.c
data.train.std.c.tree$donr = as.factor(data.train.std.c.tree$donr)

data.valid.std.c.tree= data.valid.std.c
data.valid.std.c.tree$donr = as.factor(data.valid.std.c.tree$donr)

library(randomForest)
set.seed(1)
rf.donr=randomForest(donr~.,data= data.train.std.c.tree,importance=TRUE,ntree=400)
rf.donr

importance(rf.donr)
varImpPlot(rf.donr)

# Evaluate Full Tree on Validation Set
post.valid.rf<- predict(rf.donr, data.valid.std.c.tree, type="class") # n.valid post probs

# Calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.rf <- cumsum(14.5*c.valid[order(post.valid.rf, decreasing=T)]-2)
plot(profit.rf) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.rf) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.rf)) # report number of mailings and maximum profit
# 1066 11266

##Performance on Validation Set
table(post.valid.rf, c.valid) # classification table
mean(post.valid.rf== c.valid) #Accuracy 0.8934589  
treeError <- mean(post.valid.rf!= c.valid) #Error 0.1065411
treeError
xtab.rf=table(post.valid.rf, c.valid)
confusionMatrix(xtab.rf, positive = "1")

#AUC
library(pROC)
prob <- predict(rf.donr, newdata=data.valid.std.c.tree, type='prob')
auc<-auc(data.valid.std.c$donr,prob[,2])
auc #0.9598652

##### Boosting ##### 

library(gbm)

#Find ideal parameter using CV
#Use CV to find ideal parameters
data.train.std.c.boost= data.train.std.c
data.train.std.c.boost$donr = as.factor(data.train.std.c.boost$donr)

#Find ideal parameter using CV using manual grid search
#set.seed(7)
#control <- trainControl(method="cv", number=5)
#grid <- expand.grid(interaction.depth=seq(1,7,by=2),n.trees=seq(100,1500,by=50),shrinkage=c(0.01,0.1),n.minobsinnode = 10)
#model <- train(donr~., data.train.std.c.boost, method="gbm", trControl=control, tuneGrid=grid)
#print(model)
#The final values used for the model were n.trees = 500, interaction.depth = 1, shrinkage = 0.1 and n.minobsinnode = 10.

#Find ideal parameter using CV using automatic grid search
library(caret)
fitControl <- trainControl(method="repeatedcv", number=5, repeats=5)
set.seed(1)
gbmFit <- train(donr ~ ., data = data.train.std.c.boost,
                method = "gbm", trControl = fitControl, verbose = FALSE, tuneLength=5)
gbmFit

#The final values used for the model were n.trees = 200, interaction.depth = 2, shrinkage = 0.1 and n.minobsinnode = 10.

#Fit Boost
set.seed(1)
boost.donr=gbm(donr~.,data = data.train.std.c,distribution="bernoulli",n.trees=200,shrinkage = 0.1,interaction.depth=2)

# Evaluate Full Tree on Validation Set
post.valid.boost<- predict(boost.donr, data.valid.std.c, type="response", n.trees = 200) # n.valid post probs

# Calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.boost <- cumsum(14.5*c.valid[order(post.valid.boost, decreasing=T)]-2)
plot(profit.boost) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.boost) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.boost)) # report number of mailings and maximum profit
# 1236.0 11955.5

cutoff.boost <- sort(post.valid.boost, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.boost <- ifelse(post.valid.boost > cutoff.boost, 1, 0) # mail to everyone above the cutoff

#Performance
table(chat.valid.boost, c.valid) # classification table
mean(chat.valid.boost== c.valid) #Accuracy 0.8786   
boostError <- mean(chat.valid.boost!= c.valid) #Error 0.1214
boostError
xtab.boost=table(chat.valid.boost, c.valid)
confusionMatrix(xtab.boost, positive = "1")

#AUC
library(ROCR)
library(cvAUC)
labels <- data.valid.std.c[,"donr"]
AUC(predictions = post.valid.boost, labels = labels) #0.9715712

##### Support Vector Classifier ##### 

# Create datasets with donr coded as factor variable for svm
data.train.std.c.svm= data.train.std.c
data.train.std.c.svm$donr = as.factor(data.train.std.c.svm$donr)

data.valid.std.c.svm= data.valid.std.c
data.valid.std.c.svm$donr = as.factor(data.valid.std.c.svm$donr)

#Use cross-validation to choose cost (Book Method)
#set.seed(800)
#tune.out=tune(svm,donr~.,data=data.train.std.c.svm,kernel="linear",ranges=list(cost=c(0.001, 0.01, 0.1,0.15, 1)))
#summary(tune.out)  #Best is 0.15
#bestmod=tune.out$best.model
#summary(bestmod)

#bestmod=svm(donr~.,data=data.train.std.c.svm,kernel="linear",cost=0.10)
#summary(bestmod)

#Grid Search (Caret Package Method)
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid <- expand.grid(C = c(0.001, 0.01, 0.1,0.15, 1))
set.seed(800)
svm_Linear_Grid <- train(donr ~., data = data.train.std.c.svm, method = "svmLinear",
                         trControl=trctrl,
                         preProcess = c("center", "scale"),
                         tuneGrid = grid,
                         tuneLength = 10)
svm_Linear_Grid
plot(svm_Linear_Grid) #C = 0.01

test_pred_grid <- predict(svm_Linear_Grid, newdata = data.valid.std.c.svm)
test_pred_grid

confusionMatrix(test_pred_grid, data.valid.std.c.svm$donr, positive = "1")

# Evaluate Full SVM on Validation Set
post.valid.svm<- predict(svm_Linear_Grid, data.valid.std.c.svm) # n.valid post probs

# Calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.svm <- cumsum(14.5*c.valid[order(post.valid.svm, decreasing=T)]-2)
plot(profit.svm) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.svm) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.svm)) # report number of mailings and maximum profit
# 1822.0 10522.5

##Performance on Validation Set
table(post.valid.svm, c.valid) # classification table
mean(post.valid.svm== c.valid) #Accuracy  0.8374628
svmError <- mean(post.valid.svm!= c.valid) #Error 0.1625372
svmError
xtab.svm=table(post.valid.svm, c.valid)
confusionMatrix(xtab.svm, positive = "1")

##### Support Vector Machine with Radial Kernel ##### 

#Use cross-validation to choose cost (Book Method)

#set.seed(800)
#tune.out=tune(svm, donr~., data=data.train.std.c.svm, kernel="radial", ranges=list(cost=c(0.01,0.1,1, 5, 10),gamma=c(0.005, 0.01, 0.025, 0.05)))
#summary(tune.out)
#bestmod=tune.out$best.model #Cost 5, gamma 0.025
#summary(bestmod)
#bestmod=svm(donr~., data=data.train.std.c.svm, kernel="radial",gamma=0.025, cost=5, probability=TRUE)
#summary(bestmod)

#Grid Search (Caret Package Method)
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid_radial <- expand.grid(sigma = c(0.005, 0.01, 0.025, 0.05),
                           C = c(0.01,0.1,1, 5, 10))
set.seed(800)

svm_Radial_Grid <- train(donr ~., data = data.train.std.c.svm, method = "svmRadial",
                         trControl=trctrl,
                         preProcess = c("center", "scale"),
                         tuneGrid = grid_radial,
                         tuneLength = 10)

svm_Radial_Grid
plot(svm_Radial_Grid) #Cost 10, gamma 0.025

test_pred_Radial_Grid <- predict(svm_Radial_Grid, newdata = data.valid.std.c.svm)
confusionMatrix(test_pred_Radial_Grid, data.valid.std.c.svm $donr,positive = "1")

# Evaluate Full Svm on Validation Set
post.valid.svm<- predict(svm_Radial_Grid, data.valid.std.c.svm) # n.valid post probs

# Calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.svm <- cumsum(14.5*c.valid[order(post.valid.svm, decreasing=T)]-2)
plot(profit.svm) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.svm) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.svm)) # report number of mailings and maximum profit
# 1057 11023

##Performance on Validation Set
table(post.valid.svm, c.valid) # classification table
mean(post.valid.svm== c.valid) #Accuracy 0.8791  
svmError <- mean(post.valid.svm!= c.valid) #Error 0.1209118
svmError
xtab.svm=table(post.valid.svm, c.valid)
confusionMatrix(xtab.svm, positive = "1")

##### Results Model Comparison (Example) #####

# n.mail Profit  Model
# 1291   11642.5 Log1
# 1329   11624.5 LDA1
# 1439   11274.0 QDA1
#...see report for full details

###### Final Model Selection #####

#Select boost.donr since it has maximum profit in the validation sample

post.test <- predict(boost.donr, data.test.std, type="response", n.trees = 200) # post probs for test data

###### Oversampling adjustment for calculating number of mailings for test set ######

n.mail.valid <- which.max(profit.boost)
tr.rate <- .1 # typical response rate is .1
vr.rate <- .5 # whereas validation response rate is .5
adj.test.1 <- (n.mail.valid/n.valid.c)/(vr.rate/tr.rate) # adjustment for mail yes
adj.test.0 <- ((n.valid.c-n.mail.valid)/n.valid.c)/((1-vr.rate)/(1-tr.rate)) # adjustment for mail no
adj.test <- adj.test.1/(adj.test.1+adj.test.0) # scale into a proportion
adj.test #Optimal test mailing rate is 0.1494
n.mail.test <- round(n.test*adj.test, 0) # calculate number of mailings for test set; #300

cutoff.test <- sort(post.test, decreasing=T)[n.mail.test+1] # set cutoff based on n.mail.test
chat.test <- ifelse(post.test>cutoff.test, 1, 0) # mail to everyone above the cutoff
table(chat.test)

#chat.test
#0    1 
#1707  300 

# Based on this model we'll mail to the 300 highest posterior probabilities

# See below for saving chat.test into a file for submission

###################################### PREDICTION MODELING ######################################

###### Least Squares Regression ######

###Least Squares Regression (Baseline)###

#Full Model for Variable Selection & Baseline

model.lsfull <- lm(damt ~ ., data.train.std.y)
anova(model.lsfull)
summary(model.lsfull)
varImp(model.lsfull)

#Stepwise Regression for Variable Selection
stepwisemodel <- lm(formula = damt ~ ., data.train.std.y)
model.stepwise <- stepAIC(stepwisemodel, direction = "both")
summary(model.stepwise)
anova(model.stepwise)

model.ls1_base <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                  avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                data.train.std.y)

anova(model.ls1_base)
summary(model.ls1_base)
par(mfrow=c(2,2))  # visualize four graphs at once
plot(model.ls1_base)
vif(model.ls1_base)
varImp(model.ls1_base)
par(mfrow=c(1,1))

#Performance Metrics
mse <- function(sm) 
  mean(sm$residuals^2) 
mse(model.ls1_base) #M1: 1.602694;M2: 1.602695
AIC(model.ls1_base) #M1:6646.579;M2:6644.58 
BIC(model.ls1_base) #M1:6769.743;M2:6762.146

#Performance
pred.valid.ls1 <- predict(model.ls1_base, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ls1)^2) # mean prediction error
# M1: 1.867523;M2:1.867433

sd((y.valid - pred.valid.ls1)^2)/sqrt(n.valid.y) # std error
# M1:0.1696615;M2:0.1696498

###Least Squares Regression (Baseline 2)###
model.ls1_base2 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + 
                       avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                     data.train.std.y)

anova(model.ls1_base2)
summary(model.ls1_base2)
par(mfrow=c(2,2))  # visualize four graphs at once
plot(model.ls1_base2)
vif(model.ls1_base2)
varImp(model.ls1_base2)
par(mfrow=c(1,1))

#Performance Metrics
mse <- function(sm) 
  mean(sm$residuals^2) 
mse(model.ls1_base2) #M1: 1.602694;M2: 1.602695
AIC(model.ls1_base2) #M1:6646.579;M2:6644.58 
BIC(model.ls1_base2) #M1:6769.743;M2:6762.146

#Performance
pred.valid.ls1 <- predict(model.ls1_base2, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ls1)^2) # mean prediction error
# M1: 1.867523;M2:1.867433

sd((y.valid - pred.valid.ls1)^2)/sqrt(n.valid.y) # std error
# M1:0.1696615;M2:0.1696498

###Least Squares Regression Model 1###

model.ls1 <- lm(damt ~ reg2 + reg3 + reg4 + home+ + chld + hinc + incm + 
                  plow + npro + rgif + agif, 
                data.train.std.y) 

anova(model.ls1)
summary(model.ls1)
par(mfrow=c(2,2))  # visualize four graphs at once
plot(model.ls1)
vif(model.ls1)
varImp(model.ls1)
par(mfrow=c(1,1))

#Performance Metrics
mse <- function(sm) 
  mean(sm$residuals^2) 
mse(model.ls1) #1.614779
AIC(model.ls1) #6643.565
BIC(model.ls1) #6716.344

#Performance
pred.valid.ls1 <- predict(model.ls1, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ls1)^2) # mean prediction error
# 1.846847

sd((y.valid - pred.valid.ls1)^2)/sqrt(n.valid.y) # std error
# 0.1682736

###Best Subset Selection with k-fold cross-validation Model 1###

predict.regsubsets=function(object,newdata,id,...){
  form=as.formula(object$call[[2]])
  mat=model.matrix(form,newdata)
  coefi=coef(object,id=id)
  mat[,names(coefi)]%*%coefi
}

set.seed(11)

#k=10 folds
folds=sample(rep(1:10,length=nrow(data.train.std.y)))
folds

#Balanced folds
table(folds)

#20 columns for the 10 variables and 10 rows for 10 folds. Train all all observations, but the k fold. Look at all
#subsets. 
cv.errors=matrix(NA,10,20)

for(k in 1:10){
  best.fit=regsubsets(damt~.,data=data.train.std.y[folds!=k,],nvmax=20,method="forward")
  for(i in 1:20){
    pred=predict(best.fit,data.train.std.y[folds==k,],id=i)
    cv.errors[k,i]=mean((data.train.std.y$damt[folds==k]-pred)^2)
  }
}

mean.cv.errors=apply(cv.errors,2,mean)
mean.cv.errors
which.min(mean.cv.errors) #lowest is 10 variable model

rmse.cv=sqrt(apply(cv.errors,2,mean))
plot(rmse.cv,pch=19,type="b")

reg.best=regsubsets(damt~.,data=data.train.std.y, nvmax=20)
coef(reg.best,10)

model.regfit1 <- lm(damt ~ reg3 + reg4 + home + chld + hinc + incm +
                  plow + npro + rgif + agif, 
                data.train.std.y) #only difference from standard ols is minus reg2 

summary(model.regfit1)
anova(model.regfit1)
par(mfrow=c(2,2))  # visualize four graphs at once
plot(model.regfit1)
vif(model.regfit1)
varImp(model.regfit1)
par(mfrow=c(1,1))

#Performance Metrics
mse <- function(sm) 
  mean(sm$residuals^2) 
mse(model.regfit1) #1.616685
AIC(model.regfit1) #6643.918
BIC(model.regfit1) #6711.099

#Performance
pred.valid.regfit1<- predict(model.regfit1, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.regfit1)^2) # mean prediction error
# 1.857947

sd((y.valid - pred.valid.regfit1)^2)/sqrt(n.valid.y) # std error
# 0.1693538

###Principal Components Regression###

set.seed(1)
pcr.fit = pcr(damt ~ ., data = data.train.std.y,scale = TRUE, validation = "CV")

summary(pcr.fit)

par(mfrow=c(1,1))
validationplot(pcr.fit, val.type = "MSEP", type = "b") #elbow at component = 5, but second elbow at 15
par(mfrow=c(1,1))

#Performance
set.seed(1)
pred.valid.pcr1<- predict(pcr.fit, newdata = data.valid.std.y,ncomp=15) # validation predictions
mean((y.valid - pred.valid.pcr1)^2) # mean prediction error
# 1.865497

sd((y.valid - pred.valid.pcr1)^2)/sqrt(n.valid.y) # std error
# 0.1698902

###Partial Least Squares###
set.seed(1)
pls.fit = plsr(damt ~ ., data = data.train.std.y,scale = TRUE, validation = "CV")
varImp(pls.fit)

summary(pls.fit)

validationplot(pls.fit, val.type = "MSEP", type = "b") #elbow at component 3

#Performance
set.seed(1)
pred.valid.pls1<- predict(pls.fit, newdata = data.valid.std.y,ncomp=3) # validation predictions
mean((y.valid - pred.valid.pls1)^2) # mean prediction error
# 1.879345

sd((y.valid - pred.valid.pls1)^2)/sqrt(n.valid.y) # std error
# 0.1718698

###Ridge Regression###

x=model.matrix(damt~.-1,data=data.train.std.y)
y=data.train.std.y$damt

fit.ridge=glmnet(x,y,alpha=0)
dim(coef(fit.ridge))

plot(fit.ridge,xvar="lambda",label=TRUE)

#10 fold cross-validation
set.seed(1)
cv.ridge=cv.glmnet(x,y,alpha=0)

plot(cv.ridge)
bestlam = cv.ridge$lambda.min
bestlam

#Performance
ridge.valid = as.matrix(data.valid.std.y)
ridge.valid <- ridge.valid[,-21]

ridge.pred = predict(fit.ridge, s=bestlam, newx=ridge.valid)

set.seed(1)
mean((y.valid - ridge.pred)^2) # mean prediction error
# 1.873279

sd((y.valid - ridge.pred)^2)/sqrt(n.valid.y) # std error
# 0.1711228

###The Lasso###

x=model.matrix(damt~.-1,data=data.train.std.y)
y=data.train.std.y$damt

fit.lasso=glmnet(x,y,alpha=1)
dim(coef(fit.lasso))

plot(fit.lasso,xvar="lambda",label=TRUE) #Top shows how many variables are non-zero in the model. 
plot(fit.lasso,xvar="dev",label=TRUE) ##Fraction Deviance Explained is similar to R2. 
#Coefficients grow very large at end of path, shows overfitting. 

#10 fold cross-validation
set.seed(1)
cv.lasso=cv.glmnet(x,y,alpha=1)
plot(cv.lasso)
coef(cv.lasso) 

bestlam_lasso = cv.lasso$lambda.min
bestlam_lasso

#Performance
lasso.valid = as.matrix(data.valid.std.y)
lasso.valid <- lasso.valid[,-21]

set.seed(1)
lasso.pred = predict(fit.lasso, s=bestlam_lasso, newx=lasso.valid)

mean((y.valid - lasso.pred)^2) # mean prediction error
# 1.861709

sd((y.valid - lasso.pred)^2)/sqrt(n.valid.y) # std error
# 0.1694331

##### Neural Network #####

#Use cv/grid search to find optimal
fitControl <- trainControl(method="repeatedcv", number=5)

set.seed(1)
gbmFit <- train(damt ~ ., data.train.std.y,
                method = "nnet", trControl = fitControl, verbose = FALSE)
gbmFit
plot(gbmFit)
#The final values used for the model were size = 1 and decay = 1e-04. However, model below is better. 

set.seed(800)
model.nnet2 <- avNNet(damt ~ ., data.train.std.y, size = 5, linout = TRUE, decay=0.01, maxit=500) #5
model.nnet2

#Performance
pred.nnet2 <- predict(model.nnet2, newdata = data.valid.std.y)

mean((y.valid - pred.nnet2)^2) # mean prediction error
# 1.466398

sd((y.valid - pred.nnet2)^2)/sqrt(n.valid.y) # std error
#  0.1620263

##### Decision Tree ##### 

library(MASS)
tree.damt=tree(damt~., data.train.std.y)
summary(tree.damt)

#Plot the Tree
plot(tree.damt)
text(tree.damt,pretty=0)

#Use Cross-Validation to Prune Tree #
cv.damt=cv.tree(tree.damt)
par(mfrow=c(1,1))
plot(cv.damt$size,cv.damt$dev,type='b')
par(mfrow=c(1,1))

prune.damt=prune.tree(tree.damt,best=11)
plot(prune.damt )
text(prune.damt,pretty=0)

#Performance
pred.tree1 <- predict(prune.damt, newdata = data.valid.std.y)

mean((y.valid - pred.tree1)^2) # mean prediction error
# 2.241075

sd((y.valid - pred.tree1)^2)/sqrt(n.valid.y) # std error
#  0.1920681

##### Bagging ##### 

library(randomForest)
set.seed(1)
bag.damt=randomForest(damt~.,data= data.train.std.y,mtry=20,importance=TRUE, ntree=100)
bag.damt

#Performance
pred.bag1 <- predict(bag.damt, newdata = data.valid.std.y)

mean((y.valid - pred.bag1)^2) # mean prediction error
# 1.699838

sd((y.valid - pred.bag1)^2)/sqrt(n.valid.y) # std error
#  0.1759852

##### Random Forest ##### 

library(randomForest)
set.seed(1)

#OOB Error vs. Validation Error to find ideal mtry

#oob.err=double(20)
#test.err=double(20)
#for(mtry in 1:20){
#fit=randomForest(damt~., data= data.train.std.y, mtry=mtry, ntree=400)
#oob.err[mtry]=fit$mse[400]
#pred=predict(fit, data.valid.std.y)

#test.err[mtry]=with(data.valid.std.y,mean((damt-pred)^2))
#cat(mtry," ")
#}

#matplot(1:mtry,cbind(test.err,oob.err),pch=19,col=c("red","blue"),type="b",ylab="Mean Squared Error")
#legend("topright", legend=c("OOB", "Test"), pch=19, col=c("red", "blue"))

set.seed(1)
rf.damt=randomForest(damt~.,data= data.train.std.y, mtry=4, importance=TRUE,ntree=400)
rf.damt

importance(rf.damt)
varImpPlot(rf.damt)

#Performance
set.seed(1)
pred.rf1 <- predict(rf.damt, newdata = data.valid.std.y)

mean((y.valid - pred.rf1)^2) # mean prediction error
#  1.663914

sd((y.valid - pred.rf1)^2)/sqrt(n.valid.y) # std error
#  0.1733175

##### Boosting ##### 

#Boosting - Regression 

require(gbm)
library(caret)

#Find ideal parameter using CV using manual grid search
set.seed(7)
control <- trainControl(method="cv", number=5)
grid <- expand.grid(interaction.depth=seq(1,7,by=2),n.trees=seq(100,1000,by=50),shrinkage=c(0.01,0.1),n.minobsinnode = 10)
model <- train(damt~., data=data.train.std.y, method="gbm", trControl=control, tuneGrid=grid)
print(model)
#The final values used for the model were n.trees = 400, interaction.depth = 1, shrinkage = 0.1 and n.minobsinnode = 10.

#Find ideal parameter using CV using automatic grid search
#fitControl <- trainControl(method="repeatedcv", number=5, repeats=5)
#set.seed(1)
#gbmFit <- train(damt ~ ., data = data.train.std.y,
                #method = "gbm", trControl = fitControl)
#gbmFit
#The final values used for the model were n.trees = 150, interaction.depth = 2, shrinkage = 0.1 and n.minobsinnode = 10. 
#NOTE: Ran CV, but below model performed better. 

#Fit Boost
set.seed(1)
boost.damt=gbm(damt~.,data = data.train.std.y,distribution="gaussian",n.trees=400,shrinkage = 0.1,interaction.depth=1)
summary(boost.damt)
plot(boost.damt,i="chld")

#Performance
pred.boost1 <- predict(boost.damt, newdata = data.valid.std.y,n.trees=400)

mean((y.valid - pred.boost1)^2) # mean prediction error
#  1.334194

sd((y.valid - pred.boost1)^2)/sqrt(n.valid.y) # std error
#  0.1515649

#Find number of ideal trees using the test error as a function of the number of trees, and make a plot
set.seed(1)
n.trees=seq(from=25,to=1000, by=25)
predmat=predict(boost.damt,newdata= data.valid.std.y,n.trees = n.trees)
dim(predmat)

berr=with(data.valid.std.y,apply((predmat-damt)^2,2,mean)) #columwise MSE
plot(n.trees,berr,pch=19,ylab="Mean Squared Error",xlab="# Trees", main = "Boosting
     Test Error")

##### Support Vector Regression - Linear Kernel #####
#Grid Search (Caret Package Method)
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid <- expand.grid(C = c(0.001, 0.01, 0.1,0.15, 1))
set.seed(800)
svm_Linear_Grid <- train(damt ~., data = data.train.std.y, method = "svmLinear",
                         preProcess = c("center", "scale"),
                         tuneGrid = grid,
                         tuneLength = 10)

svm_Linear_Grid
plot(svm_Linear_Grid) #C = 0.01

#Performance
pred.svm_Linear_Grid <- predict(svm_Linear_Grid, newdata = data.valid.std.y)

mean((y.valid - pred.svm_Linear_Grid)^2) # mean prediction error
# 1.865914

sd((y.valid - pred.svm_Linear_Grid)^2)/sqrt(n.valid.y) # std error
#  0.1785505

##### Support Vector Regression - Non-Linear Radial Kernel #####
#Grid Search (Caret Package Method)
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid_radial <- expand.grid(sigma = c(0.005, 0.01, 0.025, 0.05),
                           C = c(0.01,0.1,1, 5, 10))
set.seed(800)
svm_Radial_Grid <- train(damt ~., data = data.train.std.y, method = "svmRadial",
                         preProcess = c("center", "scale"),
                         tuneGrid = grid_radial,
                         tuneLength = 10)

svm_Radial_Grid
plot(svm_Radial_Grid) #Cost 5, gamma 0.005

#Performance
pred.svm_Radial_Grid<- predict(svm_Radial_Grid, newdata = data.valid.std.y)

mean((y.valid - pred.svm_Radial_Grid)^2) # mean prediction error
# 1.568629

sd((y.valid - pred.svm_Radial_Grid)^2)/sqrt(n.valid.y) # std error
#  0.1734832

##### Results Model Comparison (Example) #####

# MPE  Model
# 1.867523 LS1
# 1.867433 LS2
#...see report for full details

###### Final Model Selection #####

# Select boost.damt since it has minimum mean prediction error in the validation sample

yhat.test <- predict(boost.damt, newdata = data.test.std,n.trees=400) # test predictions

###################################### FINAL RESULTS ######################################

# Save final results for both classification and regression

#Classification Model
length(chat.test) # check length = 2007
chat.test[1:10] # check this consists of 0s and 1s

#Prediction Model
length(yhat.test) # check length = 2007
yhat.test[1:10] # check this consists of plausible predictions of damt
summary(yhat.test)

#Final Results Summary for Classification and Predictiion Amount
results.data.frame <- data.frame(chat = chat.test, yhat = yhat.test)
View(results.data.frame)

#Check expected profit from mailing
Predicted_Donors <- subset(results.data.frame, chat==1)
Predicted_Profit <- summary(Predicted_Donors$yhat)
Predicted_Profit #Predicted profit is $4303.29

Predicted_Donation_Amount <- mean(Predicted_Donors$yhat)
Predicted_Donation_Amount 
#Out of those who are expected to respond (donate) to the mailing, 
#the average predicted donation is $14.34

#Check for missing values
sum(is.na(results.data.frame))

#Save Test Set Predictions for Classification and Prediction Model
#Submit to Canvas
ip <- data.frame(chat=chat.test, yhat=yhat.test) # data frame with two variables: chat and yhat
write.csv(ip, file="BDY.csv", row.names=FALSE) # use your initials for the file name
