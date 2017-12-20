#get data
library(readr)
library(DMwR)
library(data.table)
library(rpart)
library(randomForest)

NYPD_Motor_Vehicle_Collisions <- read_csv("D:/School/Data Mining/NYPD_Motor_Vehicle_Collisions.csv")

#training data
nypd <- NYPD_Motor_Vehicle_Collisions[c(1:100000), c(2, 5:6, 11, 19:20, 25:26)]
setnames(nypd, 'NUMBER OF PERSONS INJURED', 'INJURED')
#handle nas
nrow(nypd[!complete.cases(nypd),])
clean.nypd<-na.omit(nypd)

#linear models
lm.injured <- lm(clean.nypd$INJURED ~ ., data = clean.nypd)
summary(lm.injured)
#plot(lm.injured)
#backwards elimination of non useful variables
final.lm <- step(lm.injured)
summary(final.lm)
plot(final.lm)

#regression trees
rt.injured <- rpart(clean.nypd$INJURED ~ ., data = clean.nypd)
rt.injured
prettyTree(rt.injured)
printcp(rt.injured)
#select best tree, 1-SE rule
(rt.injured <- rpartXse(clean.nypd$`INJURED` ~ ., data = clean.nypd))

#prediction models
lm.predictions.injured <- predict(final.lm, clean.nypd)
rt.predictions.injured <- predict(rt.injured, clean.nypd)

#regression evaluation metrics
regr.eval(clean.nypd[, "INJURED"], lm.predictions.injured, train.y = clean.nypd$INJURED)
regr.eval(clean.nypd[, "INJURED"], rt.predictions.injured, train.y = clean.nypd$INJURED)

#scatter plot of errors
old.par <- par(mfrow = c(1,2))
plot(lm.predictions.injured, clean.nypd$INJURED, main = "Linear Model", xlab = "Predictions", ylab = "True Values")
abline(0, 1, lty = 2)
plot(rt.predictions.injured, clean.nypd$INJURED, main = "Regression Tree", xlab = "Predictions", ylab = "True Values")
abline(0, 1, lty = 2)
par(old.par)

#improve linear model performance
sensible.lm.predictions.injured <- ifelse(lm.predictions.injured < 0, 0, lm.predictions.injured)
regr.eval(clean.nypd[, "INJURED"], lm.predictions.injured, stats = c("mae", "mse"))
regr.eval(clean.nypd[, "INJURED"], sensible.lm.predictions.injured, stats = c("mae", "mse"))

cv.rpart <- function(form,train,test,...) {
   m <- rpartXse(form,train,...)
   p <- predict(m,test)
   mse <- mean((p-resp(form,test))^2)
   c(nmse=mse/mean((mean(resp(form,train))-resp(form,test))^2))
}

cv.lm <- function(form,train,test,...) {
   m <- lm(form,train,...)
   p <- predict(m,test)
   p <- ifelse(p < 0,0,p)
   mse <- mean((p-resp(form,test))^2)
   c(nmse=mse/mean((mean(resp(form,train))-resp(form,test))^2))
}

#cross validation comparison with linear model and regression tree
res <- experimentalComparison(
  c(dataset(INJURED ~ ., data = clean.nypd[,1:4, 7:8],'INJURED')),
  c(variants('cv.lm'),
    variants('cv.rpart',se=c(0,0.5,1))),
  cvSettings(3,10,1234)
)

summary(res)
plot(res)
getVariant("cv.rpart.v1", res)
bestScores(res)
compAnalysis(res,against = 'cv.rpart.v1',datasets = 'INJURED')

#cross validation comparison with linear model, regression tree and random forest
#cv.rf <- function(form,train,test,...) {
#   m <- randomForest(form,train,...)
#   p <- predict(m,test)
#   mse <- mean((p-resp(form,test))^2)
#   c(nmse=mse/mean((mean(resp(form,train))-resp(form,test))^2))
#}

#res.all <- experimentalComparison(
#  (dataset(INJURED ~ ., data = clean.nypd[,1:4, 7:8],'INJURED')),
#   c(variants('cv.lm'),
#       variants('cv.rpart',se=c(0,0.5,1)),
#       variants('cv.rf',ntree=c(200,500,700))
#       ),
#   cvSettings(5,10,1234))

#summary(res.all)
#plot(res.all)
#getVariant("", res.all)
#bestScores(res)
#compAnalysis(res.all,against = '',datasets = 'INJURED')

#obtaining predictions
bestModelsNames <- sapply(bestScores(res),function(x) x['nmse','system'])
learners <- c(rpart='rpartXse')
funcs <- learners
parSetts <- lapply(bestModelsNames,function(x) getVariant(x,res)@pars)
bestModels <- list()
form <- as.formula(paste(names(clean.nypd)[4],'~ .'))
bestModels[[1]] <- do.call(funcs,c(list(form,clean.nypd[,c(1:7)]),parSetts[[1]]))

#test data
test.nypd <- NYPD_Motor_Vehicle_Collisions[c(100001:110000), c(2, 5:6, 11, 19:20, 25:26)]
setnames(test.nypd, 'NUMBER OF PERSONS INJURED', 'INJURED')
#handle nas
nrow(test.nypd[!complete.cases(test.nypd),])
clean.test.nypd<-na.omit(test.nypd)

#prediction matrix
preds <- matrix(ncol = 1, nrow = 6211)

for(i in 1:nrow(clean.test.nypd)) {
  preds[i,] <- sapply(1,
                      function(x)
                        predict(bestModels[[x]], clean.test.nypd[i,]))
                      
}

avg.vals <- apply(clean.test.nypd[, 4],2,mean)
avg.preds <- apply(preds,2,mean)
final <- matrix(ncol = 2, nrow = 1)
final[1, 1] <- avg.preds
final[1, 2] <- avg.vals
colnames(final) <- c("average real injuries", " |  average predicted injures")
rownames(final) <- "values"
final
