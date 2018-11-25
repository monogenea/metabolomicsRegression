

#### PREDICT BIOMASS IN MAIZE HYBRIDS USING THEIR METABOLIC PROFILES ####
# UPDATED - Sun Nov 25 13:04:35 2018 ------------------------------


### Load biomass and root metabolic data set described in de Abreu e Lima et al. (2017)

library(caret)
library(pheatmap)
library(doMC)

data <- read.delim("rootdata.txt", row.names = 1) # metabolic data set
biomass <- read.delim("biomass.txt", row.names = 1) # biomass data set; also contains Dry Weight

### The entries in both data sets should match; n = 363, p = 165
all(rownames(data) == rownames(biomass)) # TRUE

# Metabolites were log-transformed prior to BLUEs (cf. de Abreu e Lima et al. (2017))
# No missing values as well; standardize metabolite levels
scaledData <- scale(data)

# Inspect colinearity
pheatmap(cor(scaledData), scale = "none",
         show_rownames = F, show_colnames = F, breaks = seq(-1,1,length=100))

# Remove highly correlated metabolites (r > 0.9)

corMetabs <- findCorrelation(cor(scaledData), cutoff = 0.9, verbose = F, exact = T)
predictors <- scaledData[,-corMetabs] # replace "scaledData" above with "predictors" to see difference

### MODELS

# For a 10x 5-fold CV, create multifolds so that all models are given the same
# If you want to use CV on a training data, split train and test sets and run
# following on the train set, then predict (use predict function) on the test set
set.seed(100)
myFolds <- createMultiFolds(biomass$FW, k = 5, times = 10)

# Define control parameters, apply "oneSE" rule
myControl <- trainControl(method = "repeatedcv", index = myFolds, selectionFunction = "oneSE")

# Parallelise
registerDoMC(6)

# Fit OLS
mod1 <- train(x = predictors, y = biomass$FW,
              method = "lm",
              metric = "Rsquared",
              trControl = myControl)
mod1

# Fit PLS
mod2 <- train(x = predictors, y = biomass$FW,
              method = "pls",
              # no. of components = tuning parameter! try from 1 to 10
              tuneLength = 10, 
              metric = "Rsquared",
              trControl = myControl)
plot(mod2) # optimal no. of comps = 6

# Fit LASSO

mod3 <- train(x = predictors, y = biomass$FW,
              method = "lasso",
              # fraction = tuning parameter! try 10 values from 0.01 to 0.5
              tuneGrid = data.frame(.fraction = seq(.01,.5,length=10)),
              metric = "Rsquared",
              trControl = myControl)
plot(mod3) # optimal fraction = 0.119

# Fit ridge

mod4 <- train(x = predictors, y = biomass$FW,
              method = "ridge",
              # lambda = tuning parameter! try 10 values 0.1 to 0.6
              tuneGrid = data.frame(.lambda = seq(.1,.6,length=10)),
              metric = "Rsquared",
              trControl = myControl)
plot(mod4) # optimal lambda = 0.6

# Fit ENET

mod5 <- train(x = predictors, y = biomass$FW,
              method = "enet",
              # now we have 2 tuning pars! Provide all combinations of alpha and lambda
              tuneGrid = expand.grid(.lambda = c(0, 0.2, 0.4),
                                    .fraction = c(0.01, 0.1, 0.2)),
              metric = "Rsquared",
              trControl = myControl)
plot(mod5) # optimal parameters lambda = 0 and fraction = 0.1, suggesting LASSO is more important 

# Fit SVR

mod6 <- train(x = predictors, y = biomass$FW,
              method = "svmRadial",
              # 2 tuning pars; we let the kernel sigma to be determined by a default method
              tuneLength = 10, # try C with 0.25, 0.5, 1, 2, 4 and 8
              metric = "Rsquared",
              trControl = myControl)
plot(mod6) # optimal C = 2

# Fit RF

mod7 <- train(x = predictors, y = biomass$FW,
              method = "ranger",
              tuneGrid = expand.grid(mtry = seq(5, (2/3)*ncol(predictors), length = 6),
                                    splitrule = "variance",
                                    min.node.size = 5),
              num.trees = 1000,
              metric = "Rsquared",
              trControl = myControl)
plot(mod7) # 


## Compile all six models and compared performance (R2)

allModels <- resamples(list("OLS" = mod1,
                            "PLS" = mod2,
                            "LASSO" = mod3,
                            "Ridge" = mod4,
                            "ENET" = mod5,
                            "SVR" = mod6,
                            "RF" = mod7))

bwplot(allModels, metric = "Rsquared")

# As expected, the OLS is overfitting in all folds, failing to generalize to the others
# The LASSO seems to perform the best, with R2 ~ 0.40 (as seen above, ENET also relies more on the LASSO penalty)

### Investigate variable importance (ViP) in the LASSO

plot(varImp(mod3),20) # Top 20 metabolites

# Write session out
writeLines(capture.output(sessionInfo()), "sessionInfo")
