

library(MASS)
library(pROC)
library(car)       
library(reshape2)
library(ggplot2)
library(vcd)
library(scales)
library(mgcv)
library(dplyr)
library(tidyr)
#load data, create sets


FullData <- read.csv("/Users/austinphillips/Desktop/School/Fall 2025/Data mining/card_transdata.csv", 
                     header = TRUE)

trainset100     <- FullData[1:100, ]
trainset500     <- FullData[1:500, ]
trainset2500    <- FullData[1:2500, ]
trainset25000   <- FullData[1:25000, ]
trainset125000  <- FullData[1:125000, ]
trainset500000  <- FullData[1:500000, ]
testset         <- FullData[500001:1000000, ]


#test multicollinearity 

pearson_res <- cor.test(FullData$distance_from_home,
                        FullData$distance_from_last_transaction,
                        method = "pearson")

spearman_res <- cor.test(FullData$distance_from_home,
                         FullData$distance_from_last_transaction,
                         method = "spearman")

cat(sprintf("Pearson  r = %.4f, p = %.2e\n", pearson_res$estimate, pearson_res$p.value))
cat(sprintf("Spearman ρ = %.4f, p = %.2e\n", spearman_res$estimate, spearman_res$p.value))

tab <- table(FullData$used_chip, FullData$used_pin_number,
             dnn = c("Chip", "PIN"))

print(tab)


cramv <- assocstats(tab)$cramer
cat(sprintf("Cramér’s V = %.4f\n", cramv))


# naive models
fullmodel100     <- glm(fraud ~ ., data = trainset100,     family = binomial)
fullmodel500     <- glm(fraud ~ ., data = trainset500,     family = binomial)
fullmodel2500    <- glm(fraud ~ ., data = trainset2500,    family = binomial)
fullmodel25000   <- glm(fraud ~ ., data = trainset25000,   family = binomial)
fullmodel125000  <- glm(fraud ~ ., data = trainset125000,  family = binomial)
fullmodel500000  <- glm(fraud ~ ., data = trainset500000,  family = binomial)

#eval
evaluateModel <- function(model, testData) {
  predProb  <- predict(model, newdata = testData, type = "response")
  predClass <- ifelse(predProb > 0.5, 1, 0)
  
  aucVal <- auc(roc(testData$fraud, predProb))
  cm     <- table(Predicted = predClass, Actual = testData$fraud)
  
  accuracy     <- sum(diag(cm)) / sum(cm)
  sensitivity  <- cm[2,2] / sum(cm[2,])
  specificity  <- cm[1,1] / sum(cm[1,])
  precision    <- cm[2,2] / sum(cm[,2])
  f1           <- 2 * precision * sensitivity / (precision + sensitivity)
  
  return(data.frame(
    AUC = aucVal,
    Accuracy = accuracy,
    Sensitivity = sensitivity,
    Precision = precision,
    Specificity = specificity,
    F1 = f1
  ))
}

#evaluate naive models
resultsnaive <- rbind(
  evaluateModel(fullmodel100,     testset),
  evaluateModel(fullmodel500,     testset),
  evaluateModel(fullmodel2500,    testset),
  evaluateModel(fullmodel25000,   testset),
  evaluateModel(fullmodel125000,  testset),
  evaluateModel(fullmodel500000,  testset)
)

resultsnaive$TrainSize <- c(100, 500, 2500, 25000, 125000, 500000)

print(resultsnaive)

#plot prework

plotData <- melt(resultsnaive, 
                 id.vars = "TrainSize",
                 measure.vars = c("AUC", "Accuracy", "Sensitivity", 
                                  "Precision", "Specificity", "F1"),
                 variable.name = "Metric",
                 value.name = "Value")

plotData$Metric <- factor(plotData$Metric,
                          levels = c("AUC", "Accuracy", "Specificity", 
                                     "Precision", "Sensitivity", "F1"))

#plot
naiveplot <- ggplot(plotData, aes(x = TrainSize, y = Value, color = Metric)) +
  geom_line(size = 1.2) +
  geom_point(size = 2.5, shape = 21, fill = "white", stroke = 1.2) +
  
  scale_x_log10(breaks = c(100, 500, 2500, 25000, 125000, 500000),
                labels = scales::comma) +
  
  scale_y_continuous(limits = c(0.5, 1.0), 
                     breaks = seq(0.5, 1.0, 0.1),
                     labels = scales::percent_format()) +
  scale_color_brewer(palette = "Set1") +
  
  labs(title = "Learning Curves: Naive Model",
       x = "Training Set Size (log scale)",
       y = "Performance Metric",
       color = "Metric") +
  
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray50"),
    axis.title = element_text(face = "bold"),
    legend.position = "right",
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "gray90"),
    plot.caption = element_text(color = "gray60", hjust = 0)
  ) 
print(naiveplot)


#transformation

vars <- c("distance_from_home",
          "distance_from_last_transaction",
          "ratio_to_median_purchase_price")

X <- FullData[ , vars]

#shift values for box-cox
bc_lambda <- sapply(vars, function(v) {
  y <- X[[v]]
  if (any(y <= 0)) {
    y <- y + abs(min(y, na.rm = TRUE)) + 1
  }
  bc <- boxcox(y ~ 1, lambda = seq(-2, 2, 0.1), plotit = FALSE)
  lambda <- bc$x[which.max(bc$y)]
  round(lambda, 3)
})

bc_df <- data.frame(Variable = vars, Lambda = bc_lambda)
print(bc_df)


#apply transformations

FullDataLog <- FullData
FullDataLog$dist_home_log     <- log(FullData$distance_from_home + 1)
FullDataLog$dist_last_log     <- log(FullData$distance_from_last_transaction + 1)
FullDataLog$ratio_log         <- log(FullData$ratio_to_median_purchase_price + 1)


trainset100     <- FullDataLog[1:100, ]
trainset500     <- FullDataLog[1:500, ]
trainset2500    <- FullDataLog[1:2500, ]
trainset25000   <- FullDataLog[1:25000, ]
trainset125000  <- FullDataLog[1:125000, ]
trainset500000  <- FullDataLog[1:500000, ]
testset         <- FullDataLog[500001:1000000, ]


fulllogmodel100 <- glm(fraud ~ dist_home_log + dist_last_log + ratio_log +
                      repeat_retailer + used_chip + used_pin_number + online_order,
                    data = trainset100, family = binomial)
fulllogmodel500 <- glm(fraud ~ dist_home_log + dist_last_log + ratio_log +
                         repeat_retailer + used_chip + used_pin_number + online_order,
                       data = trainset500, family = binomial)
fulllogmodel2500 <- glm(fraud ~ dist_home_log + dist_last_log + ratio_log +
                         repeat_retailer + used_chip + used_pin_number + online_order,
                       data = trainset2500, family = binomial)
fulllogmodel25000 <- glm(fraud ~ dist_home_log + dist_last_log + ratio_log +
                          repeat_retailer + used_chip + used_pin_number + online_order,
                        data = trainset25000, family = binomial)
fulllogmodel125000 <- glm(fraud ~ dist_home_log + dist_last_log + ratio_log +
                           repeat_retailer + used_chip + used_pin_number + online_order,
                         data = trainset125000, family = binomial)
fulllogmodel500000 <- glm(fraud ~ dist_home_log + dist_last_log + ratio_log +
                           repeat_retailer + used_chip + used_pin_number + online_order,
                         data = trainset500000, family = binomial)

resultslog <- rbind(
  evaluateModel(fulllogmodel100,     testset),
  evaluateModel(fulllogmodel500,     testset),
  evaluateModel(fulllogmodel2500,    testset),
  evaluateModel(fulllogmodel25000,   testset),
  evaluateModel(fulllogmodel125000,  testset),
  evaluateModel(fulllogmodel500000,  testset)
)

resultslog$TrainSize <- c(100, 500, 2500, 25000, 125000, 500000)
resultslog$RunTime <- c(.01,.01,.04,.17, .97,3.9)

print(resultslog)


# plot log plot

plotLog <- melt(resultslog,
                id.vars = "TrainSize",
                measure.vars = c("AUC","Accuracy","Sensitivity",
                                 "Precision","Specificity","F1"),
                variable.name = "Metric",
                value.name = "Value")
plotLog$Model <- "Log-Transformed"


plotAll <- plotLog

plotAll$Metric <- factor(plotAll$Metric,
                         levels = c("AUC","Accuracy","Specificity",
                                    "Precision","Sensitivity","F1"))

logplot <- ggplot(plotAll, aes(x = TrainSize, y = Value,
                         color = Metric, linetype = Model)) +
  geom_line(size = 1.2) +
  geom_point(size = 2.5, shape = 21, fill = "white", stroke = 1.2) +
  
  scale_x_log10(breaks = c(100,500,2500,25000,125000,500000),
                labels = comma) +
  
  scale_y_continuous(limits = c(0.5,1.0),
                     breaks = seq(0.5,1.0,0.1),
                     labels = percent_format()) +
  
  scale_color_brewer(palette = "Set1") +
  
  labs(title = "Learning Curves – Log-Transformed Predictors",
       x = "Training Set Size (log scale)",
       y = "Performance Metric",
       color = "Metric",
       linetype = "Model",
       caption = "Sufficient n ≈ 25 000 (AUC > 0.94, F1 > 0.89)") +
  
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray50"),
    axis.title    = element_text(face = "bold"),
    legend.position = "right",
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "gray90"),
    plot.caption  = element_text(color = "gray60", hjust = 0)
  ) +
  

print(logplot)


#test linear logit

train_large <- FullDataLog[1:500000, ]

#transform data
train_large$dist_home_pos <- train_large$distance_from_home + 1
train_large$dist_last_pos <- train_large$distance_from_last_transaction + 1
train_large$ratio_pos     <- train_large$ratio_to_median_purchase_price + 1

#interactions
train_large$dist_home_bt <- train_large$dist_home_pos * log(train_large$dist_home_pos)
train_large$dist_last_bt <- train_large$dist_last_pos * log(train_large$dist_last_pos)
train_large$ratio_bt     <- train_large$ratio_pos * log(train_large$ratio_pos)

# Fit model
bt_model <- glm(fraud ~ dist_home_pos + dist_home_bt +
                  dist_last_pos + dist_last_bt +
                  ratio_pos + ratio_bt +
                  repeat_retailer + used_chip + used_pin_number + online_order,
                data = train_large, family = binomial)

#selest valuse
bt_summary <- summary(bt_model)$coefficients
bt_rows <- grep("_bt$", rownames(bt_summary))
bt_pvals <- bt_summary[bt_rows, "Pr(>|z|)"]
names(bt_pvals) <- c("dist_home_logx", "dist_last_logx", "ratio_logx")

print(round(bt_pvals, 6))


# gam models

# Use log-transformed vars + binary ones
gam_model100 <- gam(fraud ~ s(dist_home_log) + s(dist_last_log) + s(ratio_log) +
                      repeat_retailer + used_chip + used_pin_number + online_order,
                    data = trainset100, family = binomial)
gam_model500 <- gam(fraud ~ s(dist_home_log) + s(dist_last_log) + s(ratio_log) +
                   repeat_retailer + used_chip + used_pin_number + online_order,
                 data = trainset500, family = binomial)
gam_model2500 <- gam(fraud ~ s(dist_home_log) + s(dist_last_log) + s(ratio_log) +
                      repeat_retailer + used_chip + used_pin_number + online_order,
                    data = trainset2500, family = binomial)
gam_model25000 <- gam(fraud ~ s(dist_home_log) + s(dist_last_log) + s(ratio_log) +
                       repeat_retailer + used_chip + used_pin_number + online_order,
                     data = trainset25000, family = binomial)
gam_model125000 <- gam(fraud ~ s(dist_home_log) + s(dist_last_log) + s(ratio_log) +
                        repeat_retailer + used_chip + used_pin_number + online_order,
                      data = trainset125000, family = binomial)
gam_model500000 <- gam(fraud ~ s(dist_home_log) + s(dist_last_log) + s(ratio_log) +
                      repeat_retailer + used_chip + used_pin_number + online_order,
                    data = trainset500000, family = binomial)

resultsgam <- rbind(
  evaluateModel(gam_model100,     testset),
  evaluateModel(gam_model500,     testset),
  evaluateModel(gam_model2500,    testset),
  evaluateModel(gam_model25000,   testset),
  evaluateModel(gam_model125000,  testset),
  evaluateModel(gam_model500000,  testset)
)


resultsgam$TrainSize <- c(100, 500, 2500, 25000, 125000, 500000)
resultsgam$RunTime <- c(.9,8.7,1.7,12.8,71.3,347.7)
print(resultsgam)


#gam plot

plotLog <- melt(resultsgam,
                id.vars = "TrainSize",
                measure.vars = c("AUC","Accuracy","Sensitivity",
                                 "Precision","Specificity","F1"),
                variable.name = "Metric",
                value.name = "Value")
plotLog$Model <- "Log-Transformed"


plotAll <- plotLog

plotAll$Metric <- factor(plotAll$Metric,
                         levels = c("AUC","Accuracy","Specificity",
                                    "Precision","Sensitivity","F1"))

logplot <- ggplot(plotAll, aes(x = TrainSize, y = Value,
                               color = Metric, linetype = Model)) +
  geom_line(size = 1.2) +
  geom_point(size = 2.5, shape = 21, fill = "white", stroke = 1.2) +
  
  scale_x_log10(breaks = c(100,500,2500,25000,125000,500000),
                labels = comma) +
  
  scale_y_continuous(limits = c(0.5,1.0),
                     breaks = seq(0.5,1.0,0.1),
                     labels = percent_format()) +
  
  scale_color_brewer(palette = "Set1") +
  
  labs(title = "Learning Curves – GAB Predictors",
       x = "Training Set Size (log scale)",
       y = "Performance Metric",
       color = "Metric",
       linetype = "Model") +
  
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray50"),
    axis.title    = element_text(face = "bold"),
    legend.position = "right",
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "gray90"),
    plot.caption  = element_text(color = "gray60", hjust = 0)
  ) +
  
  geom_vline(xintercept = 25000, linetype = "dashed",
             color = "darkred", size = 0.8) +
  annotate("text", x = 25000, y = 0.55,
           label = "Sufficient n", angle = 90,
           vjust = -0.5, color = "darkred", fontface = "bold")

print(logplot)


#gam vs naive 





resultsnaive$AUC <- as.numeric(resultsnaive$AUC)
resultsgam$AUC   <- as.numeric(resultsgam$AUC)

# Now build comparison
comp_25k <- bind_rows(
  resultsnaive %>% filter(TrainSize == 25000) %>% mutate(Model = "Naïve GLM"),
  resultsgam   %>% filter(TrainSize == 25000) %>% mutate(Model = "GAM")
) %>%
  select(Model, AUC, Accuracy, Sensitivity, Precision, Specificity, F1) %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value") %>%
  mutate(
    Metric = factor(Metric, levels = c("AUC", "Accuracy", "Specificity",
                                       "Precision", "Sensitivity", "F1")),
    Model  = factor(Model, levels = c("Naïve GLM", "GAM"))
  )

# Plot
bar_25k <- ggplot(comp_25k, aes(x = Metric, y = Value, fill = Model)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7, colour = "black") +
  geom_text(aes(label = percent(Value, accuracy = 0.1)),
            position = position_dodge(width = 0.75), vjust = -0.5,
            size = 3.5, fontface = "bold") +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
  scale_fill_brewer(palette = "Set1") +
  labs(title = "GAM vs Naïve GLM Where n = 25,000 ",
       x = "Metric", y = "Value", fill = "Model") +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.title = element_text(face = "bold"),
    legend.position = "top",
    panel.grid.major.x = element_blank()
  )

print(bar_25k)
#gam vs log reg run time

runtime_df <- bind_rows(
  resultslog %>% select(TrainSize, RunTime) %>% mutate(Model = "Log-GLM"),
  resultsgam %>% select(TrainSize, RunTime) %>% mutate(Model = "GAM")
) %>%
  mutate(Model = factor(Model, levels = c("Log-GLM","GAM")))

runtime_plot <- ggplot(runtime_df, aes(x = TrainSize, y = RunTime,
                                       colour = Model, linetype = Model)) +
  geom_line(size = 1.2) +
  geom_point(size = 3, shape = 21, fill = "white", stroke = 1.2) +
  scale_x_log10(breaks = c(100,500,2500,25000,125000,500000),
                labels = comma) +
  scale_y_continuous(labels = comma) +
  scale_colour_brewer(palette = "Set1") +
  labs(title = "Training Run-time: Log-Transformed GLM vs GAM",
       x = "Training Set Size (log scale)",
       y = "Run-time (seconds)",
       colour = "Model", linetype = "Model") +
  theme_minimal(base_size = 12) +
  theme(
    plot.title   = element_text(face = "bold", size = 14),
    axis.title   = element_text(face = "bold"),
    legend.position = "top",
    panel.grid.minor = element_blank()
  )

print(runtime_plot)
