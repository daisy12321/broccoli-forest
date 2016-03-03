library(ggplot2)
library(tidyr)
library(dplyr)
setwd("~/Documents/MIT/15.097/broccoli-forest")


recover = read.csv("hw1_output.csv", header = TRUE)

# colnames(recover) = 1:10
recover_long = recover %>% gather(koverm, recovered,X1:X10)
recover_long$koverm = as.numeric(gsub("X", "", recover_long$koverm))/10
recover_long$recovered = factor(recover_long$recovered)
colnames(recover_long) = c("m_over_n", "k_over_m", "recovered")
recover_long %>% ggplot(aes_string("m_over_n", y="k_over_m", color = "recovered")) + geom_point(size=3)
ggsave("hw1_output.pdf")
