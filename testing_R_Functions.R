
library(DataCombine)

neuron_pair <- read.csv("C:\\Users\\john\\Desktop\\hy590-21\\data\\pair_1_1105.csv")

A <- neuron_pair$V2
### 1 -> shift left by 1
for (c in 1:10){
  A <- (A + shift(A,1,0))
}

ifelse(neuron_pair$V2&A,1,0)