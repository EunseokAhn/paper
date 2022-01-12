library(data.table)
library(dplyr)
library(ggplot2)
library(tidyverse)
library(lubridate)
library(timetk)
library(tidyquant)
library(TTR)
library(zoo)
setwd("/Users/an-eunseog/Desktop/bike_paper/data")

edge_num_mat = fread('Edge_num_mat2.csv')
rent_return = fread('rent_return2.csv')
edge_ind = fread('edge_ind2.csv')
pred = fread('pred.csv')


seq(from=as.POSIXct("2019-05-13 07:00:00", tz="UTC"), 
    to=as.POSIXct("2019-05-30 23:00:00", tz="UTC"), by="hour") %>% length()

date = data.frame(date = seq(
  from=as.POSIXct("2019-5-13 7:00", tz="UTC"),
  to=as.POSIXct("2019-5-30 23:00", tz="UTC"),
  by="hour"
))  


pred_mat = matrix(pred$V1[2:109226],257,425)
edge_num_mat1 = edge_num_mat[,1016:1440]

rolling_avg_num = zoo::rollmean(as.numeric(edge_num_mat1[ind,]), k=4, fill = NA)

ind = 22

data = tibble(Time = c(date$date, date$date), 
              Traffic = c(pred_mat[ind,], rolling_avg_num),
              Value = c(rep('Pred', 425), rep('True', 425)))

data %>% ggplot(aes(x = Time, y = Traffic, colour = Value)) +
  geom_line() +
  theme_bw() +
  theme(legend.title = element_blank()) +
  theme(legend.text = element_text(colour="black", size=10, face="bold")) +
  scale_color_manual(values = c("#2C3E50", "#E31A1C")) -> pred_all_day_22


data1 = tibble(Time = c(date$date[2:241], date$date[2:241]), 
               Traffic = c(pred_mat[ind,2:241], rolling_avg_num[2:241]),
               Value = c(rep('Pred', 240), rep('True', 240)))

data1 %>% ggplot(aes(x = Time, y = Traffic, colour = Value)) +
   geom_line() +
   theme_bw() +
   theme(legend.title = element_blank()) +
   theme(legend.text = element_text(colour="black", size=10, face="bold")) +
   scale_color_manual(values = c("#2C3E50", "#E31A1C")) -> pred_ten_day_22


data2 = tibble(Time = c(date$date[2:25], date$date[2:25]), 
               Traffic = c(pred_mat[ind,2:25], rolling_avg_num[2:25]),
               Value = c(rep('Pred', 24), rep('True', 24)))

data2 %>% ggplot(aes(x = Time, y = Traffic, colour = Value)) +
  geom_line() +
  theme_bw() +
  theme(legend.title = element_blank()) +
  theme(legend.text = element_text(colour="black", size=10, face="bold")) +
  scale_color_manual(values = c("#2C3E50", "#E31A1C")) -> pred_one_day_22

















pred_mat[ind,] %>% length()

pred_mat1 = as.data.frame(pred_mat)



data = tibble(date = date$date, pred = pred_mat[ind,], 
              label = zoo::rollmean(as.numeric(edge_num_mat1[ind,]), k=4, fill = NA))

data1 = data[2:241,]



data %>% ggplot() +
  geom_line(aes(x = date, y = label), color = palette_light()[[1]]) + 
  #geom_point(aes(x = date, y = label), color = palette_light()[[1]])

  geom_line(aes(x = date, y = pred), color = palette_light()[[2]]) 
  #geom_point(aes(x = date, y = pred), color = palette_light()[[2]])


data1 %>% ggplot() +
  geom_line(aes(x = date, y = label), color = palette_light()[[1]]) + 
  #geom_point(aes(x = date, y = label), color = palette_light()[[1]])
  
  geom_line(aes(x = date, y = pred), color = palette_light()[[2]]) 
#geom_point(aes(x = date, y = pred), color = palette_light()[[2]])
  

#______________________


