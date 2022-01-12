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

rent_return %>% filter(rent_place == "ST-99")

edge_num_mat1 = t(edge_num_mat)

date = data.frame(date = seq(
  from=as.POSIXct("2019-4-1 00:00"),
  to=as.POSIXct("2019-5-30 23:00"),
  by="hour"
)) 

ind1 = 109
ind2 = 249

rolling_avg_num_ind1 = zoo::rollmean(as.numeric(edge_num_mat[ind1,]), k=4, fill = NA)
rolling_avg_num_ind2 = zoo::rollmean(as.numeric(edge_num_mat[ind2,]), k=4, fill = NA)
# time________________
data = tibble(Time = c(date$date, date$date), 
              Traffic = c(rolling_avg_num_ind1, rolling_avg_num_ind2),
              Value = c(rep(paste0("(", rent_return[ind1,1],",",rent_return[ind1,2],")"), 1440), 
                        rep(paste0("(", rent_return[ind2,1],",",rent_return[ind2,2],")"), 1440)))

data1 = data %>% filter(as.Date(Time) == c("2019-5-24"))



data1 %>% ggplot(aes(x =  Time, y = Traffic, colour = Value)) +
  geom_line() +
  theme_bw() +
  theme(legend.title = element_blank(),
        legend.text = element_text(colour="black", size=10, face="bold"),
        plot.title = element_text(colour="black", size=15, face="bold", hjust = 0.5)) +
  scale_color_manual(values = c("#2C3E50", "#E31A1C")) +
  ylim(c(0,5)) +
  theme(legend.position = "top",
        legend.text = element_text(colour="black", size=20, face="bold"),
        legend.key.width = unit(5, 'cm'))+
  xlab("")+
  ylab("")



which.max(edge_num_mat[ind1,])
which.max(as.numeric(edge_num_mat[ind1,])  )
as.character(data1$Time)

unique(rent_return$rent_place)

which(rent_return$rent_place=="ST-516")

c('8', '16', '32', '64', '100', '128')
#hyper parameter________
hidden = data.frame(Hidden_unit = 1:6,
                    RMSE = c(1.3128, 1.3704, 1.2844, 1.2783, 1.2686, 1.2802),
                    MAE = c(0.8667, 0.8954, 0.8177, 0.8046, 0.7864, 0.8035))


bias = 0.45

ggplot(hidden, aes(x=Hidden_unit)) +
  geom_line(aes(y=RMSE), color='black') + 
  geom_line(aes(y=MAE + bias), color='red') +
  geom_point(aes(y=RMSE), color='black') +
  geom_point(aes(y=MAE + bias), color='red')+
  geom_vline(aes(xintercept=5), linetype="dotted",col="blue", size = 1)+
  theme_bw()+
  scale_x_continuous(
    name = "Hidden unit",
    breaks = c(1:6),
    labels = c("8","16","32", "64", "100", "128")
  )+
  scale_y_continuous(
    name = "RMSE",
    sec.axis = sec_axis(~.-bias, name="MAE")
  ) + 
  
  theme(
    axis.title.y = element_text(color = 'black', size=13),
    axis.title.y.right = element_text(color = 'red', size=13),
    axis.title.x = element_text(color = 'black', size=13)
  ) 


  
epoch = data.frame(RMSE = c(1.2686, 1.2629, 1.2608, 1.2597, 1.2592, 1.2589, 1.2586, 1.2583),
                   MAE = c(0.7864, 0.7786, 0.7754, 0.7734, 0.7751, 0.7726, 0.7734, 0.7773),
                   EPOCH = 1:8)

bias = 0.485

ggplot(epoch, aes(x=EPOCH)) +
  geom_line(aes(y=RMSE), color='black') + 
  geom_line(aes(y=MAE + bias), color='red') +
  geom_point(aes(y=RMSE), color='black') +
  geom_point(aes(y=MAE + bias), color='red')+
  geom_vline(aes(xintercept=7), linetype="dotted",col="blue", size = 1)+
  theme_bw()+
  scale_x_continuous(
    name = "Epoch",
    breaks = c(1:8),
    labels = c("100","200","300", "400", "500", "600", "700", "800")
  )+
  scale_y_continuous(
    name = "RMSE",
    sec.axis = sec_axis(~.-bias, name="MAE")
  ) + 
  
  theme(
    axis.title.y = element_text(color = 'black', size=13),
    axis.title.y.right = element_text(color = 'red', size=13),
    axis.title.x = element_text(color = 'black', size=13)
  ) 
