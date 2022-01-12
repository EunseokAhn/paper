library(data.table)
library(dplyr)

rm(list = ls(all=TRUE))

setwd("/Users/an-eunseog/Desktop/bike_paper/data")

# change to one-way path________________________________________________

data = fread('dat_7.csv')

data1 <- data %>% dplyr::select(rent_place, return_place)

unique_rent_return1 <- matrix(0 ,nrow = nrow(data1), ncol = 2)
unique_rent_return2 <- as.matrix(data1)

unique_rent_return3 <- matrix(0 ,nrow = nrow(data1), ncol = 2)
unique_rent_return4 <- as.matrix(data1)

station_1 = rep(NA, nrow(data1))
station_2 = rep(NA, nrow(data1))

# get the numeric vector of stations
data2_mat = as.matrix(data1)
stns = unique(c(data2_mat[,1], data2_mat[,2]))
stn_prenum = unlist(strsplit(stns, "ST-"))[(1:length(stns))*2]
stn_numeric = as.numeric(stn_prenum)
# get the numeric vector of origins
or_stns = as.numeric(unlist(strsplit(data2_mat[,1], "ST-"))[(1:nrow(data2_mat))*2])
# get the numeric vector of destinations
de_stns = as.numeric(unlist(strsplit(data2_mat[,2], "ST-"))[(1:nrow(data2_mat))*2])

index_vec = rep(NA, nrow(data1))

# the new station pairs with order ignored
new_stn_pair = matrix(NA, ncol=2, nrow=nrow(data1))

N_stns = length(stns)

for (i in 1:(N_stns-1)){
  or_i_ind = or_stns == stn_numeric[i] 
  de_i_ind = de_stns == stn_numeric[i] 
  ith_ind = or_i_ind | de_i_ind
  for (j in (i+1):N_stns){
    or_j_ind = or_stns == stn_numeric[j] 
    de_j_ind = de_stns == stn_numeric[j] 
    jth_ind = or_j_ind | de_j_ind
    
    ijth_ind = ith_ind & jth_ind
    
    hit_count = sum(ijth_ind)
    
    if (hit_count>0){
      large_stn = max(stn_numeric[i],stn_numeric[j])
      small_stn = min(stn_numeric[i],stn_numeric[j])
      
      stn_pair = paste0("ST-",c(small_stn,large_stn))
      
      new_stn_pair[ijth_ind,] = rep(stn_pair, c(hit_count, hit_count))
    }
  }
  print(i)
}

write.csv(new_stn_pair, file = "/Users/an-eunseog/Desktop/paper/data/new_stn_pair.csv", row.names = F)

#Hourly data generation________________________________________________

colnames(new_stn_pair) = c('rent_place', 'return_place')

data2 = data %>% dplyr::select(-rent_place, -return_place)
data3 = cbind(data2, new_stn_pair)
data4 = data3 %>% dplyr::select(rent_place, return_place, month, day, hour)

# delete self-roof
data5 = na.omit(data4)

data6 = data5 %>% mutate(edge = paste0("(",data5$rent_place, ",",data5$return_place,")" ))
data7 = data6 %>% group_by(edge, month, day, hour) %>% summarise(n=n())

# over3 mean
data8 = data6 %>% group_by(edge) %>% summarise(n=n())
data9 = data8 %>% filter(n>=180)
data10 = data7 %>% filter(edge %in% data9$edge)

write.csv(data10, file = "/Users/an-eunseog/Desktop/paper/data/data10.csv", row.names = F)





