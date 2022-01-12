library(data.table)
library(dplyr)

setwd("/Users/an-eunseog/Desktop/paper/data")

# edge count
data10 = fread('data10.csv')
edge1 = unique(data10$edge)

Edge_num_mat_month4 = matrix(0, nrow = length(unique(data10$edge)), ncol = 720)

data10_month4 = data10 %>% filter(month == 4)

for (i in 1:length(unique(data10$edge))) {
  for (j in 0:29) {
    for (v in 1:24) {
      a = data10_month4 %>% filter(edge == edge1[i], day == j+1, hour == v-1)
      Edge_num_mat_month4[i, v+(j*24)] = ifelse(length(a$n) == 0, 0, a$n)
    }
  }
}

write.csv(Edge_num_mat_month4, file = "/Users/an-eunseog/Desktop/paper/data/Edge_num_mat_month4.csv", row.names = F)

Edge_num_mat_month5 = matrix(0, nrow = length(unique(data10$edge)), ncol = 720)

data10_month5 = data10 %>% filter(month == 5)

for (i in 1:length(unique(data10$edge))) {
  for (j in 0:29) {
    for (v in 1:24) {
      a = data10_month5 %>% filter(edge == edge1[i], day == j+1, hour == v-1)
      Edge_num_mat_month5[i, v+(j*24)] = ifelse(length(a$n) == 0, 0, a$n)
    }
  }
}

write.csv(Edge_num_mat_month5, file = "/Users/an-eunseog/Desktop/paper/data/Edge_num_mat_month5.csv", row.names = F)


Edge_num_mat = cbind(Edge_num_mat_month4, Edge_num_mat_month5)
Edge_num_mat %>% dim() # 3147 1440

write.csv(Edge_num_mat, file = "/Users/an-eunseog/Desktop/paper/data/Edge_num_mat.csv", row.names = F)


# adj mat
edge_split <- data.frame(do.call('rbind', strsplit(as.character(edge1), split = ",", fixed = TRUE))) 

edge_split

rent = data.frame(do.call('rbind', strsplit(as.character(edge_split$X1), split = "(", fixed = TRUE)))
return = data.frame(do.call('rbind', strsplit(as.character(edge_split$X2), split = ")", fixed = TRUE)))

rent_return = data.frame(rent_place = rent$X2, return_place = return[[1]])

unique(c(rent_return$rent_place, rent_return$return_place)) %>% length() # 1186
rent1 = data.frame(rent_place = unique(c(rent_return$rent_place, rent_return$return_place)), rent = 1:1186)
return1 = data.frame(return_place = unique(c(rent_return$rent_place, rent_return$return_place)), return = 1:1186)

rent_return = left_join(rent_return, rent1, by = "rent_place")
rent_return = left_join(rent_return, return1, by = "return_place")

write.csv(rent_return, file = "/Users/an-eunseog/Desktop/paper/data/rent_return.csv", row.names = F)

A_mat = matrix(0, 3147, 3147)

for (i in 1:3147) {
  for (j in 1:3147) {
    a = rent_return[i,3]
    b = rent_return[i,4]
    c = c(a,b)
    if(rent_return[j,3] %in% c | rent_return[j,4] %in% c){
      A_mat[i,j] = 1
    }else{
      A_mat[i,j] = 0
    }
  }
}

diag(A_mat) <- c(0)
a = apply(A_mat, 1, sum)
b = apply(A_mat, 2, sum)
sum(a!=b)
which(a==0)


# Remove isolated edges
ind = which(a==0)

Edge_num_mat1 = Edge_num_mat[-ind,]
Edge_num_mat1 %>% dim

write.csv(Edge_num_mat1, file = "/Users/an-eunseog/Desktop/paper/data/Edge_num_mat1.csv", row.names = F)


rent_return1 = rent_return[-ind,]
rent_return1 %>% dim

write.csv(rent_return1, file = "/Users/an-eunseog/Desktop/paper/data/rent_return1.csv", row.names = F)

A_mat1 = matrix(0, 3135, 3135)

for (i in 1:3135) {
  for (j in 1:3135) {
    a = rent_return1[i,3]
    b = rent_return1[i,4]
    c = c(a,b)
    if(rent_return1[j,3] %in% c | rent_return1[j,4] %in% c){
      A_mat1[i,j] = 1
    }else{
      A_mat1[i,j] = 0
    }
  }
}

diag(A_mat1) <- c(0)
a = apply(A_mat1, 1, sum)
b = apply(A_mat1, 2, sum)
sum(a!=b)
which(a==0)


edge_ind <- which(A_mat1 != 0, arr.ind = T)
edge_ind <- as.data.frame(edge_ind)
edge_ind1 <- edge_ind %>% mutate(row = edge_ind$row - 1, col = edge_ind$col - 1)
edge_ind1 %>% dim()

write.csv(edge_ind1, file = "/Users/an-eunseog/Desktop/paper/data/edge_ind1.csv", row.names = F)





    