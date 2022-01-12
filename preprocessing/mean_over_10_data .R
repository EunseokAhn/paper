library(data.table)
library(dplyr)

setwd("/Users/an-eunseog/Desktop/bike_paper/data")

# over mean 10

edge_num_mat = fread('Edge_num_mat1.csv')
rent_return = fread("rent_return1.csv")

sum(apply(edge_num_mat, 1, sum)>=600)

edgd_num_mat1 = edge_num_mat[apply(edge_num_mat, 1, sum)>=600,]
rent_return1 = rent_return[apply(edge_num_mat, 1, sum)>=600,]

unique(c(rent_return1$rent_place, rent_return1$return_place)) %>% length() # 339
rent1 = data.frame(rent_place = unique(c(rent_return1$rent_place, rent_return1$return_place)), rent = 1:339)
return1 = data.frame(return_place = unique(c(rent_return1$rent_place, rent_return1$return_place)), return = 1:339)

rent_return2 = rent_return1 %>% dplyr::select(-rent, -return)

rent_return2 = left_join(rent_return2, rent1, by = "rent_place")
rent_return2 = left_join(rent_return2, return1, by = "return_place")


A_mat = matrix(0, 294, 294)

for (i in 1:294) {
  for (j in 1:294) {
    a = rent_return2[i,3]
    b = rent_return2[i,4]
    c = c(a,b)
    if(rent_return2[j,3] %in% c | rent_return2[j,4] %in% c){
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
which(a==0) %>% length()


ind = which(a==0) 



edgd_num_mat2 = edgd_num_mat1[-ind,]
rent_return3 = rent_return2[-ind,]

edgd_num_mat2 %>% dim()
rent_return3 %>% dim()

unique(c(rent_return3$rent_place, rent_return3$return_place)) %>% length()

rent2 = data.frame(rent_place = unique(c(rent_return3$rent_place, rent_return3$return_place)), rent = 1:265)
return2 = data.frame(return_place = unique(c(rent_return3$rent_place, rent_return3$return_place)), return = 1:265)

rent_return4 = rent_return3 %>% dplyr::select(-rent, -return)

rent_return4 = left_join(rent_return4, rent2, by = "rent_place")
rent_return4 = left_join(rent_return4, return2, by = "return_place")


A_mat = matrix(0, 257, 257)

for (i in 1:257) {
  for (j in 1:257) {
    a = rent_return4[i,3]
    b = rent_return4[i,4]
    c = c(a,b)
    if(rent_return4[j,3] %in% c | rent_return4[j,4] %in% c){
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
which(a==0) %>% length()


edge_ind <- which(A_mat != 0, arr.ind = T)
edge_ind <- as.data.frame(edge_ind)
edge_ind1 <- edge_ind %>% mutate(row = edge_ind$row - 1, col = edge_ind$col - 1)
edge_ind1 %>% dim()


c(edge_ind1$row, edge_ind1$col) %>% max()

write.csv(edgd_num_mat2, file = "Edge_num_mat2.csv", row.names = F)
write.csv(rent_return4, file = "rent_return2.csv", row.names = F)
write.csv(A_mat, file = "Adj_mat2.csv", row.names = F)
write.csv(edge_ind1, file = "edge_ind2.csv", row.names = F)

