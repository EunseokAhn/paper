# paper

## Prediction of bike-sharing system traffic through neural network models that capture spatiotemporal dependence

Accurate prediction of the amount of traffic between rental stations in a bike-sharing system is very important in preventing bicycle imbalance problems and bicycle accidents. However, the bike-sharing system network has complex spatial dependencies. Therefore, in this paper, the graph structure is used to maintain the complex spatial dependence structure and the traffic volume prediction problem is solved using the line graph. Additionally,in order to reflect the time dependence that differs depending on the time period, we propose a temporal line graph convolutional network(T-LGCN), a model that uses time-dependent traffic as a feature matrix and captures the spatio-temporal dependence at the same time. At this time, the Line Graph Convolutional Network(LGCN) is used to capture spatial dependence.
Gated Recurrent Unit(GRU)[Zhao et al., 2019] is used for timedependent capture. This paper compares the performance with reference models using real data and shows that the proposed model has better performance
