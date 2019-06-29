# Long-term-Joint-Scheduling

Long-term Joint Scheduling for Urban Traffic.
Here we use implement a simple demo by PaddlePaddle, using reinforcement learning to schedule bike-sharing.

## KDD CUP VIDEO
This is the [link](https://youtu.be/t5M2wVPhTyk) to video

## 模拟数据生成文档
### 假设
我们基于以下假设生成数据
1.	假设每个车站在每天不同时刻流量是不同的, 分布如下图所示

![pic](https://zeroxf.oss-cn-shanghai.aliyuncs.com/pic.png)

(注明: 此分布是从百度单车查询记录生成)
2.	不同车站的单车流动是随时间变化的

### 生成过程
- 随机生成station_num个单车车站, 和bikes_num个总的单车数目
- 每个车站初始拥有的单车数目是随机分配的
- 根据百度地图每天的单车使用量, 来随机生成一天内单车的总需求量, 然后再根据假设1的分布生成一天每个时刻的单车总需求.
在时刻t,将总单车使用量随机分配到每个车站, 同时随机生成每个车站单车的归还量. 站点之间的客流量按照每个站点的使用需求和归还需求按相应的比例生成（具体细节见代码）

## Data Simulation
### Assumptions
First we generate the data based on the following assumptions:
- Each station has different flow rates at different moments of the day, as shown in the following figure.

![pic](https://zeroxf.oss-cn-shanghai.aliyuncs.com/pic.png)

(Note: This distribution is generated from Baidu bicycle query records)
- Bicycle flow at different stations will be changed over times.

### Steps:
- Randomly generate the number of bikes and bike-sharing stations.
- We randomly initialize the number of bikes at each station.
- According to Baidu map query records, we randomly generate the total demand for bicycles during the day. Then we generate the total demand for bicycles at each moment of the day according to the distribution of hypothesis 1.
- At time t, the total bicycle usage is randomly assigned to each station, and the return amount of each station bicycle is randomly generated. The passenger flow between stations is generated from the demands of using and returning bikes according to the corresponding proportion (see the code for details)

