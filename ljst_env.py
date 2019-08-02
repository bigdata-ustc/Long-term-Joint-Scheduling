from numpy import *
from pandas import *
from scipy import stats
class BikeFlow(object):
    def __init__(self, station_num):
        self.graph_flow_list = []
        self.return_demand_list = []
        self.biking_demand_list = []
        for i in range(24):
            self.graph_flow_list.append(zeros((station_num, station_num)))
            self.return_demand_list.append(zeros(station_num))
            self.biking_demand_list.append(zeros(station_num))
    
    def set_demand_flow(self, graph_flow, return_demand, biking_demand, t):
        self.graph_flow_list[t] = graph_flow.copy()
        self.return_demand_list[t] = return_demand.copy()
        self.biking_demand_list[t] = biking_demand.copy()
        # self.update_static_info()

    def get_graph_flow(self, t):
        # 获取预测的时刻t的单车流动图， 是一个矩阵形式
        # print("graph_flow:\n", self.graph_flow_list[t])
        return self.graph_flow_list[t].copy()
        
    def get_return_demand(self, t):
        # 获取预测的时刻t的单车站点归还数目
        # print("return_demand:\n", self.return_demand_list[t])
        return self.return_demand_list[t].copy()
        
    def get_biking_demand(self, t):
        # 获取预测时刻t的单车站点使用数目
        # print("biking_demand:\n", self.biking_demand_list[t])
        return self.biking_demand_list[t].copy()
    
def get_integer_array(float_array):
    a = float_array.reshape(-1)
    idx = arange(size(a))[a>0]
    suma0 = int(sum(a+1e-9))
    a = a.astype(int)
    suma1 = int(sum(a))
    adda = suma0 - suma1
    if adda < 1:
        return a.reshape(float_array.shape)
    idx = random.choice(idx, adda, replace=False)
    a[idx] += 1
    if int(sum(float_array+1e-9)) != int(sum(a)):
        print("get_integer_array not equal")
    return a.reshape(float_array.shape).astype(int)

class BikeFlowFac(object):
    def __init__(self, station_num=10, bikes_num = None, day_mean=500):
        # 单车数据生成工厂
        # 1. station_num指定生成的单车站点数目
        # 2. bikes_num 指定生成单车数目
        # 3. day_mean一天平均的单车使用次数
        #
        self.station_num = station_num
        self.day_mean = day_mean
        self.bikes_num = bikes_num
        self.bike_record = read_csv("./bike_record.csv")
        bike_day_flow = self.bike_record.day.value_counts().values
        bike_day_flow = bike_day_flow/mean(bike_day_flow)
        bike_hour_flow = self.bike_record.hour.values
#         bike_hour_flow = bike_hour_flow/sum(bike_hour_flow)
        self.bike_day_kde = stats.gaussian_kde(bike_day_flow)
        self.bike_hour_kde = stats.gaussian_kde(bike_hour_flow)
        self.day_demand = 0
        
        
    def gen_station_bikes_num(self):
        # 生成一个新的单车分布， 每天7点的初始状态
        bikes_num = random.randint(8,20)*self.station_num if self.bikes_num is None else self.bikes_num
        station_bikes_num = get_integer_array(self.get_p(p_type=0)*bikes_num)
        return station_bikes_num

    def gen_station_docks_num(self):
        # 生成一个随机的每个车站的最大可容量单车数目， 在15~30之间
        station_docks_num = random.randint(10, 30, self.station_num)
        return station_docks_num
        
    def get_day_flow(self):
#         return self.bike_day_kde.resample(1)
        return max(self.bike_day_kde.resample(1), 0.2)
    
    def get_hour_flow(self):
        # 根据百度地图查询记录，生成一个新的一天不同时刻的单车使用变化情况
        hour_demand = zeros(24)
        record = self.bike_hour_kde.resample(1000)
        record = get_integer_array(record)
        record = record[record>=7]
        record = record[record<24]
        record_value_counts = Series(record).value_counts()
        hour_demand[record_value_counts.index] = record_value_counts.values
        hour_demand = hour_demand/sum(hour_demand)
        return hour_demand
    
    def get_p(self, p_type=None):
        # 生成一个随机概率数组， 总和为1
        # 为了保证多样性，有三种不同风格
        #
        p = random.rand(self.station_num)
        # rand change p
        if p_type is None:
            p_type = random.randint(1e9+7)%4
        if p_type == 1:
            p = p*p
        elif p_type == 2:
            p = p
        elif p_type == 3:
            p = sqrt(p)
        p = p/sum(p)
        return p
    
    def sample_from_p(self, num, p, zero_idx=-1):
        # 生成一个数组， 总和等于num，数组每个值ans[i]分配的期望是p[i]*num, 这里保证ans[i]是整数
        # zero_idx 表示这个数组第zero_idx强制为0，因为每个车站往其他车站流动的时候不会往自己车站流动
        pp = p.copy()
        sz = size(p)
        ans = zeros(sz)
        if zero_idx >= 0:
            pp[zero_idx] = 0
            pp = pp/sum(pp)
        record = random.choice(sz, num, p=p)
        record_value_counts = Series(record).value_counts()
        ans[record_value_counts.index] = record_value_counts.values
        ans = get_integer_array(ans)
        return ans      

    def get_bike_flow(self):
        # 生成一个新的一天类的单车流动， 返回BikeFlow类
        # 生产思路：
        # 1. 先生成一天总的单车流量， 这个过程有随机性， 随机性和百度数据中相似
        # 2. 根据百度地图的查询记录， 生成一天每小时单车使用变化情况， 总和为1生成的值
        # 3. 有了每小时的总的单车使用次数， 然后随机分配到每个车站，[7, 11, 15, 19, 23]这些时刻是随机生成，其他时刻是插值法
        #
        bike_flow = BikeFlow(self.station_num)
#         all_bike_demand = random.randint(500,1000) if self.day_mean is None else self.day_mean
#         day_demand = min(int(self.get_day_flow()*all_bike_demand), 1000)
        all_bike_demand = self.day_mean
        self.day_demand = min(max(int(self.get_day_flow()*all_bike_demand), 200),1000)
        hour_demand = get_integer_array(self.get_hour_flow()*self.day_demand)
        gen_hour_list = [7, 11, 15, 19, 23, 24]
        # gen_hour_list = [7, 11,12]
        gen_biking_p = [self.get_p() for i in gen_hour_list]
        gen_return_p = [self.get_p() for i in gen_hour_list]
        last = -1
        left = 0
        for t in range(7, 24):
            if t in gen_hour_list:
                last = last + 1
            biking_demand_p = ((4-left)*gen_biking_p[last]+left*gen_biking_p[last+1])/4.0
            biking_demand = self.sample_from_p(hour_demand[t], biking_demand_p)
            return_demand_p = ((4-left)*gen_return_p[last]+left*gen_return_p[last+1])/4.0
            graph_flow = zeros((self.station_num, self.station_num))
            for i in range(self.station_num):
                graph_flow[i] = self.sample_from_p(biking_demand[i], return_demand_p, i)
            return_demand = sum(graph_flow, axis=0)
            biking_demand = sum(graph_flow, axis=1)
            bike_flow.set_demand_flow(graph_flow, return_demand, biking_demand, t)
            left = (left+1)%4
        return bike_flow

class BikeGame(object):
    def __init__(self, station_num=8, bikes_num=None, day_mean= 500, max_scheduling_num=10):
        # 复杂模拟环境, 提供和环境交互的接口
        # 一天默认时间是7~23点
        #
        # 1. station_num 车站这里默认数目是8
        # 2. bikes_num 总的单车数目默认是随机的，可以指定，随机的小车数目是车站数目8~20倍
        # 3. day_mean 制定一天平均总的单车流量， 但是制定之后，会有一点随机性
        # 4. max_scheduling_num 最大可调度单车数目
        self.station_num = station_num
        self.fac = BikeFlowFac(station_num, bikes_num, day_mean)
        self.max_scheduling_num = max_scheduling_num
        self.new_game()
    
    def get_graph_flow(self, t=None):
        # t is None默认获取当前时段的单车流动估计图
        # t < 0, 表示未来-t时间段的单车流动估计图
        if t is None:
            t = self.time
        elif t < 0:
            t = int(self.time - t)
        # 获取预测的时刻t的单车流动图， 是一个矩阵形式
        if t >= 24:
            return zeros((self.station_num, self.station_num))
        return self.bike_flow.get_graph_flow(t).copy()

    def get_day_demand(self):
        return self.fac.day_demand
        
    def get_return_demand(self, t = None):
        # t is None默认获取当前时段的单车归还需求
        # t < 0, 表示未来-t时间段的单车归还需求
        if t is None:
            t = self.time
        elif t < 0:
            t = int(self.time - t)
        if t >= 24 or t < 0:
            return zeros(self.station_num)
        # 获取预测的时刻t的单车站点归还数目
        return self.bike_flow.get_return_demand(t).copy()
        
    def get_biking_demand(self, t = None):
        # t is None默认获取当前时段的单车使用需求
        # t < 0, 表示未来-t时间段的单车使用需求
        if t is None:
            t = self.time
        elif t < 0:
            t = int(self.time - t)
        if t >= 24 or t < 0:
            return zeros(self.station_num)
        # 获取预测时刻t的单车站点使用数目
        return self.bike_flow.get_biking_demand(t).copy()

    def get_station_bikes_num(self):
        return self.station_bikes_num.copy()

    def get_station_docks_num(self):
        return self.station_docks_num.copy()

    def get_award(self):
        return self.award.copy()

    def new_game(self):
        # 初始化交互环境， 生产新的一轮数据
        self.award = 0
        self.docks_loss = 0
        self.time = 7
        self.bike_flow = self.fac.get_bike_flow()
        self.station_bikes_num = self.fac.gen_station_bikes_num()
        self.station_docks_num = self.fac.gen_station_docks_num()
        self._init_station_bikes_num = self.station_bikes_num.copy()
        self._init_station_docks_num = self.station_docks_num.copy()
    
    def reset_game(self):
        self.award = 0
        self.docks_loss = 0
        self.time = 7
        self.station_bikes_num = self._init_station_bikes_num.copy()
        self.station_docks_num = self._init_station_docks_num.copy()

    def do_bike_flow(self, scheduling_award = 0):
        # 实际的模拟单车流程程序， 执行这一时刻的单车流动， 然后时间+1
        bikes_use_num = minimum(self.bike_flow.biking_demand_list[self.time], self.station_bikes_num)
        self.award += scheduling_award
        self.award += sum(bikes_use_num)
        self.station_bikes_num = self.station_bikes_num - bikes_use_num
        graph_flow = self.bike_flow.graph_flow_list[self.time]
        bikes_return = zeros(self.station_num).astype(int)
        for i in range(self.station_num):
            if sum(graph_flow[i]) == 0:
                continue
            tmp = get_integer_array(graph_flow[i]/sum(graph_flow[i])*bikes_use_num[i])
            if int(sum(tmp)) != int(bikes_use_num[i]):
                print("not equal after random rebalance")
            bikes_return += tmp.astype(int)
        if int(sum(bikes_return+1e-9)) != int(sum(bikes_use_num+1e-9)):
            print("----------------------------------------")
            print("sum(bikes_return) != sum(bikes_use_num)")
            print("----------------------------------------")
        self.station_bikes_num += bikes_return
        idx = self.station_bikes_num>self.station_docks_num
        docks_loss = sum(self.station_bikes_num[idx]-self.station_docks_num[idx])
        #self.award -= docks_loss*0.25
        self.docks_loss += docks_loss*0.25
        self.time += 1
        return self.award

    def get_game_state(self):
        # 返回当前调度车观察的状态
        # 1. 它包含对未来2个时刻的预测骑行和归还需求
        # 2. 包含当前每个车站可用单车数目， 和可停放单车数目
        next_graph_flow = self.get_graph_flow(self.time).reshape(-1)
        nxt_next_graph_flow = self.get_graph_flow(self.time+1).reshape(-1)
        current_biking_demand = self.get_biking_demand(self.time)
        current_return_demand = self.get_return_demand(self.time)
        nxt_biking_demand = self.get_biking_demand(self.time+1)
        nxt_return_demand = self.get_return_demand(self.time+1)
        game_state = concatenate([next_graph_flow, nxt_next_graph_flow, \
            current_biking_demand, current_return_demand, self.station_bikes_num, \
                self.station_docks_num, nxt_biking_demand, nxt_return_demand])
        game_state =  expand_dims(game_state,axis=0)
        return game_state.reshape(16,11)
    
    def do_action(self, st, ed, num):
        # 和环境交互接口，表示调度小车，每个时刻默认调度一次， 合法操作返回0
        # 调度小车之后， 这里模拟执行这一时刻的单车流动，并进行到下一时刻
        # 
        # 1. st表示从哪个车站调度小车， ed表示将小车调度到哪里， num表示调度小车的数目
        # 2. num <= 0, st<0或者ed<0的时候 -> 当前调度车不执行操作
        # 3. 规定st != ed
        # 4. 当前调度的车数目必须小于等于该车站的单车数目
        num = int(max(min(num, self.max_scheduling_num), 0))
        if self.time >= 24:
            return self.award
        if num <= 0 or st < 0 or ed < 0 or (st == ed):
            return self.do_bike_flow(0)
        elif self.station_bikes_num[st] >= num:
        # print("st", st, " ed", ed, " num", num)
            self.station_bikes_num[st] -= num
            self.station_bikes_num[ed] += num
            return self.do_bike_flow(-5)
        else:
            return self.do_bike_flow(0)
        
class Env(object):
    def __init__(self,station_num = 8,
                 all_bikes_num = None,
                 day_mean = 500,
                 max_scheduling_num = 10):
        self.bikeGame = BikeGame(station_num, all_bikes_num, day_mean, max_scheduling_num)
        self.station_num = station_num
        self.max_scheduling_num = max_scheduling_num
        
    def reset(self):
        self.bikeGame.new_game()
        s0 = self.bikeGame.get_game_state()
        return s0
    
    def step(self,action):
        isOver = False
        info = None
        st = np.argmax(action[0:8])
        ed = np.argmax(action[8:16])
        num = int(action[16]) * 10
        reward = self.bikeGame.do_action(st,ed,num)
        state = self.bikeGame.get_game_state()
        if self.bikeGame.time >= 24:
            isOver = True
        return state,reward,isOver,info
