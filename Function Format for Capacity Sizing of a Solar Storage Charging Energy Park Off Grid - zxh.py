import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import importlib.util
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.stats import lognorm, uniform, norm
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Sequence, Dict, Any
# ========== 全局变量定义区 ==========
# 用户输入参数
xl_user = np.array([900, 900, 1500, 1500, 100, 100])  # 下限（单位：kW 或 kWh）
xu_user = np.array([1200, 1200, 4000, 4000, 400, 400])  # 上限（单位：kW 或 kWh）
park_space = 160000
car_number = 140
# 时间步长
Delta_t = 1  # 小时
# 电价和碳排放因子
electricity_buy_price = 0.68  # 元/kWh
sell_factor = 0.8  # 卖电系数
electricity_sell_price = 0.68 * sell_factor # 元/kWh
charge_price = 1.05  # 元/kWh
grid_co2_factor = 0.6  # kg/kWh
# 储能充放电效率
eb_charge_eff = 0.9
eb_discharge_eff = 0.9
# 各模块单价
AC_PV_system_unit_price = 4500
DC_PV_system_unit_price = 3700
#光伏系统（AC和DC光伏）单价
AC_Energy_storage_system_unit_price = 2000
DC_Energy_storage_system_unit_price = 500
#储能系统（AC和DC电池）单价 
AC_EV_charger_unit_price = 430 # 7kW约需要3k 
DC_EV_charger_unit_price = 420 # 120kW约需要5w
#充电桩（AC和DC充电桩）单价 
#年运维成本的比例
annual_maintenance_rate=0.01
# 定义储能最大充/放电功率限制（这里假定为额定容量的1倍，你可根据实际需求调整比例）
charge_rate_ratio = 0.1
discharge_rate_ratio = 0.1
# 直流侧和交流侧之间的转换效率
DC_to_AC_conversion_efficiency = 0.97
AC_to_DC_conversion_efficiency = 0.97
# 
OG_flag = 0 
GC_flag = 0
Park_Load_Flag = 0
# NEW ▶
pop_size = 100 #种群个数 # 这里可以根据需要调整种群规模
car_flow_number = 1000 #车流量实际值也设为1000
max_dc_charge_load = 0
car_flow_number_max =  -(-max_dc_charge_load // 120) #同时充电车辆数量峰值
solar_irradiance = np.array([]) #光照一天（DNI）的值
dc_charge_load_profile_raw = np.array([]) #直流充电负荷功率数据（每0.01h一个点）
solar_irrad_profile = np.array([]) #光照一天的标幺值
dc_charge_load_profile_year = np.array([]) #一年的直流充电桩负荷数据
dc_charge_load_profile = np.array([]) #一天的直流充电桩负荷数据
max_dc_charge_load = 0 # 最大直流充电负荷
dc_park_load_profile = np.array([]) # 一天的直流园区负荷数据
ac_charge_load_profile_year = np.array([]) # 一年的交流充电桩负荷数据
ac_charge_load_profile = np.array([]) #一天的交流充电桩负荷数据
ac_park_load_profile = np.array([]) # 一天的交流园区负荷数据
dc_load_total = np.array([])  #定义的是一天的量
ac_load_total = np.array([])  #定义的是一天的量
load_total = np.array([])  #定义的是一天的量
# 结果变量
res = None
res_grid = None
n_hours = 0
n_days = 365  # 一年365天
# 最终选定方案参数
pv_ac_selected = 0
pv_dc_selected = 0
eb_ac_selected = 0
eb_dc_selected = 0
charger_ac_selected = 0
charger_dc_selected = 0

class MicrogridOptimization_1(Problem): #定义一个新的类 MicrogridOptimization，继承自 pymoo.core.problem.Problem

    def __init__(self):
        super().__init__(n_var=6, n_obj=3, n_constr=0, xl=xl_user, xu=xu_user)
        #离网条件n_obj=3
        #决策变量6个 优化目标3个 没有显式约束条件 决策变量下限数组和上限数组，请注意决策变量这12个值由用户输入
    def _evaluate(self, X, out, *args, **kwargs):  
        #_evaluate为父类 Problem 中定义的“抽象方法”
        #如果你的种群规模是100，决策变量是6个，那么：X.shape = (100, 6)
        objs = []
        dc_ac_power_flow = []  #用来记录每个时刻直流与交流之间的功率流动，后续可以进行记录，绘制曲线
        #创建一个空列表 objs，用于存储每个解对应的4个目标函数值。
        for row in X:  # 全部都是24分析
            pv_ac, pv_dc, eb_ac, eb_dc, charger_ac, charger_dc = row 
            max_ac_charge_power = eb_ac * charge_rate_ratio
            max_dc_charge_power = eb_dc * charge_rate_ratio
            max_ac_discharge_power = eb_ac * discharge_rate_ratio
            max_dc_discharge_power = eb_dc * discharge_rate_ratio
            pv_power_ac = pv_ac * solar_irrad_profile   #pv_ac kW irrad_profile 无量纲 pv_power_ac kW
            pv_power_dc = pv_dc * solar_irrad_profile   
            #额定容量 × 归一化光照强度
            # NEW ▶
            ac_load_total = ac_charge_load_profile+ac_park_load_profile   #定义的是一天的量
            if charger_dc < car_flow_number_max:
                dc_load_total = charger_dc/car_flow_number_max * dc_charge_load_profile+dc_park_load_profile   #基于实际的充电桩数量进行修改
            else:
                dc_load_total = dc_charge_load_profile+dc_park_load_profile   #定义的是一天的量
            # load_total = ac_load_total+dc_load_total   #定义的是一天的量
            #总的负荷功率需求，直接取之前处理好的逐小时负荷
            soc_ac = eb_ac / 2
            soc_dc = eb_dc / 2
            #假设一开始电池电量为额定容量的50% 
            grid_energy = 0
            pv_wasted_ac = 0
            pv_wasted_dc = 0
            pv_wasted = 0
            pv_curtailment = 0
            total_pv_generated = 0
            total_pv_used = 0
            missing_energy = 0  
            #各类累积量的初始化（电网购电量、光伏浪费量、光伏总发电量、光伏实际利用量、供电缺口累计量）
            for t in range(24):
                
                pv_total_ac = pv_power_ac[t]
                pv_total_dc = pv_power_dc[t]
                load_dc = dc_load_total[t]
                load_ac = ac_load_total[t]
                flag_max_dc_discharge_power = max_dc_discharge_power
                flag_max_ac_discharge_power = max_ac_discharge_power
                flag_max_dc_charge_power = max_dc_charge_power
                flag_max_ac_charge_power = max_ac_charge_power
                total_pv_generated += (pv_total_ac + pv_total_dc)

                # 先用直流光伏满足直流负荷
                supply_dc = min(pv_total_dc, load_dc)
                pv_total_dc -= supply_dc
                load_dc -= supply_dc

                # 再用交流光伏满足交流负荷
                supply_ac = min(pv_total_ac, load_ac)
                pv_total_ac -= supply_ac
                load_ac -= supply_ac
                
                total_pv_used += (supply_dc + supply_ac)

                # ========== 一侧富余支援另一侧负载 ==========
                if pv_total_ac > 0 and load_dc > 0:
                    support_dc = min(pv_total_ac*AC_to_DC_conversion_efficiency, load_dc)
                    pv_total_ac -= support_dc/AC_to_DC_conversion_efficiency
                    load_dc -= support_dc
                    total_pv_used += support_dc/AC_to_DC_conversion_efficiency

                if pv_total_dc > 0 and load_ac > 0:
                    support_ac = min(pv_total_dc*DC_to_AC_conversion_efficiency, load_ac)
                    pv_total_dc -= support_ac/DC_to_AC_conversion_efficiency
                    load_ac -= support_ac
                    total_pv_used += support_ac/DC_to_AC_conversion_efficiency

                # ========== 电池放电供负荷 ==========
                if load_dc > 0:
                    if soc_dc >  0.1 * eb_dc:  # Ensure SOC is above 10% before discharging
                        # 增加放电功率上限约束
                        max_release = min(flag_max_dc_discharge_power, (soc_dc-0.1*eb_dc) / Delta_t)
                        release = min(load_dc / eb_discharge_eff, max_release)
                        flag_max_dc_discharge_power -= release
                        soc_dc -= release * Delta_t
                        load_dc -= release * eb_discharge_eff

                if load_ac > 0:
                    if soc_ac > 0.1 * eb_ac:
                        max_release = min(flag_max_ac_discharge_power, (soc_ac-0.1*eb_ac)  / Delta_t)
                        release = min(load_ac / eb_discharge_eff, max_release)
                        flag_max_ac_discharge_power -= release
                        soc_ac -= release * Delta_t
                        load_ac -= release * eb_discharge_eff

                # ========== 电池放电供另一侧负荷 ==========
                if load_dc > 0:
                    if load_dc > 0 and soc_ac > 0.1 * eb_ac:
                        max_release = min(flag_max_ac_discharge_power, (soc_ac-0.1*eb_ac) / Delta_t)
                        release = min(load_dc / eb_discharge_eff, max_release*AC_to_DC_conversion_efficiency)
                        flag_max_ac_discharge_power -= release/AC_to_DC_conversion_efficiency
                        soc_ac -= release/AC_to_DC_conversion_efficiency * Delta_t
                        load_dc -= release * eb_discharge_eff

                if load_ac > 0:
                    if load_ac > 0 and soc_dc > 0.1 * eb_dc:
                        max_release = min(flag_max_dc_discharge_power, (soc_dc-0.1*eb_dc)  / Delta_t)
                        release = min(load_ac / eb_discharge_eff, max_release*DC_to_AC_conversion_efficiency)
                        flag_max_dc_discharge_power -= release/DC_to_AC_conversion_efficiency
                        soc_dc -= release/DC_to_AC_conversion_efficiency * Delta_t
                        load_ac -= release * eb_discharge_eff
                
                # ========== 缺电统计（无电网兜底） ==========
                remaining_deficit = load_dc + load_ac
                if remaining_deficit > 0:
                    missing_energy += remaining_deficit * Delta_t
                
                # ========== 剩余分别充各自储能 ==========
                if pv_total_ac > 0:
                    if soc_ac <  0.9 * eb_ac:
                        max_release = min(flag_max_ac_charge_power, (0.9*eb_ac - soc_ac) / Delta_t)
                        charge_power = min(pv_total_ac, max_release)   # 受限于PV和储能最大充电功率
                        flag_max_ac_charge_power -= charge_power
                        ac_charge = charge_power * eb_charge_eff
                        soc_ac += ac_charge
                        total_pv_used += ac_charge / eb_charge_eff
                        pv_total_ac -= ac_charge / eb_charge_eff

                if pv_total_dc > 0:
                    if soc_dc <  0.9 * eb_dc:
                        max_release = min(max_dc_charge_power, (0.9*eb_dc - soc_dc) / Delta_t)
                        charge_power = min(pv_total_dc, max_release)
                        flag_max_dc_charge_power -= charge_power
                        dc_charge = charge_power * eb_charge_eff
                        soc_dc += dc_charge
                        total_pv_used += dc_charge / eb_charge_eff
                        pv_total_dc -= dc_charge / eb_charge_eff

                # ========== 一侧富余支援另一侧储能 ==========
                if pv_total_dc > 0 and pv_total_ac == 0:  #说明dc侧储能已充满，ac侧储能已充但未满
                    if soc_ac < 0.9 * eb_ac:
                        max_release = min( pv_total_dc * DC_to_AC_conversion_efficiency , flag_max_ac_charge_power)
                        ac_charge = max_release * eb_charge_eff
                        if soc_ac + ac_charge * Delta_t > 0.9*eb_ac:
                            pv_total_dc=pv_total_dc-(0.9*eb_ac-soc_ac)/eb_charge_eff/DC_to_AC_conversion_efficiency
                            total_pv_used +=(0.9*eb_ac-soc_ac)/eb_charge_eff/DC_to_AC_conversion_efficiency
                            soc_ac=0.9*eb_ac
                            flag_max_ac_charge_power -= (0.9*eb_ac-soc_ac)/eb_charge_eff
                        else:
                            soc_ac=soc_ac + ac_charge * Delta_t
                            flag_max_ac_charge_power -= ac_charge/eb_charge_eff
                            total_pv_used +=pv_total_dc
                            pv_total_dc=0
                    else:
                        pv_total_dc=pv_total_dc

                    pv_wasted_dc += pv_total_dc

                if pv_total_ac > 0 and pv_total_dc == 0:  #说明ac侧储能已充满，dc侧储能已充但未满
                    if soc_dc <  0.9* eb_dc:
                        max_release = min (pv_total_ac * AC_to_DC_conversion_efficiency , flag_max_dc_charge_power)
                        dc_charge = eb_charge_eff * max_release
                        if soc_dc + dc_charge * Delta_t >  0.9* eb_dc:
                            pv_total_ac=pv_total_ac-( 0.9* eb_dc-soc_dc)/eb_charge_eff/AC_to_DC_conversion_efficiency
                            total_pv_used +=( 0.9* eb_dc-soc_dc)/eb_charge_eff/AC_to_DC_conversion_efficiency
                            soc_dc= 0.9* eb_dc
                            flag_max_dc_charge_power -= ( 0.9* eb_dc-soc_dc)/eb_charge_eff
                        else:
                            soc_dc=soc_dc + dc_charge * Delta_t
                            flag_max_dc_charge_power -= ac_charge/eb_charge_eff                    
                            total_pv_used +=pv_total_ac
                            pv_total_ac=0
                    else:
                        pv_total_ac=pv_total_ac

                    pv_wasted_ac += pv_total_ac  
                
                # ========== 光伏弃光统计 ==========
                pv_wasted = pv_wasted_ac + pv_wasted_dc
                if pv_wasted > 0:
                    pv_curtailment += pv_wasted * Delta_t    
                # ========== 缺电统计（无电网兜底） ==========
                remaining_deficit = load_dc + load_ac
                if remaining_deficit > 0:
                    missing_energy += remaining_deficit * Delta_t
            #========== 评价指标构建（离网状态） ==========
            investment_cost = AC_EV_charger_unit_price/24 * pv_ac + DC_EV_charger_unit_price/24* pv_dc + 3*AC_Energy_storage_system_unit_price/8 * eb_ac + 3*DC_Energy_storage_system_unit_price/8 * eb_dc + AC_EV_charger_unit_price/24 * charger_ac +DC_EV_charger_unit_price/24* charger_dc
            #总投资成本 = 光伏系统成本 + 储能系统成本 + 充电桩系统成本         
            maintenance_cost = annual_maintenance_rate * investment_cost
            #每年运维成本 = 投资成本的 1%
            revenue = np.sum(dc_charge_load_profile) * charge_price * n_days
            #充电收入 充电服务收费标准 
            total_cost = investment_cost + maintenance_cost - revenue  #这个指标低比较好
            #年化总成本
            pv_utilization = total_pv_used / total_pv_generated if total_pv_generated > 0 else 0
            # New ▶
            if charger_dc < car_flow_number_max:
                missing_energy+=(1-charger_dc/car_flow_number_max) * np.sum(dc_park_load_profile[:24])
            else:
                missing_energy+=0
            lpsp = missing_energy / np.sum(dc_load_total + ac_load_total)

            if pv_utilization <= 1 and lpsp < 0.1:
                objs.append([total_cost, lpsp, 100*(1 - pv_utilization)])
            else:
                objs.append([1e9, 1.0, 1.0])  # 设为劣解，强迫排除

        out["F"] = np.array(objs)

def ceil_to_multiple(value, base):
    """
    返回大于等于 value 的最小 base 的倍数
    例如，value=15，base=7，则返回21,用于修改充电桩容量
    """
    import math
    return math.ceil(value / base) * base


# ============================================
# function1: 预处理输入参数
# ============================================
def preprocess_inputs(typical_day_avg_load, signals=None):
    global xl_user, xu_user, park_space, car_number
    global Delta_t
    global electricity_buy_price, electricity_sell_price, charge_price, grid_co2_factor
    global eb_charge_eff, eb_discharge_eff
    global AC_PV_system_unit_price, DC_PV_system_unit_price
    global AC_Energy_storage_system_unit_price, DC_Energy_storage_system_unit_price
    global AC_EV_charger_unit_price, DC_EV_charger_unit_price
    global annual_maintenance_rate
    global charge_rate_ratio, discharge_rate_ratio
    global DC_to_AC_conversion_efficiency, AC_to_DC_conversion_efficiency
    global solar_irradiance, dc_charge_load_profile_raw
    global OG_flag, GC_flag
    global solar_irrad_profile, dc_charge_load_profile_year, dc_charge_load_profile, dc_park_load_profile
    global ac_charge_load_profile_year, ac_charge_load_profile , ac_park_load_profile
    global res, res_grid, n_hours, n_days
    global pv_ac_selected, pv_dc_selected, eb_ac_selected, eb_dc_selected, charger_ac_selected, charger_dc_selected
    global car_flow_number, max_dc_charge_load, car_flow_number_max
    # 用户输入：6个决策变量上下限 (依次是pv_ac, pv_dc, eb_ac, eb_dc, charger_ac, charger_dc)
    # xl_user = np.array([float(x) for x in input("请输入6个决策变量的下限，用空格分隔(kW kW kWh kWh kW kW): \n参考示例 3000 3000 6000 6000 1000 1000").split()])
    # xu_user = np.array([float(x) for x in input("请输入6个决策变量的上限，用空格分隔(kW kW kWh kWh kW kW): \n参考示例 6000 6000 10000 10000 5000 5000").split()])
    # [pv_ac, pv_dc, eb_ac, eb_dc, charger_ac, charger_dc]
    # 用户输入：园区占地面积  特大型储能园区：临沂市河东区光储充产业园项目，占地面积约240000平方米，总建筑面积约为340000平方米 
    # park_space = float(input("园区占地面积 (平方米): \n参考示例 240000"))
    if xu_user[0]+xu_user[1] > park_space*0.3*175/1000: #后续应该修改 
        xu_user[0]=park_space*0.3*175/2000
        xu_user[1]=park_space*0.3*175/2000  #光伏功率上限（园区限制）

    if xl_user[0]+xl_user[1] > park_space*0.3*175/1000:
        xl_user[0]=park_space*0.3*175/4000
        xl_user[1]=park_space*0.3*175/4000  #光伏功率下限（自定义，可以考虑修改）
        
    # 用户输入：典型日园区负荷均值 (单位：kW) 根据《城市电力规划规范》（GB/T50293-1999），工业用地用电指标为200-800kW/公顷。240000平方米的园区面积为24公顷，按照最低值200kW/公顷计算，负荷功率为4800kW；按照最高值800kW/公顷计算，负荷功率为19200kW。
    # typical_day_avg_load = float(input("请输入典型日园区平均每小时负荷 (kW): \n根据《城市电力规划规范》（GB/T50293-1999），工业用地用电指标为200-800kW/公顷。240000平方米的园区面积为24公顷，按照最低值200kW/公顷计算，负荷功率为4800kW；按照最高值800kW/公顷计算，负荷功率为19200kW\n"))
    # typical_day_avg_load = 800
    # 用户输入：园区车辆保有量（辆）
    # car_number = int(input("请输入园区车辆保有量(辆): \n工业厂房根据标准每100平方米建筑面积配建0.2个车位，对于一个占地面积为240000平方米的园区，假设容积率为1.5，则总建筑面积为360000平方米，车辆保有量约为720个车位\n"))
    num_chargers = car_number // 10
    ac_charger_power = 7  # 单位kW

    if xl_user[5] >  num_chargers*ac_charger_power:
        xl_user[5]=num_chargers*ac_charger_power*6/7

    if xu_user[5] >  num_chargers*ac_charger_power*10/7:
        xu_user[5]=num_chargers*ac_charger_power*10/7

    # ============================================
    # Step2: 从CSV&XLSX文件读入光照、车流量数据
    # ============================================

    # 读取光照数据
    solar_data = pd.read_excel(r'.\input_data\Solar_data_use.xlsx')
    # 取前24行的第二列（DNI）的值
    solar_irradiance = solar_data.iloc[:24, 1].values # 取第2列（平均辐射强度）

    #光照数据由1天扩展到1年
    solar_irradiance_year = np.tile(solar_irradiance, 365)  # 复制365次得到8760小时

    # 归一化处理（假设1000W/m²为标准最大值） 额定功率STC标准下即在1000W/m2 长度为一天
    solar_irrad_profile = solar_irradiance / 1000.0

    # 读取直流充电负荷功率数据（每0.01h一个点）
    # dc_charge_load_data = pd.read_csv('load_simulation_results.csv')
    # dc_charge_load_profile_raw = dc_charge_load_data.iloc[:, 1].values
    module_path = r'.\Function Format for Construction of Typical Daily Load Curves for DC Charging Stations.py'
    spec = importlib.util.spec_from_file_location("dc_charging_load", module_path)
    dc_charging_load = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dc_charging_load)
    # 调用函数
    result = dc_charging_load.simulate_ev_charging(P_che=120, N_c=500)


    # 获取负荷曲线（每0.01h一个点）
    dc_charge_load_profile_raw = result['average_load_curve']
    # 将负荷曲线按小时整合
    sub_steps_per_hour = int(1 / 0.01)  # 100
    # 每100个数据点取平均，得到每小时负荷
    dc_charge_load_profile= np.mean(dc_charge_load_profile_raw.reshape(-1, sub_steps_per_hour), axis=1)
    dc_charge_load_profile_year = np.tile(dc_charge_load_profile, n_days)

    max_dc_charge_load = np.max(dc_charge_load_profile)

    # 直流充电桩负荷限制
    if xl_user[5] <  max_dc_charge_load:
        xl_user[5]=max_dc_charge_load*0.8

    if xu_user[5] <  max_dc_charge_load*2:
        xu_user[5]=max_dc_charge_load*2

    # NEW ▶
    car_flow_number = 1000 #车流量实际值也设为1000
    car_flow_number_max =  -(-max_dc_charge_load // 120) #同时充电车辆数量峰值
    # ============================================
    # Step3: 基于园区典型日负荷和充电桩数据生成园区负载曲线和交流充电负荷曲线
    # ============================================

    n_hours = len(solar_irrad_profile)*n_days
    Park_Load_Flag=1
    # typical_day_avg_load = 800.0
    if Park_Load_Flag == 1:
        # 读文件模式
        ac_path  = os.path.join('input_data', 'AC_Park_Load_Profile.xlsx')
        dc_path  = os.path.join('input_data', 'DC_Park_Load_Profile.xlsx')

        ac_park_load_profile  = pd.read_excel(ac_path ).iloc[:, 1].to_numpy()  # 第2列是功率
        dc_park_load_profile  = pd.read_excel(dc_path ).iloc[:, 1].to_numpy()

        # 长度校验
        if len(ac_park_load_profile) != len(solar_irrad_profile) or len(dc_park_load_profile) != len(solar_irrad_profile):
            raise ValueError('导入的负荷长度不是24小时，请检查！')
    else:   # Park_Load_Flag == 0
        # 代码生成模式        
        rand_amp = 0.05              # ±5 % 随机幅度

        # 24 h 手工系数（可按需改动）
        hourly_factor = np.array([
            0.19, 0.15, 0.10, 0.11, 0.13, 0.14,   # 0–5
            0.25, 0.50, 0.70, 0.88, 0.85, 0.82,   # 6–11
            0.80, 0.84, 0.82, 0.78, 0.72,         # 12–16
            0.68, 0.63, 0.58, 0.50, 0.43, 0.35, 0.30  # 17–23
        ])

        # -------- Build profile with your double loop --------
        random_variation = rand_amp * (np.random.rand(n_hours) - 0.5)
        total_park_load_profile_year = np.zeros(n_hours)

        for day in range(n_days):
            for hour in range(24):
                idx = day * 24 + hour
                base_load = typical_day_avg_load * hourly_factor[hour]
                total_park_load_profile_year[idx] = base_load * (1 + random_variation[idx])
        # 直流侧和交流侧负荷设定 #可以根据需求进行修改
        ac_park_load_profile_year = total_park_load_profile_year * (5/6)
        dc_park_load_profile_year = total_park_load_profile_year * (1/6) 
        ac_park_load_profile = ac_park_load_profile_year[:24]
        dc_park_load_profile = dc_park_load_profile_year[:24]
        # New ▶
        # 创建 DataFrame
        df_ac_park = pd.DataFrame({'AC_Load': ac_park_load_profile})
        df_dc_park = pd.DataFrame({'DC_Load': dc_park_load_profile})

        # 导出为 Excel 文件
        df_ac_park.to_excel(r'.\output_data\AC_Park_OG_Auto.xlsx', index=False)
        df_dc_park.to_excel(r'.\output_data\DC_Park_OG_Auto.xlsx', index=False)

    # 计算交流充电桩负荷曲线（9:00-18:00均匀分布）
    ac_charge_load_profile_year = np.zeros(n_hours)
    charging_hours = np.arange(9, 18)
    for day in range(365):
        for hour in charging_hours:
            ac_charge_load_profile_year[day*24 + hour] = num_chargers * ac_charger_power / len(charging_hours)
    ac_charge_load_profile = ac_charge_load_profile_year[:24]

    # 上下限修改
    dc_load_total = dc_charge_load_profile+dc_park_load_profile   #定义的是一天的量
    ac_load_total = ac_charge_load_profile+ac_park_load_profile   #定义的是一天的量
    load_total = ac_load_total+dc_load_total   #定义的是一天的量
    #总的负荷功率需求，直接取之前处理好的逐小时负荷
    dc_pv_caculated_load = sum(dc_load_total)  # 计算总负荷功率需求
    ac_pv_caculated_load = sum(ac_load_total)  # 计算总负荷功率需求
    solar_irrad_profile_caculated_load = sum(solar_irrad_profile[:24])  # 取前24小时的光照数据
    xl_user[0] = ac_pv_caculated_load * 0.97 /solar_irrad_profile_caculated_load
    xl_user[1] = dc_pv_caculated_load * 0.97 /solar_irrad_profile_caculated_load
    xu_user[0] = ac_pv_caculated_load * 1.03 /solar_irrad_profile_caculated_load
    xu_user[1] = dc_pv_caculated_load * 1.03 /solar_irrad_profile_caculated_load
    xl_user[2] = ac_pv_caculated_load * 0.5 /solar_irrad_profile_caculated_load
    xl_user[3] = dc_pv_caculated_load * 0.5 /solar_irrad_profile_caculated_load
    xu_user[2] = ac_pv_caculated_load * 3.0 /solar_irrad_profile_caculated_load
    xu_user[3] = dc_pv_caculated_load * 3.0 /solar_irrad_profile_caculated_load

    n_hours = len(solar_irrad_profile)  # 总小时数，比如24或8760
    hours = np.arange(n_hours)  # [0, 1, 2, ..., 23]

    



def run_OG_optimization():
    global xl_user, xu_user, park_space, car_number
    global Delta_t
    global electricity_buy_price, electricity_sell_price, charge_price, grid_co2_factor
    global eb_charge_eff, eb_discharge_eff
    global AC_PV_system_unit_price, DC_PV_system_unit_price
    global AC_Energy_storage_system_unit_price, DC_Energy_storage_system_unit_price
    global AC_EV_charger_unit_price, DC_EV_charger_unit_price
    global annual_maintenance_rate
    global charge_rate_ratio, discharge_rate_ratio
    global DC_to_AC_conversion_efficiency, AC_to_DC_conversion_efficiency
    global solar_irradiance, dc_charge_load_profile_raw
    global OG_flag, GC_flag
    global solar_irrad_profile, dc_charge_load_profile_year, dc_charge_load_profile, dc_park_load_profile
    global ac_charge_load_profile_year, ac_charge_load_profile , ac_park_load_profile
    global res, res_grid, n_hours, n_days
    global pv_ac_selected, pv_dc_selected, eb_ac_selected, eb_dc_selected, charger_ac_selected, charger_dc_selected
    global car_flow_number, max_dc_charge_load, car_flow_number_max
    global pop_size

    # ============================================
    # Step7：离网模式确定光伏、充电桩容量——优化问题配置与运行 NSGA-II
    # ============================================
    pop_size = 100 #种群个数 # 这里可以根据需要调整种群规模
    problem = MicrogridOptimization_1()

    algorithm = NSGA2(
        pop_size=pop_size,
    )

    res_grid = minimize(
        problem,
        algorithm,
        ('n_gen', 100),
        seed=1,
        verbose=True
    )


    Scatter().add(res_grid.F).show()

    # ============================================
    # Step8：离网模式确定光伏、充电桩容量——帕累托解集可视化
    # ============================================
    # 提取三个目标函数值
    f1 = res_grid.F[:, 0]  # 年化总成本
    f2 = res_grid.F[:, 1]  # 负荷缺电率
    f3 = res_grid.F[:, 2]  # 1 - 光伏消纳率
    # 解的容量（6个决策变量）
    pv_ac     = res_grid.X[:, 0]
    pv_dc     = res_grid.X[:, 1]
    eb_ac     = res_grid.X[:, 2]
    eb_dc     = res_grid.X[:, 3]
    charger_ac = res_grid.X[:, 4]
    charger_dc = res_grid.X[:, 5]

    # 创建交互式3D散点图
    fig = go.Figure(data=[go.Scatter3d(
        x=f1,
        y=f2,
        z=f3,
        mode='markers',
        marker=dict(
            size=5,
            opacity=0.8
        ),
        text=[f"方案 {i+1}<br>Cost: {f1[i]:.1f} CNY<br>LPSP: {f2[i]:.4f}<br>1-PV Util.: {f3[i]:.4f}%"
            for i in range(len(f1))],  # 悬停文字
        hoverinfo='text'
    )])

    # 设置布局
    fig.update_layout(
        title='Pareto Front of Multi-Objective Optimization for PV-Storage-Charging Park (3D Interactive Visualization)',
        scene=dict(
            xaxis_title='Annualized Total Cost (CNY)',
            yaxis_title='Load Shortage Rate (LPSP)',
            zaxis_title='PV Utilization Rate (%)'
        ),
        width=800,
        height=700
    )
    # 显示
    fig.show()

    # 构建 DataFrame
    df_OG_pareto = pd.DataFrame({
        "方案编号": [f"方案{i+1}" for i in range(pop_size)],
        "年化总成本 (CNY)": f1,
        "负荷缺电率 (LPSP)": f2,
        "1 - 光伏消纳率（%）": f3,
        "PV_AC 容量 (kW)": pv_ac,
        "PV_DC 容量 (kW)": pv_dc,
        "EB_AC 容量 (kWh)": eb_ac,
        "EB_DC 容量 (kWh)": eb_dc,
        "Charger_AC 容量 (kW)": charger_ac,
        "Charger_DC 容量 (kW)": charger_dc
    })
    # pv_ac, pv_dc, eb_ac, eb_dc, charger_ac, charger_dc = res_grid.X[choice]
    # 设置本地保存路径
    local_path = os.path.join(os.getcwd(), 'output_data')
    os.makedirs(local_path, exist_ok=True)
    grid_result_path = os.path.join(local_path, "Offgrid_Pareto_Results_3D.xlsx")

    # 导出为 Excel 文件
    df_OG_pareto.to_excel(grid_result_path, index=False)

    grid_result_path

    # 显示已有方案数量
    num_solutions = len(res_grid.X)

    return num_solutions

# while True: 这个指令交给用户决定 确定时OG_flag = True
# ============================================
# function3: 显示用户选定的方案      
def show_OG_selected_solution(num_solutions):    
    global xl_user, xu_user, park_space, car_number
    global Delta_t
    global electricity_buy_price, electricity_sell_price, charge_price, grid_co2_factor
    global eb_charge_eff, eb_discharge_eff
    global AC_PV_system_unit_price, DC_PV_system_unit_price
    global AC_Energy_storage_system_unit_price, DC_Energy_storage_system_unit_price
    global AC_EV_charger_unit_price, DC_EV_charger_unit_price
    global annual_maintenance_rate
    global charge_rate_ratio, discharge_rate_ratio
    global DC_to_AC_conversion_efficiency, AC_to_DC_conversion_efficiency
    global solar_irradiance, dc_charge_load_profile_raw
    global GC_flag
    global solar_irrad_profile, dc_charge_load_profile_year, dc_charge_load_profile, dc_park_load_profile
    global ac_charge_load_profile_year, ac_charge_load_profile , ac_park_load_profile
    global res, res_grid, n_hours, n_days
    global pv_ac_selected, pv_dc_selected, eb_ac_selected, eb_dc_selected, charger_ac_selected, charger_dc_selected
    global car_flow_number, max_dc_charge_load, car_flow_number_max
    global pop_size
    # ============================================
    # Step9：离网模式确定光伏、充电桩容量——用户选定合适方案
    # ============================================
    # 显示已有方案数量
    # 1. 让用户输入要查看的方案编号
    #choice = input(f"请输入要查看的离网方案编号 (1-{num_solutions}): ")
    choice = "1"  # 强制选择第1个方案（字符串形式）
    if not (choice.isdigit() and 1 <= int(choice) <= num_solutions):
        return
    choice = int(choice) - 1

    # 取用户选的方案
    pv_ac, pv_dc, eb_ac, eb_dc, charger_ac, charger_dc = res_grid.X[choice]

    max_ac_charge_power = eb_ac * charge_rate_ratio
    max_dc_charge_power = eb_dc * charge_rate_ratio
    max_ac_discharge_power = eb_ac * discharge_rate_ratio
    max_dc_discharge_power = eb_dc * discharge_rate_ratio
    pv_power_ac = pv_ac * solar_irrad_profile   #pv_ac kW irrad_profile 无量纲 pv_power_ac kW
    pv_power_dc = pv_dc * solar_irrad_profile   
    #额定容量 × 归一化光照强度
    ac_load_total = ac_charge_load_profile+ac_park_load_profile   #定义的是一天的量
    if charger_dc < car_flow_number_max:
        dc_load_total = charger_dc/car_flow_number_max * dc_charge_load_profile+dc_park_load_profile   #基于实际的充电桩数量进行修改
    else:
        dc_load_total = dc_charge_load_profile+dc_park_load_profile   #定义的是一天的量
    load_total = ac_load_total+dc_load_total
    # 初始化储能SOC数组（每小时）
    soc_ac_array = np.zeros(n_hours)
    soc_dc_array = np.zeros(n_hours)
    grid_energy_profile = np.zeros(n_hours)  # 每小时电网买卖功率，买电为正，卖电为负
    pv_curtailment_profile = np.zeros(n_hours)  # 每小时电网买卖功率，买电为正，卖电为负

    #总的负荷功率需求，直接取之前处理好的逐小时负荷
    soc_ac = eb_ac / 2
    soc_dc = eb_dc / 2
    #假设一开始电池电量为额定容量的50% 
    grid_energy = 0
    pv_wasted_ac = 0
    pv_wasted_dc = 0
    pv_wasted = 0
    pv_curtailment = 0
    total_pv_generated = 0
    total_pv_used = 0
    missing_energy = 0  
    #各类累积量的初始化（电网购电量、光伏浪费量、光伏总发电量、光伏实际利用量、供电缺口累计量）

    for t in range(24):
        # if t < 20:
        grid_energy_hour=0
        pv_curtailment_hour=0
        pv_wasted=0
        pv_wasted_ac=0
        pv_wasted_dc=0
        # hour = t % 24  # 当前小时（0-23）
        # soc_ac_array[t] = soc_ac
        # soc_dc_array[t] = soc_dc
        pv_total_ac = pv_power_ac[t]
        pv_total_dc = pv_power_dc[t]
        load_dc = dc_load_total[t]
        load_ac = ac_load_total[t]
        flag_max_dc_discharge_power = max_dc_discharge_power
        flag_max_ac_discharge_power = max_ac_discharge_power
        flag_max_dc_charge_power = max_dc_charge_power
        flag_max_ac_charge_power = max_ac_charge_power
        total_pv_generated += (pv_total_ac + pv_total_dc)

        # 先用直流光伏满足直流负荷
        supply_dc = min(pv_total_dc, load_dc)
        pv_total_dc -= supply_dc
        load_dc -= supply_dc

        # 再用交流光伏满足交流负荷
        supply_ac = min(pv_total_ac, load_ac)
        pv_total_ac -= supply_ac
        load_ac -= supply_ac

        total_pv_used += (supply_dc + supply_ac)

        # ========== 一侧富余支援另一侧负载 ==========
        if pv_total_ac > 0 and load_dc > 0:
            support_dc = min(pv_total_ac*AC_to_DC_conversion_efficiency, load_dc)
            pv_total_ac -= support_dc/AC_to_DC_conversion_efficiency
            load_dc -= support_dc
            total_pv_used += support_dc/AC_to_DC_conversion_efficiency

        if pv_total_dc > 0 and load_ac > 0:
            support_ac = min(pv_total_dc*DC_to_AC_conversion_efficiency, load_ac)
            pv_total_dc -= support_ac/DC_to_AC_conversion_efficiency
            load_ac -= support_ac
            total_pv_used += support_ac/DC_to_AC_conversion_efficiency


        # ========== 电池放电供负荷 ==========
        if load_dc > 0:
            if soc_dc >  0.1 * eb_dc:  # Ensure SOC is above 10% before discharging
                # 增加放电功率上限约束
                max_release = min(flag_max_dc_discharge_power, (soc_dc-0.1*eb_dc) / Delta_t)
                release = min(load_dc / eb_discharge_eff, max_release)
                flag_max_dc_discharge_power -= release
                soc_dc -= release * Delta_t
                load_dc -= release * eb_discharge_eff

        if load_ac > 0:
            if soc_ac > 0.1 * eb_ac:
                max_release = min(flag_max_ac_discharge_power, (soc_ac-0.1*eb_ac)  / Delta_t)
                release = min(load_ac / eb_discharge_eff, max_release)
                flag_max_ac_discharge_power -= release
                soc_ac -= release * Delta_t
                load_ac -= release * eb_discharge_eff

        if load_dc > 0:
            if load_dc > 0 and soc_ac > 0.1 * eb_ac:
                max_release = min(flag_max_ac_discharge_power, (soc_ac-0.1*eb_ac) / Delta_t)
                release = min(load_dc / eb_discharge_eff, max_release*AC_to_DC_conversion_efficiency)
                flag_max_ac_discharge_power -= release/AC_to_DC_conversion_efficiency
                soc_ac -= release/AC_to_DC_conversion_efficiency * Delta_t
                load_dc -= release * eb_discharge_eff

        if load_ac > 0:
            if load_ac > 0 and soc_dc > 0.1 * eb_dc:
                max_release = min(flag_max_dc_discharge_power, (soc_dc-0.1*eb_dc)  / Delta_t)
                release = min(load_ac / eb_discharge_eff, max_release*DC_to_AC_conversion_efficiency)
                flag_max_dc_discharge_power -= release/DC_to_AC_conversion_efficiency
                soc_dc -= release/DC_to_AC_conversion_efficiency * Delta_t
                load_ac -= release * eb_discharge_eff

        # ========== 缺电统计（无电网兜底） ==========
        remaining_deficit = load_dc + load_ac
        if remaining_deficit > 0:
            missing_energy += remaining_deficit * Delta_t
            grid_energy_hour += remaining_deficit * Delta_t

        # ========== 剩余分别充各自储能 ==========
        if pv_total_ac > 0:
            if soc_ac <  0.9 * eb_ac:
                max_release = min(flag_max_ac_charge_power, (0.9*eb_ac - soc_ac) / Delta_t)
                charge_power = min(pv_total_ac, max_release)   # 受限于PV和储能最大充电功率
                flag_max_ac_charge_power -= charge_power
                ac_charge = charge_power * eb_charge_eff
                soc_ac += ac_charge
                total_pv_used += ac_charge / eb_charge_eff
                pv_total_ac -= ac_charge / eb_charge_eff

        if pv_total_dc > 0:
            if soc_dc <  0.9 * eb_dc:
                max_release = min(flag_max_dc_charge_power, (0.9*eb_dc - soc_dc) / Delta_t)
                charge_power = min(pv_total_dc, max_release)
                flag_max_dc_charge_power -= charge_power
                dc_charge = charge_power * eb_charge_eff
                soc_dc += dc_charge
                total_pv_used += dc_charge / eb_charge_eff
                pv_total_dc -= dc_charge / eb_charge_eff

        # ========== 一侧富余支援另一侧储能 ==========
        if pv_total_dc > 0 and pv_total_ac == 0:  #说明dc侧储能已充满，ac侧储能已充但未满
            if soc_ac < 0.9 * eb_ac:
                max_release = min( pv_total_dc * DC_to_AC_conversion_efficiency , flag_max_ac_charge_power)
                ac_charge = max_release * eb_charge_eff
                if soc_ac + ac_charge * Delta_t > 0.9*eb_ac:
                    pv_total_dc=pv_total_dc-(0.9*eb_ac-soc_ac)/eb_charge_eff/DC_to_AC_conversion_efficiency
                    total_pv_used +=(0.9*eb_ac-soc_ac)/eb_charge_eff/DC_to_AC_conversion_efficiency
                    soc_ac=0.9*eb_ac
                    flag_max_ac_charge_power -= (0.9*eb_ac-soc_ac)/eb_charge_eff
                else:
                    soc_ac=soc_ac + ac_charge * Delta_t
                    flag_max_ac_charge_power -= ac_charge/eb_charge_eff
                    total_pv_used +=pv_total_dc
                    pv_total_dc=0
            else:
                pv_total_dc=pv_total_dc

            pv_wasted_dc += pv_total_dc

        if pv_total_ac > 0 and pv_total_dc == 0:  #说明ac侧储能已充满，dc侧储能已充但未满
            if soc_dc <  0.9* eb_dc:
                max_release = min (pv_total_ac * AC_to_DC_conversion_efficiency , flag_max_dc_charge_power)
                dc_charge = eb_charge_eff * max_release
                if soc_dc + dc_charge * Delta_t >  0.9* eb_dc:
                    pv_total_ac=pv_total_ac-( 0.9* eb_dc-soc_dc)/eb_charge_eff/AC_to_DC_conversion_efficiency
                    total_pv_used +=( 0.9* eb_dc-soc_dc)/eb_charge_eff/AC_to_DC_conversion_efficiency
                    soc_dc= 0.9* eb_dc
                    flag_max_dc_charge_power -= ( 0.9* eb_dc-soc_dc)/eb_charge_eff
                else:
                    soc_dc=soc_dc + dc_charge * Delta_t
                    flag_max_dc_charge_power -= ac_charge/eb_charge_eff                    
                    total_pv_used +=pv_total_ac
                    pv_total_ac=0
            else:
                pv_total_ac=pv_total_ac

            pv_wasted_ac += pv_total_ac  
        # ========== 光伏弃光统计 ==========
        pv_wasted = pv_wasted_ac + pv_wasted_dc
        if pv_wasted > 0:
            pv_curtailment += pv_wasted * Delta_t                 
        # 记录SOC
        soc_ac_array[t] = soc_ac

        grid_energy_profile[t] = grid_energy_hour
        pv_curtailment_profile[t] = pv_wasted

    # 绘图

    time_hours = np.arange(n_hours)


    ac_load_24 = ac_load_total[:24]
    dc_load_24 = dc_load_total[:24]
    grid_energy_24 = grid_energy_profile[:24]
    pv_curtailment_24 = pv_curtailment_profile[:24]
    # 画图时用 ac_load_24 和 dc_load_24


    # 本地保存路径
    local_path = os.path.join(os.getcwd(), 'output_data')

    # 创建路径（如果不存在）
    os.makedirs(local_path, exist_ok=True)

    # === 直流侧 ===
    df_dc_pv = pd.DataFrame({
        "Hour": time_hours,
        "PV_DC_Power (kW)": pv_power_dc
    })

    df_dc_storage = pd.DataFrame({
        "Hour": time_hours,
        "Storage_DC (kWh)": soc_dc_array
    })

    df_dc_load = pd.DataFrame({
        "Hour": time_hours,
        "DC_Load (kW)": dc_load_24
    })

    # === 交流侧 ===
    df_ac_pv = pd.DataFrame({
        "Hour": time_hours,
        "PV_AC_Power (kW)": pv_power_ac
    })

    df_ac_storage = pd.DataFrame({
        "Hour": time_hours,
        "Storage_AC (kWh)": soc_ac_array
    })

    df_ac_load = pd.DataFrame({
        "Hour": time_hours,
        "AC_Load (kW)": ac_load_24
    })
    # 保存路径列表
    local_paths = {
        "DC_PV": os.path.join(local_path, "DC_PV_OG.xlsx"),
        "DC_Storage": os.path.join(local_path, "DC_Storage_OG.xlsx"),
        "DC_Load": os.path.join(local_path, "DC_Load_OG.xlsx"),
        "AC_PV": os.path.join(local_path, "AC_PV_OG.xlsx"),
        "AC_Storage": os.path.join(local_path, "AC_Storage_OG.xlsx"),
        "AC_Load": os.path.join(local_path, "AC_Load_OG.xlsx"),
    }

    # 保存文件
    df_dc_pv.to_excel(local_paths["DC_PV"], index=False)
    df_dc_storage.to_excel(local_paths["DC_Storage"], index=False)
    df_dc_load.to_excel(local_paths["DC_Load"], index=False)

    df_ac_pv.to_excel(local_paths["AC_PV"], index=False)
    df_ac_storage.to_excel(local_paths["AC_Storage"], index=False)
    df_ac_load.to_excel(local_paths["AC_Load"], index=False)

    OG_flag =  "1"
    if OG_flag == "1":
        # 确认选定，跳出选择循环，进入并网优化
        pv_ac_selected, pv_dc_selected, eb_ac_selected, eb_dc_selected, charger_ac_selected, charger_dc_selected = res_grid.X[choice]
        charger_ac_selected = ceil_to_multiple(charger_ac_selected, 7)
        charger_dc_selected = ceil_to_multiple(charger_dc_selected, 120)



        files = {
            "DC_PV_OG": os.path.join(local_path, "DC_PV_OG.xlsx"),
            "DC_Storage_OG": os.path.join(local_path, "DC_Storage_OG.xlsx"),
            "DC_Load_OG": os.path.join(local_path, "DC_Load_OG.xlsx"),
            "AC_PV_OG": os.path.join(local_path, "AC_PV_OG.xlsx"),
            "AC_Storage_OG": os.path.join(local_path, "AC_Storage_OG.xlsx"),
            "AC_Load_OG": os.path.join(local_path, "AC_Load_OG.xlsx"),
            # "Choice_OG": os.path.join(local_base_path, "Offgrid_Pareto_Results_3D.xlsx")
        }                   
        # 打开每个表格，添加对应容量为最后一行
        df_ac_pv = pd.read_excel(files["AC_PV_OG"])
        df_dc_pv = pd.read_excel(files["DC_PV_OG"])
        df_ac_storage = pd.read_excel(files["AC_Storage_OG"])
        df_dc_storage = pd.read_excel(files["DC_Storage_OG"])
        df_ac_load = pd.read_excel(files["AC_Load_OG"])
        df_dc_load = pd.read_excel(files["DC_Load_OG"])
        # df_choice = pd.read_excel(files["Choice_OG"])

        # 添加最后一行数据（将 Hour 设置为 "最终方案" 或 NaN）
        df_ac_pv.loc[len(df_ac_pv)] = ["离网方案", pv_ac_selected]
        df_dc_pv.loc[len(df_dc_pv)] = ["离网方案", pv_dc_selected]
        df_ac_storage.loc[len(df_ac_storage)] = ["离网方案", eb_ac_selected]
        df_dc_storage.loc[len(df_dc_storage)] = ["离网方案", eb_dc_selected]
        df_ac_load.loc[len(df_ac_load)] = ["离网负荷部分充电桩方案", charger_ac_selected]
        df_dc_load.loc[len(df_dc_load)] = ["离网负荷部分充电桩方案", charger_dc_selected]
        # df_choice.loc[len(df_choice)] = ["离网选择方案", choice+1]
        # 保存修改
        df_ac_pv.to_excel(files["AC_PV_OG"], index=False)
        df_dc_pv.to_excel(files["DC_PV_OG"], index=False)
        df_ac_storage.to_excel(files["AC_Storage_OG"], index=False)
        df_dc_storage.to_excel(files["DC_Storage_OG"], index=False)
        df_ac_load.to_excel(files["AC_Load_OG"], index=False)
        df_dc_load.to_excel(files["DC_Load_OG"], index=False)
        # df_choice.to_excel(files["Choice_OG"], index=False)

        # 设置路径
        path = r'.\output_data\Offgrid_Pareto_Results_3D.xlsx'

        # 读取原始表格
        df = pd.read_excel(path)

        # 创建一个空行，只有前两列填写，其他列自动为 NaN
        new_row = {df.columns[0]: "离网选择方案", df.columns[1]: choice + 1}
        df.loc[len(df)] = new_row

        # 保存回原文件
        df.to_excel(path, index=False)

        df_ac_SOC = pd.DataFrame({
            "Hour": time_hours,
            "SOC_AC (%)": 100*soc_ac_array/eb_ac_selected
        })

        df_dc_SOC = pd.DataFrame({
            "Hour": time_hours,
            "SOC_DC (%)": 100*soc_dc_array/eb_dc_selected
        })
        # 保存路径列表
        local_paths = {
            "SOC_AC": os.path.join(local_path, "AC_SOC_OG.xlsx"),
            "SOC_DC": os.path.join(local_path, "DC_SOC_OG.xlsx"),
        }

        # 保存文件
        df_ac_SOC.to_excel(local_paths["SOC_AC"], index=False)
        df_dc_SOC.to_excel(local_paths["SOC_DC"], index=False)
        #files      
   
        return
    else:
        return
    
def main():
    # Step 1: 预处理输入参数（生成负荷、光照等数据）
    typical_day_avg_load=800
    preprocess_inputs(typical_day_avg_load)

    # Step 2: 运行离网优化，得到解集和数量
    num_solutions = run_OG_optimization()

    # Step 3: 让用户选择方案并展示结果
    show_OG_selected_solution(num_solutions)

if __name__ == "__main__":
    main()