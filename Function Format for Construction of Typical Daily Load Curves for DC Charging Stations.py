import numpy as np
from scipy.stats import lognorm, uniform, norm
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Sequence, Dict, Any

# ------------------------------------------------------------
# 核心函数：仅两个必选输入参数
#   P_che : 充电功率 (kW)，可以是单个数值或可迭代序列(例如 [7,60])
#   N_c   : 每次蒙特卡洛仿真的车辆数量
# 其它建模假设参数在函数内部定义，如需改动再升级为可选参数
# ------------------------------------------------------------

def simulate_ev_charging(P_che: Union[float, int, Sequence[float]], N_c: int) -> Dict[str, Any]:
    """模拟典型日光储充园区电动车充电负荷（蒙特卡洛）。

    Parameters
    ----------
    P_che : float | int | Sequence[float]
        充电功率集合 (kW)。若为单个数值，则表示所有车辆采用相同功率；
        若为序列，则每辆车在其充电区间随机抽取其中一个功率（等概率，可按需扩展为自定义概率）。
    N_c : int
        每次蒙特卡洛仿真的车辆数量。

    Returns
    -------
    result : dict
        包含以下键：
        - 'time_steps' : np.ndarray, 时间序列 (h)
        - 'average_load_curve' : np.ndarray, 平均负荷曲线 (kW)
        - 'average_num_curve'  : np.ndarray, 平均并发充电车辆数曲线
        - 'total_load_matrix'  : np.ndarray, shape=(M_c, T)
        - 'total_num_matrix'   : np.ndarray, shape=(M_c, T)
        - 'peak_time'          : float, 平均并发车辆峰值出现时间 (h)
        - 'peak_num'           : int,   平均并发车辆峰值数量
        - 'dataframes'         : dict,  {'load_df': DataFrame, 'num_df': DataFrame}

    Notes
    -----
    * 时间步长为 0.01 h (=36 秒)。
    * 行驶里程 m ~ LogNormal(mu_l, sigma_l)；初始 SOC ~ U[a,b]。
    * 到站 SOC = 初始SOC - 路耗/(电池容量)。路耗电量= 行驶里程 * (随机季节能耗/100)。
    * 充电时长 = (1 - 到站SOC)*E_h / (P * η)。
    * 车辆到达时间：偶数索引车辆服从上午正态 N(mu_c_am, sigma_c_am)，奇数索引服从下午正态 N(mu_c_pm, sigma_c_pm)。
    * 结果未做季节加权输出拆分；只记录综合典型日。
    """

    # ------------------ 固定建模参数（可按需改成可选） ------------------
    M_c = 30                  # 蒙特卡洛仿真次数
    mu_l = 3.2                # log(m) 的均值
    sigma_l = 0.88            # log(m) 的标准差
    soc_min, soc_max = 0.8, 1.0
    mu_c_am, mu_c_pm = 10.50, 19.00   # 到达时间均值 (h)
    sigma_c_am, sigma_c_pm = 2.14, 3.14
    eta = 0.90               # 充电效率
    E_h = 80.0               # 电池总容量 (kWh)
    season_Eb = [20.54, 18.89, 20.0]  # 冬 / 夏 / 过渡季 (kWh/100km)
    season_prob = [0.25, 0.25, 0.50]  # 各季节概率
    dt = 0.01                # 时间步长 (h)

    # 处理功率输入
    if isinstance(P_che, (int, float)):
        P_choices = np.array([float(P_che)])
    else:
        P_choices = np.array(list(P_che), dtype=float)
    if P_choices.size == 0:
        raise ValueError("P_che 不能为空")

    # 时间轴
    time_steps = np.arange(0, 24, dt)
    T = len(time_steps)

    # 结果矩阵
    total_load_matrix = np.zeros((M_c, T))
    total_num_matrix = np.zeros((M_c, T))

    # ------------------ 蒙特卡洛主循环 ------------------
    for sim in range(M_c):
        # 行驶里程 (截断下限0)
        m_samples = lognorm.rvs(s=sigma_l, scale=np.exp(mu_l), size=N_c)
        m_samples = np.clip(m_samples, 0, None)
        # 初始 SOC
        soc0 = uniform.rvs(loc=soc_min, scale=soc_max - soc_min, size=N_c)
        # 到达时间：交替使用上午/下午分布
        Ts = np.empty(N_c)
        Ts[0::2] = norm.rvs(loc=mu_c_am, scale=sigma_c_am, size=(N_c + 1)//2)
        Ts[1::2] = norm.rvs(loc=mu_c_pm, scale=sigma_c_pm, size=N_c//2)
        Ts = np.clip(Ts, 0, 24)
        # 季节能耗采样
        Eb_samples = np.random.choice(season_Eb, size=N_c, p=season_prob)
        # 到站 SOC
        soc_arrival = soc0 - (m_samples * Eb_samples) / (E_h * 100)
        soc_arrival = np.clip(soc_arrival, 0, 1)
        # 充电功率采样
        P_sample = np.random.choice(P_choices, size=N_c)
        # 充电时长 (h)
        Tc = (1 - soc_arrival) * E_h / (P_sample * eta)

        # 逐车写入功率区间
        # 向量化思路：构造每辆车的起止索引并用累加器
        start_idx = np.floor(Ts / dt).astype(int)
        end_idx = np.floor((Ts + Tc) / dt).astype(int)
        end_idx = np.clip(end_idx, 0, T)

        # 差分数组法 (前缀和) 高效叠加功率
        diff = np.zeros(T + 1)
        for p, s, e in zip(P_sample, start_idx, end_idx):
            if s < T:
                diff[s] += p
                diff[e] -= p
        load_curve = np.cumsum(diff[:-1])
        total_load_matrix[sim] = load_curve

        # 并发车辆数（同样用差分法计数）
        diff_n = np.zeros(T + 1)
        for s, e in zip(start_idx, end_idx):
            if s < T:
                diff_n[s] += 1
                diff_n[e] -= 1
        num_curve = np.cumsum(diff_n[:-1])
        total_num_matrix[sim] = num_curve

    # 平均曲线
    average_load_curve = total_load_matrix.mean(axis=0)
    average_num_curve = total_num_matrix.mean(axis=0)

    # 峰值统计
    peak_idx = int(average_num_curve.argmax())
    peak_time = time_steps[peak_idx]
    peak_num = int(round(average_num_curve[peak_idx]))

    # 生成 DataFrame
    df_time = pd.DataFrame(time_steps, columns=["Time_h"])
    load_df = pd.concat([
        df_time,
        pd.DataFrame(average_load_curve, columns=["Avg_Load_kW"]),
        pd.DataFrame(total_load_matrix.T, columns=[f"Load_Sim_{i+1}" for i in range(M_c)])
    ], axis=1)

    num_df = pd.concat([
        df_time,
        pd.DataFrame(average_num_curve, columns=["Avg_Num_EVs"]),
        pd.DataFrame(total_num_matrix.T, columns=[f"Num_Sim_{i+1}" for i in range(M_c)])
    ], axis=1)

    return {
        'time_steps': time_steps,
        'average_load_curve': average_load_curve,
        'average_num_curve': average_num_curve,
        'total_load_matrix': total_load_matrix,
        'total_num_matrix': total_num_matrix,
        'peak_time': peak_time,
        'peak_num': peak_num,
        'dataframes': {'load_df': load_df, 'num_df': num_df}
    }


# ------------------ 示例调用（使用时可删除） ------------------
if __name__ == "__main__":
    result = simulate_ev_charging(P_che=60, N_c=1500)
    print(f"峰值并发发生在 {result['peak_time']:.2f} h, 车辆数 ≈ {result['peak_num']}")
    # 可视化
    plt.figure(figsize=(10,4))
    plt.plot(result['time_steps'], result['average_load_curve'])
    plt.xlabel('Time (h)'); plt.ylabel('Average Load (kW)'); plt.grid(True, linestyle=':')
    plt.title('Typical Daily EV Charging Load')
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(result['time_steps'], result['average_num_curve'])
    plt.xlabel('Time (h)'); plt.ylabel('Average #EVs Charging'); plt.grid(True, linestyle=':')
    plt.title('Typical Daily Concurrent EV Count')
    plt.tight_layout(); plt.show()

    # 导出 CSV 示例
    result['dataframes']['load_df'].to_csv(r'.\output_data\load_results.csv', index=False)
    result['dataframes']['num_df'].to_csv(r'.\output_data\num_results.csv', index=False)
    print('已导出 load_results.csv 与 num_results.csv')
