from cProfile import label
import numpy as np
from scipy import interpolate as spip
import matplotlib.pyplot as plt

"""
一维非稳态传热
阶跃/斜坡边界条件
均采用国际单位
"""

"""
===函数===
"""
# 由干空气温度、压强求热物性，可导入课本P536页数据并进行插值
# def air(air_tem, air_pres):

#     spip.interp1d()

#     return air_rho, air_cp, air_lam, air_a, air_miu, air_v, air_Pr


# 计算对流换热系数
# def calculate_h(air_spd, l, air_tem, air_pres=1.01325 * pow(10, 5)):
#     air_Pr = spip.interp1d()
#     air_Re = air_spd * l / air_v
#     h = 0.332 * air_lam / x * math.sqrt(air_Re) * pow(air_Pr, 1 / 3)
#     return h


# 内部单个节点的显式迭代
def next_inner_node(step_x, step_time, a, tem_preP_curT, tem_curP_curT, tem_nextP_curT):
    tem_curP_nextT = (
        a * step_time * (tem_preP_curT + tem_nextP_curT) / step_x / step_x
        + (1 - 2 * a * step_time / step_x / step_x) * tem_curP_curT
    )
    return tem_curP_nextT


# 处理对流换热边界节点
# 由于无限大简化成一维，可以直接用局部对流换热系数
# 假设是湍流
# def heat_convection_boundary():
# h已知时的对流换热边界条件
def heat_convection_boundary_const_h(
    step_x, step_time, a, tem_curP_curT, tem_nextP_curT, air_tem
):
    tem_curP_nextT = (
        tem_curP_curT
        * (
            1
            - 2 * h * step_time / rho / c / step_x
            - 2 * a * step_time / step_x / step_x
        )
        + 2 * a * step_time / step_x / step_x * tem_nextP_curT
        + 2 * h * step_time / rho / c / step_x * air_tem
    )
    return tem_curP_nextT


# 处理远处边界节点，假定远处边界恒温
def far_boundary_const_tem(step_x, step_time, a, tem_preP_curT, tem_curP_curT):
    tem_curP_nextT = tem_curP_curT
    return tem_curP_nextT


# 处理远处边界节点，假定远处边界绝热
def far_boundary_adiabatic(step_x, step_time, a, tem_preP_curT, tem_curP_curT):
    tem_curP_nextT = (
        tem_curP_curT * (1 - 2 * a * step_time / step_x / step_x)
        + 2 * a * step_time / step_x / step_x * tem_preP_curT
    )
    return tem_curP_nextT


# 所有节点的单次显式迭代
def single_iteration(num_of_nodes, step_x, step_time, a, tem_cur_time, air_tem):
    tem_next_time = np.zeros(num_of_nodes)
    for i in range(num_of_nodes):
        if i == 0:  # 第一个节点，对流换热边界
            tem_next_time[i] = heat_convection_boundary_const_h(
                step_x, step_time, a, tem_cur_time[i], tem_cur_time[i + 1], air_tem
            )
            continue
        if i == num_of_nodes - 1:  # 最后一个节点，远边界
            tem_next_time[i] = far_boundary_adiabatic(
                step_x, step_time, a, tem_cur_time[i - 1], tem_cur_time[i]
            )
            continue

        tem_next_time[i] = next_inner_node(
            step_x,
            step_time,
            a,
            tem_cur_time[i - 1],
            tem_cur_time[i],
            tem_cur_time[i + 1],
        )
    return tem_next_time


"""
===主程序===
"""
if __name__ == "__main__":
    """
    定义材料参数：
    lam         材料导热系数    W/(m*K)
    rho         材料密度        kg/m^3
    c           材料比热容      J/(kg*K)
    a           热扩散率        m^2/s
    init_tem    材料初始温度    ℃
    """
    lam = 0.2
    rho = 1171
    c = 1466
    a = lam / rho / c
    init_tem = 20

    """
    定义流体（干空气）参数：
    air_tem     温度        ℃
    air_pres    压力        Pa
    air_rho     密度        kg/m^3
    air_cp      比热        J/(kg*K)
    air_lam     导热系数    W/(m*K)
    air_a       热扩散率    m^2/s
    air_miu     黏度        kg/(m*s)
    air_v       动力粘度    m^2/s
    air_Pr      空气普朗特数
    """
    # air_tem = init_tem + 30
    air_pres = 1.01325 * pow(10, 5)
    air_rho = 0
    air_cp = 0
    air_lam = 0
    air_a = 0
    air_miu = 0
    air_v = 0
    air_Pr = 0

    """
    h(float)    对流换热系数    W/(m^2*K)
    """
    h = 50  # 默认对流换热系数为50，除非专门计算

    """
    定义仿真条件：
    num_of_nodes(int)   节点数
    step_x(float)       空间步长    m
    step_time(float)    时间步长    s
    endtime(float)      仿真时长    s
    """
    num_of_nodes = 100
    step_x = 0.0001
    step_time = 0.001
    endtime = 1000

    """
    定义结果变量：
    tem_cur_time(np.array)  当前时间各节点温度  ℃
    tem_next_time(np.array) 下一时间各节点温度  ℃
    """
    tem_cur_time = np.full(num_of_nodes, init_tem)  # 1D array
    tem_next_time = np.full(num_of_nodes, init_tem)  # 1D array
    iteration_times = int(endtime / step_time)  # 迭代次数
    boundary_tem = np.zeros(iteration_times + 1)  # 保存初始和每次迭代的壁面温度
    boundary_tem[0] = init_tem
    far_boundary_tem = np.zeros(iteration_times + 1)  # 保存远处边界温度
    far_boundary_tem[0] = init_tem
    time_array = np.zeros(iteration_times + 1)  # 保存初始和每次迭代的时刻

    air_tem = np.full(iteration_times + 1, init_tem)
    for i in range(iteration_times + 1):
        # air_tem[i] = 50  # 阶梯加热
        if i * step_time <= 30:  # 斜坡加热
            air_tem[i] = 20 + i * (50 - 20) / (30 / step_time)
        else:
            air_tem[i] = 50

    current_time = 0
    for i in range(iteration_times):
        current_time += step_time
        # print(current_time)
        time_array[i + 1] = current_time
        tem_next_time = single_iteration(
            num_of_nodes, step_x, step_time, a, tem_cur_time, air_tem[i]
        )
        boundary_tem[i + 1] = tem_next_time[0]
        far_boundary_tem[i + 1] = tem_next_time[num_of_nodes - 1]
        tem_cur_time = tem_next_time

    plt.autoscale(enable=True, axis="both")
    plt.scatter(time_array, boundary_tem, 0.5, marker="o", label="Boundary temperature")
    plt.scatter(
        time_array, far_boundary_tem, 0.5, marker="o", label="Far boundary temperature"
    )
    plt.scatter(time_array, air_tem, 0.5, marker="o", label="Mainstream temperature")
    plt.title("Temperature-time plot under boundary condition III (Analytical)")
    plt.xlabel("Time(s)")
    plt.ylabel("Temperature(℃)")
    plt.legend()
    plt.grid(ls="--")
    plt.show()
