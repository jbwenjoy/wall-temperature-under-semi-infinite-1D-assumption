import math
import matplotlib.pyplot as plt


def boundary_III(t_0, t_inf, x, time, a, h, lamda):
    result = math.erfc(x / 2 / math.sqrt(a * time)) - math.exp(
        h * x / lamda + h * h * a * time / lamda / lamda
    ) * math.erfc(x / 2 / math.sqrt(a * time) + h * math.sqrt(a * time) / lamda)
    result = result * (t_inf - t_0) + t_0
    return result


if __name__ == "__main__":
    # 定义条件与物性参数（m-kg-s）
    x = 0  # 壁面
    t_0 = 20  # 初始温度
    t_inf = 50  # 主流温度
    lamda = 0.20  # 导热
    rho = 1171  # 密度
    c = 1466  # 比热
    a = lamda / rho / c  # 热扩散率
    h = 50  # 对流换热系数

    tem = []
    timeline = []
    for i in range(1000):
        time = i / 10
        timeline.append(time)
        if time == 0:
            tem.append(t_0)
            continue
        tem.append(boundary_III(t_0, t_inf, x, time, a, h, lamda))

    data_dict = {}
    for i, j in zip(timeline, tem):
        data_dict[i] = j
    plt.title("Temperature-time plot under boundary condition III (P117)")
    plt.xlabel("Time(s)")
    plt.ylabel("Temperature(℃)")
    x = [i for i in data_dict.keys()]
    y = [i for i in data_dict.values()]
    plt.plot(x, y)
    plt.legend()
    plt.show()
