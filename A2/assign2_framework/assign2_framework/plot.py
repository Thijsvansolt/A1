import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    a1 = [2.064, 0.753, 1, 1.97, 19, 176, 45.096]
    b1 = [16.264, 12.124, 9.180, 12.875, 22.583, 27.759, 32.555]
    c1 = [16.701, 8.741, 5.188, 3.521, 3.458, 3.173, 3.740]
    d1 = [18.887, 8.627, 4.433, 2.859, 1.978, 1.732, 1.658]
    e1 = [205.205, 105.743, 65.353, 48.979, 46.727, 40.372, 47.775]

    a2 = []
    print(range(len(a1)))
    for value in range(0, len(a1)):
        a2.append(a1[0] / a1[value])
    print(a2)

    b2 = []
    for value in range(0, len(b1)):
        b2.append(b1[0] / b1[value])
    # print(b2)

    c2 = []
    for value in range(0, len(c1)):
        c2.append(c1[0] / c1[value])
    # print(c2)

    d2 = []
    for value in range(0, len(d1)):
        d2.append(d1[0] / d1[value])
    # print(d2)

    e2 = []
    for value in range(0, len(e1)):
        e2.append(e1[0] / e1[value])
    # print(e2)   

    base = [10^3, 10^3, 10^4, 10^5, 10^6, 10^7]
    plt.plot(base, a2, 'r', label='wave_equation')
    # plt.plot(base, b2, 'g', label='10^4')
    # plt.plot(base, c2, 'b', label='10^5')
    # plt.plot(base, d2, 'y', label='10^6')
    # plt.plot(base, e2, 'm', label='10^7')
    plt.legend()
    plt.xlabel('problem size')
    plt.ylabel('Speedup')
    plt.savefig("scale.png")
    plt.show()
