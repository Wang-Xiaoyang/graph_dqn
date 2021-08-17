import math
import random


def solution(Loc_DU, Loc_RU):
    """
    Allocate DUs resources based on DUs and RUs location
    :param Loc_DU: All DUs position information (latitude and longitude)
    :param Loc_RU: All RUs position information (latitude and longitude)
    :return: Allocation scheme (correspondence between DU and RU)
    """
    def geo_distance(lng1, lat1, lng2, lat2):  # 地理距离
        """
        Calculate the great circle distance between two points
            on the earth (specified in decimal degrees)
        :param lng1: The longitude of the first point
        :param lat1: The latitude of the first point
        :param lng2: The longitude of the second point
        :param lat2: The latitude of the second point
        :return: The actual distance between two points
        """
        lng1, lat1, lng2, lat2 = map(math.radians,
                                     map(float, [lng1, lat1, lng2, lat2]))  # 根据提供的函数对指定序列做映，radians:将角度转换为弧度。
        dlon = lng2 - lng1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371
        return c * r

    def pack1(w, v, C):  # 每个东西只能选择一次
        """
        Method of allocation per DU
        :param w: The demand of the RU
        :param v: Inverse of the delay in RU and DU communications (delay is inversely proportional to value)
        :param C: The capacity of DU
        :return: Allocation scheme of Each DU (correspondence between DU and RU)
        """
        dp = [[0 for _ in range(C + 1)] for _ in range(len(w) + 1)]
        for i in range(1, len(w) + 1):
            for j in range(1, C + 1):
                if j < w[i - 1]:  # 如果剩余容量不够新来的物体 直接等于之前的
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w[i - 1]] + v[i - 1])
        a = len(w)
        item = [0 for _ in range(a + 1)]
        for i in reversed(range(1, a + 1)):
            if dp[i][C] == dp[i - 1][C]:
                item[i] = 0
            elif C - w[i - 1] >= 0 and dp[i][C] == dp[i - 1][C - w[i - 1]] + v[i - 1]:
                item[i] = 1
                C -= w[i - 1]

        return item

    costs_1 = []
    costs_2 = []
    # Calculate the communication latency between each RUs and the first DU
    for i in range(0, 7):
        costs_1.append(1)
        costs_1[i] = geo_distance(Loc_DU[0][1], Loc_DU[0][0], Loc_RU[i][1], Loc_RU[i][0])
        i += 1
    # Calculate the communication latency between each RUs and the second DU
    for i in range(0, 7):
        costs_2.append(1)
        costs_2[i] = geo_distance(Loc_DU[1][1], Loc_DU[1][0], Loc_RU[i][1], Loc_RU[i][0])
        i += 1
    # Allocation of RU's to DU's with lower communication latency by comparison
    DU_1 = []
    DU_2 = []
    Num_1 = []
    Num_2 = []
    for i in range(0, 7):
        DU_1.append(0)
        DU_2.append(0)
        Num_1.append(0)
        Num_2.append(0)
        if costs_1[i] <= costs_2[i]:
            DU_1[i] = (1 / costs_1[i]) * 5
            Num_1[i] = i + 1
        else:
            DU_2[i] = (1 / costs_2[i]) * 5
            Num_2[i] = i + 1
        i += 1
    # delete zero
    for i in range(len(DU_1) - 1, -1, -1):
        if DU_1[i] == 0:
            DU_1.remove(0)
    for i in range(len(DU_2) - 1, -1, -1):
        if DU_2[i] == 0:
            DU_2.remove(0)
    for i in range(len(Num_1) - 1, -1, -1):
        if Num_1[i] == 0:
            Num_1.remove(0)
    for i in range(len(Num_2) - 1, -1, -1):
        if Num_2[i] == 0:
            Num_2.remove(0)

    a = len(DU_1)
    b = len(DU_2)
    Dem_1 = []
    Dem_2 = []
    # Using random numbers to generate the requirements for each RU
    for i in range(0, a):
        Dem_1.append(1)
        Dem_1[i] = random.randint(5, 10)
        i += 1
    for i in range(0, b):
        Dem_2.append(1)
        Dem_2[i] = random.randint(5, 10)
        i += 1
    # Priority is given to allocating resources for the DU that contains more RUs,
    # and if there is an unallocated RU divide the RU into a second DU for re-allocation of resources.
    if a != 0 and a > b:
        x = pack1(Dem_1, DU_1, 30)
        for i in range(0, a):
            if x[i + 1] == 0:
                Dem_2.append(1)
                DU_2.append(1)
                Num_2.append(1)
                Dem_2[b] = Dem_1[i]
                Dem_1[i] = 0
                DU_2[b] = (1 / costs_2[Num_1[i] - 1]) * 5
                DU_1[i] = 0
                Num_2[b] = Num_1[i]
                Num_1[i] = 0
                b += 1
        # delete zero
        for i in range(len(Dem_1) - 1, -1, -1):
            if Dem_1[i] == 0:
                Dem_1.remove(0)
        for i in range(len(DU_1) - 1, -1, -1):
            if DU_1[i] == 0:
                DU_1.remove(0)
        for i in range(len(Num_1) - 1, -1, -1):
            if Num_1[i] == 0:
                Num_1.remove(0)
        y = pack1(Dem_2, DU_2, 30)
    elif b != 0 and a <= b:
        y = pack1(Dem_2, DU_2, 30)
        for i in range(0, b):
            if y[i + 1] == 0:
                Dem_1.append(1)
                DU_1.append(1)
                Num_1.append(1)
                Dem_1[a] = Dem_2[i]
                Dem_2[i] = 0
                DU_1[a] = (1 / costs_2[Num_2[i] - 1]) * 5
                DU_2[i] = 0
                Num_1[a] = Num_2[i]
                Num_2[i] = 0
                a += 1
        # delete zero
        for i in range(len(Dem_2) - 1, -1, -1):
            if Dem_2[i] == 0:
                Dem_2.remove(0)
        for i in range(len(DU_2) - 1, -1, -1):
            if DU_2[i] == 0:
                DU_2.remove(0)
        for i in range(len(Num_2) - 1, -1, -1):
            if Num_2[i] == 0:
                Num_2.remove(0)
        x = pack1(Dem_1, DU_1, 30)
    return x, y, Num_1, Num_2
