from collections import defaultdict

import numpy as np

# 颜色原代码
color = np.array([
    [244, 242, 229],  # living room #客厅
    [253, 244, 171],  # bedroom #卧室
    [234, 216, 214],  # kitchen #厨房
    [205, 233, 252],  # bathroom #浴室
    [208, 216, 135],  # balcony #阳台
    [249, 222, 189],  # Storage #储物间

    # [230, 85, 13],  # living room #客厅
    # [31, 119, 180],  # bedroom #卧室
    # [253, 141, 60],  # kitchen #厨房
    # [107, 110, 207],  # bathroom #浴室
    # [44, 160, 44],  # balcony #阳台
    # [82, 84, 163],  # Storage #储物间

    [79, 79, 79],  # exterior wall #外墙
    [255, 225, 25],  # FrontDoor #前门
    [128, 128, 128],  # interior wall #内墙
    [255, 255, 255]  # 白色未知
], dtype=np.int64)


def get_color_map():
    cIdx = np.array([1, 2, 3, 4, 1, 2, 2, 2, 2, 5, 1, 6, 1, 10, 7, 8, 9, 10]) - 1
    return color[cIdx]


cmap = get_color_map() / 255.0

# 仿写 颜色中文代码
color_explain = {
    0: 'living_room(客厅)',
    1: 'bedroom(卧室)',
    2: 'kitchen(厨房)',
    3: 'bathroom(浴室)',
    4: 'balcony(阳台)',
    5: 'storage(仓库)',
    6: 'exterior wall(外墙)',
    7: 'FrontDoor(前门)',
    8: 'interior_wall(内墙)',
    9: 'unknow(白色未知)',
}
color_cn = []
color_en = []
for i in range(0, 10):
    explain = color_explain[i]
    id1, id2 = explain.index('('), explain.index(')')
    cn = explain[id1 + 1:id2]
    en = explain[0: id1]
    color_cn.append(cn)
    color_en.append(en)

color_cn = np.array(color_cn)
color_en = np.array(color_en)


def get_color_idx(en):
    for i in range(len(color_en)):
        if color_en[i] == en:
            return i
    return 9


def get_color_cn_map():
    cIdx = np.array([1, 2, 3, 4, 1, 2, 2, 2, 2, 5, 1, 6, 1, 10, 7, 8, 9, 10]) - 1
    return color_cn[cIdx]


def get_color_en_map():
    cIdx = np.array([1, 2, 3, 4, 1, 2, 2, 2, 2, 5, 1, 6, 1, 10, 7, 8, 9, 10]) - 1
    return color_en[cIdx]


cn_map = get_color_cn_map()
en_map = get_color_en_map()


def get_color_room_type(en):
    for i, e in enumerate(en_map):
        if e == en:
            return i
    return -1


# 用来生成颜色解释的东西
def generate_color_explain():
    arr1, arr2, arr3 = [], [], []
    for i in range(0, 10):
        cn = color_cn[i]
        en = color_en[i]
        if cn.find('门') != -1 or cn.find('墙') != -1 or cn.find('未知') != -1:
            continue
        arr1.append(color[i] / 255.0)
        arr2.append(color_cn[i])
        arr3.append(en)
    return arr1, arr2,arr3


ex1, ex2,ex3, = generate_color_explain()
len_ex = len(ex1)
