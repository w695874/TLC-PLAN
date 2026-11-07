# 最初的数据集,rplan最初的数据集是图片，需要解析的
import json
from collections import defaultdict

import numpy as np
from shapely import geometry

from config import room_color
from config.room_color import get_color_en_map

origin_datasets = 'E:/workdata/floorplan_dataset/'

# 最初的数据集增强版，图片经过了增强，_90,_180是经过了选择，_t是水平翻转,_v是垂直翻转
origin_datasets_strong = 'E:/workdata/floorplan_dataset2/'

# 数据集的图片转换成的序列，格式为
# [
#   {
#     "instruction": "[(165,87),(185,87),(185,229),(128,229),(128,221),(72,221),(72,87),(115,87),(115,53),(148,53),(148,28),(165,28)]",
#     "input": "",
#     "output": "[[1,(72.0,159.5),(72.0,221.0),(128.0,221.0),(128.0,159.5)],[3,(72.0,128.5),(72.0,159.5),(108.0,159.5),(108.0,128.5)],[0,(165.0,87.0),(165.0,28.0),(148.0,28.0),(148.0,87.0),(125.0,87.0),(125.0,128.5),(108.0,128.5),(108.0,159.5),(128.0,159.5),(128.0,207.0),(185.0,207.0),(185.0,87.0)],[7,(72.0,87.0),(72.0,128.5),(125.0,128.5),(125.0,87.0)],[2,(115.0,53.0),(115.0,87.0),(148.0,87.0),(148.0,53.0)],[9,(128.0,229.0),(185.0,229.0),(185.0,207.0),(128.0,207.0)]]"
#   }
# ]
# 文件夹是生成日期_开始_结束  最终文件是json格式,文件名为：开始_结束.json
origin_datasets_text = 'E:/workdata/floorplan_dataset3/'

origin_datasets_text2 = 'E:/workdata/floorplan_dataset4/'
# 每个文件名对应的外边界,1没有门，2有门
origin_datasets_boundary1 = 'E:/workdata/floorplan_dataset4_boundary1/'
origin_datasets_boundary2 = 'E:/workdata/floorplan_dataset4_boundary2/'

origin_datasets_boundary3 = 'E:/workdata/floorplan_dataset4_boundary3/'

# 输出生成和原图对比的文件夹
output_compare_pic = 'E:/workdata/floorplan_output_pic/'
output_compare_pic2 = 'E:/workdata/floorplan_output_pic2/'
output_compare_pic3 = 'E:/workdata/floorplan_output_pic3/'
output_compare_pic4 = 'E:/workdata/floorplan_output_pic4/'


class RplanRoom:
    def __init__(self):
        super().__init__()
        # 外边界点
        self.external_boundary = []
        # 内部房间
        self.inner_rooms = []

    def __str__(self):
        inners = ''
        for inner_room in self.inner_rooms:
            inners += (str(inner_room) + '\n')
        return f"""房间外边界点集合 :{str(self.external_boundary)}\n{len(self.inner_rooms)}个内部房间\n内部房间点集合:{inners}
        """

    def csv_data(self):
        instruction = '['
        for e in self.external_boundary:
            if e[3] == 1:
                instruction += f'({e[0]},{e[1]},1),'
            else:
                instruction += f'({e[0]},{e[1]}),'
        instruction = instruction[0:len(instruction) - 1]
        instruction += ']'

        input = ''

        response = '['
        for inner_room in self.inner_rooms:
            response += inner_room.point_data()
            response += ','
        response = response[0:len(response) - 1]
        response += ']'

        return {"instruction": instruction, "input": input, "output": response}

    def dict_data(self):
        d = {'external_boundary': {'vertx': []},
             'front_door': {'vertx': []},
             'inner_rooms': [],
             'inner_rooms_count': defaultdict(int)}
        # 参考plot_boundary方法
        boundary = self.external_boundary
        is_new = boundary[:, -1] == 1
        poly_boundary = geometry.Polygon(boundary[~is_new, :2])
        x, y = poly_boundary.exterior.xy

        # 前面两个是门
        # 坐标小的排在前面，门的两个点肯定有个坐标相同，那么另一个坐标小的排在前面,所以如果第一点大那么就交换位置
        door = boundary[:2, :2]
        idx = np.argmin(np.sum(door, axis=-1), axis=0)
        if idx == 1: door = door[[1, 0]]
        for i in range(len(x)):
            pos = f'({float(x[i])},{float(y[i])})'
            d['external_boundary']['vertx'].append(pos)

        for i, e in enumerate(door):
            pos = f'({float(e[0])},{float(e[1])})'
            d['front_door']['vertx'].append(pos)

        en_map = get_color_en_map()
        for inner_room in self.inner_rooms:
            r = {}
            room_type = inner_room.room_type()
            type_str = str(en_map[room_type])
            r['type'] = type_str
            position = []
            for e in inner_room.get_points():
                pos = f'({e[0]},{e[1]})'
                position.append(pos)
            r['vertx'] = position
            d['inner_rooms'].append(r)
            d['inner_rooms_count'][type_str] += 1

        return d

    def train_json_data(self):
        d = self.dict_data()
        del d['inner_rooms']
        return json.dumps(d)

    # llamafactory训练格式,json
    def llama_factory_data_format1(self):
        r = {}
        d = self.dict_data()
        r['instruction'] = json.dumps(
            {'external_boundary': d['external_boundary'], 'inner_rooms_count': d['inner_rooms_count'], 'front_door': d['front_door']})
        r['input'] = ''
        r['output'] = json.dumps({'inner_rooms': d['inner_rooms']})
        return r

    # llamafactory训练格式,json
    def llama_factory_data_format2(self):
        r = {}
        d = self.dict_data()
        r['instruction'] = json.dumps(
            {'external_boundary': d['external_boundary'], 'inner_rooms_count': d['inner_rooms_count']})
        r['input'] = ''
        r['output'] = json.dumps({'inner_rooms': d['inner_rooms']})
        return r

    # 解析直接文本的方法
    def json2DataFormat1(str_json: str):
        loads = json.loads(str_json)
        res = []
        for i, e in enumerate(loads):
            room = RplanRoom()
            # 解析外边界
            b = []
            if len(e['instruction']) >= 2:
                arr_str = e['instruction'][1:-1]
                while ')' in arr_str and '(' in arr_str:
                    p = []

                    l1, l2 = arr_str.index('('), arr_str.index(')'),
                    arr = arr_str[l1 + 1:l2].split(',')
                    x, y = arr[0], arr[1]
                    p.append(int(x))
                    p.append(int(y))
                    # 第3位第四位暂时写成0
                    p.append(0)
                    p.append(0)
                    # 若第4位可以取到则取
                    if len(arr) > 2:
                        p[3] = 1
                    b.append(p)

                    arr_str = arr_str[l2 + 1:]

            room.external_boundary = np.array(b)
            inner_rooms_str = e['output']
            if len(inner_rooms_str) >= 2:
                arr = inner_rooms_str[1:-1].split('],')
                # 最后一个字符串包含了],需要将]去掉
                arr[-1] = arr[-1][:-1]
                for j, e in enumerate(arr):
                    inner_room = InnerRoom()
                    # idx_first_k第一逗号位置
                    idx_first_k = e.index(',')
                    type_str = e[1:idx_first_k]
                    room_type, inner_room_arr = int(type_str), e[idx_first_k + 1:].replace('(', '').replace(')',
                                                                                                            '').split(
                        ',')
                    inner_room.type = room_type
                    # 解析 (x,y),(x,y)，把(和)去掉后，再用,号隔开,每两项是一个点
                    for j in range(0, len(inner_room_arr), 2):
                        inner_room.points.append([float(inner_room_arr[j]), float(inner_room_arr[j + 1])])
                    room.inner_rooms.append(inner_room)
            res.append(room)
        return res


class InnerRoom:
    def __init__(self):
        # 房间类型
        self.type = None
        # 房间的点集合
        self.points = []
        super().__init__()

    def __str__(self):
        return f"内部房间类型:{self.type},内部房间点集:{str(self.points)}"

    def point_data(self):
        res = '[' + str(self.type) + ','
        for point in self.points:
            res += f'({point[0]},{point[1]}),'
        res = res[0:len(res) - 1]
        res += ']'
        return res

    def room_type(self):
        return self.type

    def get_points(self):
        return self.points


def json2Data(str_json: str) ->RplanRoom:
             # d = {'external_boundary': {'vertx': []},
             # 'front_door': {'vertx': []},
             # 'inner_rooms': [],
             # 'inner_rooms_count': defaultdict(int)}
    # {"inner_rooms": [{"type": "bedroom", "vertx": ["(72.0,159.5)", "(72.0,221.0)", "(128.0,221.0)", "(128.0,159.5)", "(72.0,159.5)"]}, {"type": "bathroom", "vertx": ["(72.0,128.5)", "(72.0,159.5)", "(108.0,159.5)", "(108.0,128.5)", "(72.0,128.5)"]}, {"type": "living room", "vertx": ["(165.0,87.0)", "(165.0,28.0)", "(148.0,28.0)", "(148.0,87.0)", "(125.0,87.0)", "(125.0,128.5)", "(108.0,128.5)", "(108.0,159.5)", "(128.0,159.5)", "(128.0,207.0)", "(185.0,207.0)", "(185.0,87.0)", "(165.0,87.0)"]}, {"type": "bedroom", "vertx": ["(72.0,87.0)", "(72.0,128.5)", "(125.0,128.5)", "(125.0,87.0)", "(72.0,87.0)"]}, {"type": "kitchen", "vertx": ["(115.0,53.0)", "(115.0,87.0)", "(148.0,87.0)", "(148.0,53.0)", "(115.0,53.0)"]}, {"type": "balcony", "vertx": ["(128.0,229.0)", "(185.0,229.0)", "(185.0,207.0)", "(128.0,207.0)", "(128.0,229.0)"]}]}
    loads = json.loads(str_json)
    res = []
    for i, e in enumerate(loads):
        room = RplanRoom()
        # 解析外边界
        b = []
        if len(e['instruction']) >= 2:
            arr_str = e['instruction'][1:-1]
            while ')' in arr_str and '(' in arr_str:
                p = []

                l1, l2 = arr_str.index('('), arr_str.index(')'),
                arr = arr_str[l1 + 1:l2].split(',')
                x, y = arr[0], arr[1]
                p.append(int(x))
                p.append(int(y))
                # 第3位第四位暂时写成0
                p.append(0)
                p.append(0)
                # 若第4位可以取到则取
                if len(arr) > 2:
                    p[3] = 1
                b.append(p)

                arr_str = arr_str[l2 + 1:]

        room.external_boundary = np.array(b)
        inner_rooms_str = e['output']
        if len(inner_rooms_str) >= 2:
            arr = inner_rooms_str[1:-1].split('],')
            # 最后一个字符串包含了],需要将]去掉
            arr[-1] = arr[-1][:-1]
            for j, e in enumerate(arr):
                inner_room = InnerRoom()
                # idx_first_k第一逗号位置
                idx_first_k = e.index(',')
                type_str = e[1:idx_first_k]
                room_type, inner_room_arr = int(type_str), e[idx_first_k + 1:].replace('(', '').replace(')', '').split(
                    ',')
                inner_room.type = room_type
                # 解析 (x,y),(x,y)，把(和)去掉后，再用,号隔开,每两项是一个点
                for j in range(0, len(inner_room_arr), 2):
                    inner_room.points.append([float(inner_room_arr[j]), float(inner_room_arr[j + 1])])
                room.inner_rooms.append(inner_room)
        res.append(room)
    return res


def json2InnerRooms(str_json: str):
    inner_rooms_dict = json.loads(str_json)
    rooms_array = inner_rooms_dict['inner_rooms']
    inner_rooms = []
    for i, e in enumerate(rooms_array):
        inner_room = InnerRoom()
        t = e['type']
        v = e['vertx']
        inner_room.type = room_color.get_color_room_type(t)
        points=[]
        for j,p in enumerate(v):
            # 去掉括号
            str1=p[1:-1]
            # 逗号分割
            str1_arr=str1.split(",")
            x,y=float(str1_arr[0]),float(str1_arr[1]),
            points.append([x,y])

        inner_room.points = np.array(points)
        inner_rooms.append(inner_room)
    return inner_rooms

def json2InnerRooms2(str_json: str):
    inner_rooms_dict = json.loads(str_json)
    rooms_array = inner_rooms_dict['inner_rooms']
    inner_rooms = []
    for i, e in enumerate(rooms_array):
        inner_room = InnerRoom()
        t = e['type']
        v = e['polygon']
        inner_room.type = room_color.get_color_room_type(t)
        points=[]
        for j,p in enumerate(v):
            # 去掉括号
            str1=p[1:-1]
            # 逗号分割
            str1_arr=str1.split(",")
            x,y=float(str1_arr[0]),float(str1_arr[1]),
            points.append([x,y])

        inner_room.points = np.array(points)
        inner_rooms.append(inner_room)
    return inner_rooms
