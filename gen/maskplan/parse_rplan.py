import json

import numpy as np
from shapely.geometry.polygon import Polygon

from gen.maskplan import room
from gen.maskplan.config.room_color import en_map
from gen.maskplan.room import RplanRoom


def generate_room_by_text2(input_path: str, output_path: str):
    room_res = RplanRoom()
    with open(input_path, 'r') as f1:
        d1 = json.load(f1)
        b = []
        boundary=set()
        door=set()
        for e in d1['external_boundary']['polygon']:
        # for e in d1['external_boundary']['vertx']:
            boundary.add(e)

        for e in d1['front_door']['vertex']:
        # for e in d1['front_door']['vertx']:
            door.add(e)
            flag=1
            if e in boundary:
                flag=0
            p = parse_vertx(e)
            b.append([p[0],p[1],0,flag])

        for e in d1['external_boundary']['polygon']:
        # for e in d1['external_boundary']['vertx']:
            if e in door:
                continue
            p = parse_vertx(e)
            point=[p[0],p[1],0,0]
            b.append(point)

        room_res.external_boundary = np.array(b)
        with open(output_path, 'r') as f2:
            d2 = f2.read()
            room_res.inner_rooms = room.json2InnerRooms2(d2)
            # room_res.inner_rooms = room.json2InnerRooms(d2)
            f2.close()
        f1.close()
    return room_res

def generate_polygon_info(input_path: str, output_path: str):
    room2 = generate_room_by_text2(input_path, output_path)
    room_polygons = []
    room_types = []
    for i, e in enumerate(room2.inner_rooms):
        points = []
        room_type = e.room_type()
        type = str(en_map[room_type])
        for j, e2 in enumerate(e.points):
            points.append((e2[0], e2[1]))

        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        room_polygons.append(polygon)
        room_types.append(type)

    boundary = []
    with open(input_path, 'r') as f1:
        d1 = json.load(f1)
        b = []
        door = set()
        for e in d1['front_door']['vertex']:
            door.add(e)

        for e in d1['external_boundary']['polygon']:
            if e in door:
                continue
            x, y = parse_vertx(e)
            boundary.append((x, y))

    wall_polygon = Polygon(boundary)
    return wall_polygon, room_polygons, room_types

# 解析(x,y)格式的点
def parse_vertx(vertx:str):
    # 前门的点,格式(x,y)
    # [1,-1]去掉最外面的括号,再分割中间的逗号
    p=vertx[1:-1].split(',')
    x, y = float(p[0]), float(p[1])
    return x,y

# 把结果先写入到文本增加计算速度
def generate_data_by_id(pic_id: int):
    p = f'E:/workdata/floorplan_dataset'
    input_path = f'E:/workdata/floorplan_dataset_text/no_strong/6/boundary/{str(pic_id)}.txt'
    output_path = f'E:/workdata/floorplan_dataset_text/no_strong/6/inner_rooms/{str(pic_id)}.txt'
    return generate_polygon_info(input_path,output_path)