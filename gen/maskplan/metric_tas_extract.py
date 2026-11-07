# 参数一：外墙多边形
# 参数二：内部房间的多边形
import os
from typing import List

import numpy as np
from shapely.geometry.polygon import Polygon

from gen.maskplan.metric_polygon import calculate_coverage_and_statistics
from gen.maskplan.parse_rplan import generate_data_by_id

room_type_num = 8
room_type_dict = {
    'living_room': 0,
    'bedroom': 1,
    'kitchen': 2,
    'bathroom': 3,
    'balcony': 4,
    'storage': 5,
    'dinner_room': 6,
    'closet': 7,
}


def calculate_s(pic_id):
    import cv2
    gt_base = f'E:/workspace/python/MaskPLAN/Processed_data/img_gt2'
    pred_base = f'E:/workspace/python/MaskPLAN/Inference/Inference/Base_Deep_cross/iteration/post_19'
    pred_file = f'{pred_base}/{pic_id}.png'
    gt_file = f'{gt_base}/{pic_id}.png'
    img_pred = cv2.imread(pred_file, cv2.IMREAD_UNCHANGED)
    img_gt = cv2.imread(gt_file, cv2.IMREAD_UNCHANGED)
    gt_Svec, pred_Svec = extrac_are(img_gt,img_pred)
    return gt_Svec, pred_Svec

import numpy as np

def count_regions_and_check_connectivity(image):
    height, width, _ = image.shape
    # 记录每个颜色的所有像素位置
    color_positions = {}
    # 记录每个颜色的连通区域
    color_regions = {}
    # 记录已访问的像素
    visited = np.zeros((height, width), dtype=bool)

    # 遍历每个像素，记录颜色位置
    for i in range(height):
        for j in range(width):
            color = tuple(image[i, j])
            if color not in color_positions:
                color_positions[color] = []
            color_positions[color].append((i, j))

    # 对每个颜色进行连通区域标记
    for color in color_positions:
        positions = color_positions[color]
        regions = []
        for (i, j) in positions:
            if not visited[i, j]:
                region = []
                stack = [(i, j)]
                visited[i, j] = True
                while stack:
                    x, y = stack.pop()
                    region.append((x, y))
                    # 检查四个方向
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny] and tuple(image[nx, ny]) == color:
                            stack.append((nx, ny))
                            visited[nx, ny] = True
                regions.append(region)
        color_regions[color] = regions

    # 统计每个颜色的区域数量
    color_region_count = {color: len(regions) for color, regions in color_regions.items()}

    # 检查不同颜色之间的连通性
    color_connectivity = {}
    for color1 in color_regions:
        color_connectivity[color1] = set()
        for region1 in color_regions[color1]:
            for (i, j) in region1:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_color = tuple(image[ni, nj])
                        if neighbor_color != color1:
                            color_connectivity[color1].add(neighbor_color)

    return color_region_count, color_connectivity

def extrac_tas(pic_id):
    import cv2
    pred_base = f'E:/workspace/python/MaskPLAN/Inference/Inference/Base_Deep_cross/iteration/post_24'
    pred_file = f'{pred_base}/{pic_id}.png'
    img_pred = cv2.imread(pred_file, cv2.IMREAD_UNCHANGED)
    # 外墙，客厅living_room，浴室bathroom,储物间storage,卧室bedroom,厨房kitchen，dinner_room,阳台balcony,x
    T_list = [[79, 79, 79, 255], [229, 242, 244, 255], [252, 233, 205, 255], [189, 222, 249, 255],
              [171, 244, 253, 255], [214, 216, 234, 255], [0, 255, 0, 255], [135, 216, 208, 255],
              [127, 255, 127, 255]]
    # room_type_dict = {
    #     'living_room': 0,
    #     'bedroom': 1,
    #     'kitchen': 2,
    #     'bathroom': 3,
    #     'balcony': 4,
    #     'storage': 5,
    #     'dinner_room': 6,
    #     'closet': 7,
    # }
    color_map = {
        # living_room
        1: 0,
        # bathroom
        2: 3,
        # storage
        3: 5,
        # bedroom
        4: 1,
        # kitchen
        5: 2,

        6: 6,
        # balcony
        7: 4,

    }
    height, width = 128, 128
    color_img = np.full((128, 128), 255, dtype=np.uint8)
    # 记录每个颜色的所有像素位置
    color_positions = {}
    # 记录每个颜色的连通区域
    color_regions = {}

    pred_Svec = np.zeros(room_type_num)
    pred_Tvec = np.zeros(room_type_num)
    pred_Avec = np.zeros(room_type_num)

    gt_Svec = np.zeros(room_type_num)
    for i in range(128):
        for j in range(128):
            for k in range(len(T_list)):
                if k == 0:
                    continue
                if (img_pred[i][j][0] == T_list[k][0] and img_pred[i][j][1] == T_list[k][1] and img_pred[i][j][2] ==
                        T_list[k][2] and img_pred[i][j][3] == T_list[k][3]):
                    pred_Svec[color_map[k]] += 1
                    # 记录每个颜色位置
                    color = color_map[k]
                    if color not in color_positions:
                        color_positions[color] = []
                    color_positions[color].append((i, j))
                    color_img[i][j] = color

    # 记录已访问的像素
    visited = np.zeros((128, 128), dtype=bool)
    # 对每个颜色进行连通区域标记 (提取出每个颜色的区域)
    for color in color_positions:
        positions = color_positions[color]
        regions = []
        for (i, j) in positions:
            if not visited[i, j]:
                region = []
                stack = [(i, j)]
                visited[i, j] = True
                while stack:
                    x, y = stack.pop()
                    region.append((x, y))
                    # 检查四个方向
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny] and color_img[nx][ny] == color:
                            stack.append((nx, ny))
                            visited[nx, ny] = True
                regions.append(region)
        color_regions[color] = regions

    # 统计每个颜色的区域数量
    color_region_count = {color: len(regions) for color, regions in color_regions.items()}

    # 为每个颜色区域分配唯一ID
    region_id = 0
    # key是颜色+区域idx(比如某个颜色两个区域，那么就会有0,1两个idx)
    color_region_ids = {}  # {(color, region_idx): region_id}
    # 记录region_id对于的color
    region_id_map = dict()
    for color in color_regions:
        for idx, region in enumerate(color_regions[color]):
            color_region_ids[(color, idx)] = region_id
            region_id_map[region_id] = color
            region_id += 1

    # 构建区域邻接关系
    region_adjacency = {}  # {region_id: set(neighbor_region_ids)}
    for color in color_regions:
        for idx, region in enumerate(color_regions[color]):
            current_id = color_region_ids[(color, idx)]
            region_adjacency[current_id] = set()
            for (i, j) in region:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_color = color_img[ni][nj]
                        if neighbor_color == 255:
                            continue
                        if neighbor_color != color:
                            # 找到邻居颜色的区域ID
                            for n_idx, n_region in enumerate(color_regions[neighbor_color]):
                                if (ni, nj) in n_region:
                                    neighbor_id = color_region_ids[(neighbor_color, n_idx)]
                                    region_adjacency[current_id].add(neighbor_id)
                                    break

    # 统计房间数量
    for i in range(room_type_num):
        if i in color_region_count:
            pred_Tvec[i] = color_region_count[i]

    # 统计房间相邻
    for key in region_adjacency:
        color = region_id_map[key]
        adj_num = len(region_adjacency[key])
        pred_Avec[color] += adj_num

    t,a,s='','',''
    for i in range(room_type_num):
        t=t+f'{pred_Tvec[i]},'
        a=a+f'{pred_Avec[i]},'
        s=s+f'{pred_Svec[i]},'
    t,a,s=t[:-1],a[:-1],s[:-1]
    content = f'{t}\n{a}\n{s}'
    with open(f'E:/workdata/maskplan_tas/24/{pic_id}.txt', "w") as file:
        file.write(content)
        file.close()
    print(f'{pic_id}提取完成!')

def extrac_are(img_gt,img_pred):
    # 外墙，客厅living_room，浴室bathroom,储物间storage,卧室bedroom,厨房kitchen，dinner_room,阳台balcony,x
    T_list = [[79, 79, 79, 255], [229, 242, 244, 255], [252, 233, 205, 255], [189, 222, 249, 255],
              [171, 244, 253, 255], [214, 216, 234, 255], [0, 255, 0, 255], [135, 216, 208, 255],
              [127, 255, 127, 255]]
    # room_type_dict = {
    #     'living_room': 0,
    #     'bedroom': 1,
    #     'kitchen': 2,
    #     'bathroom': 3,
    #     'balcony': 4,
    #     'storage': 5,
    #     'dinner_room': 6,
    #     'closet': 7,
    # }
    color_map={
        #living_room
        1:0,
        #bathroom
        2:3,
        # storage
        3:5,
        # bedroom
        4:1,
        # kitchen
        5:2,

        6:6,
        # balcony
        7:4,

    }
    height, width = 128,128
    color_img=np.full((128, 128), 255, dtype=np.uint8)
    # 记录每个颜色的所有像素位置
    color_positions = {}
    # 记录每个颜色的连通区域
    color_regions = {}


    pred_Svec = np.zeros(room_type_num)
    pred_Tvec = np.zeros(room_type_num)
    pred_Avec = np.zeros(room_type_num)

    gt_Svec = np.zeros(room_type_num)
    for i in range(128):
        for j in range(128):
            for k in range(len(T_list)):
                if k == 0:
                    continue
                if (img_pred[i][j][0] == T_list[k][0] and img_pred[i][j][1] == T_list[k][1] and img_pred[i][j][2] ==
                        T_list[k][2] and img_pred[i][j][3] == T_list[k][3]):
                    pred_Svec[color_map[k]] += 1
                    # 记录每个颜色位置
                    color = color_map[k]
                    if color not in color_positions:
                        color_positions[color] = []
                    color_positions[color].append((i, j))
                    color_img[i][j]=color

                if (img_gt[i][j][0] == T_list[k][0] and img_gt[i][j][1] == T_list[k][1] and img_gt[i][j][2] ==
                        T_list[k][2] and img_gt[i][j][3] == T_list[k][3]):
                    gt_Svec[color_map[k]] += 1

    # 记录已访问的像素
    visited = np.zeros((128, 128), dtype=bool)
    # 对每个颜色进行连通区域标记 (提取出每个颜色的区域)
    for color in color_positions:
        positions = color_positions[color]
        regions = []
        for (i, j) in positions:
            if not visited[i, j]:
                region = []
                stack = [(i, j)]
                visited[i, j] = True
                while stack:
                    x, y = stack.pop()
                    region.append((x, y))
                    # 检查四个方向
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny] and color_img[nx][ny] == color:
                            stack.append((nx, ny))
                            visited[nx, ny] = True
                regions.append(region)
        color_regions[color] = regions

    # 统计每个颜色的区域数量
    color_region_count = {color: len(regions) for color, regions in color_regions.items()}

    # 为每个颜色区域分配唯一ID
    region_id = 0
    # key是颜色+区域idx(比如某个颜色两个区域，那么就会有0,1两个idx)
    color_region_ids = {}  # {(color, region_idx): region_id}
    # 记录region_id对于的color
    region_id_map=dict()
    for color in color_regions:
        for idx, region in enumerate(color_regions[color]):
            color_region_ids[(color, idx)] = region_id
            region_id_map[region_id]=color
            region_id += 1

    # 构建区域邻接关系
    region_adjacency = {}  # {region_id: set(neighbor_region_ids)}
    for color in color_regions:
        for idx, region in enumerate(color_regions[color]):
            current_id = color_region_ids[(color, idx)]
            region_adjacency[current_id] = set()
            for (i, j) in region:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_color = color_img[ni][nj]
                        if neighbor_color==255:
                            continue
                        if neighbor_color != color:
                            # 找到邻居颜色的区域ID
                            for n_idx, n_region in enumerate(color_regions[neighbor_color]):
                                if (ni, nj) in n_region:
                                    neighbor_id = color_region_ids[(neighbor_color, n_idx)]
                                    region_adjacency[current_id].add(neighbor_id)
                                    break


    # 统计房间数量
    for i in range(room_type_num):
        if i in color_region_count:
            pred_Tvec[i]=color_region_count[i]

    # 统计房间相邻
    for key in region_adjacency:
        color=region_id_map[key]
        adj_num = len(region_adjacency[key])
        pred_Avec[color]+=adj_num

    content=f'{pred_Tvec}#{pred_Avec}#{pred_Svec}'
    with open(os.path.join(out_path, str(i) + '.txt'), "w") as file:
        file.write(str(pred_Tvec))
        file.write(str(pred_Avec))
        file.write(str(pred_Avec))
        file.close()

    return gt_Svec, pred_Svec


def calculate_area_gt(wall_polygon: Polygon, room_polygons: List[Polygon], room_types: List[str]):
    # 把房间类型由字符串转换成数字
    types = []
    for _, e in enumerate(room_types):
        type_int = room_type_dict[e]
        types.append(type_int)
    room_types = types

    # Svec: 各类房间的面积总和（交于外墙）
    Svec = np.zeros(room_type_num)
    Tvec = np.zeros(room_type_num, dtype=int)
    Avec = np.zeros(room_type_num, dtype=int)

    total_room_area = 0
    total_outside_area = 0
    total_overlap_area = 0

    overlap_union_polygon = Polygon()
    outside_union_polygon = Polygon()

    for i, (poly, room_type) in enumerate(zip(room_polygons, room_types)):
        Tvec[room_type] += 1

        # 计算和墙体重叠部分，去除已统计区域
        inter_poly = wall_polygon.intersection(poly)
        new_inter_poly = inter_poly.difference(outside_union_polygon)
        inter_area = new_inter_poly.area
        if inter_area > 1e-6:
            outside_union_polygon = outside_union_polygon.union(new_inter_poly)

        # 计算房间多余部分（超出墙体），去除已统计区域
        diff_poly = poly.difference(wall_polygon)
        new_diff_poly = diff_poly.difference(outside_union_polygon)
        diff_area = new_diff_poly.area
        if diff_area > 1e-6:
            outside_union_polygon = outside_union_polygon.union(new_diff_poly)

        Svec[room_type] += poly.area
        total_room_area += inter_area
        total_outside_area += diff_area

    # 统计相邻信息和重叠面积 4846
    for i in range(len(room_polygons)):
        for j in range(i + 1, len(room_polygons)):
            poly_i = room_polygons[i]
            poly_j = room_polygons[j]

            # 判断是否有重叠面积
            inter_poly = poly_i.intersection(poly_j)
            new_overlap_poly = inter_poly.difference(overlap_union_polygon)
            inter_area = new_overlap_poly.area
            if inter_area > 1e-6:
                total_overlap_area += inter_area
                overlap_union_polygon = overlap_union_polygon.union(new_overlap_poly)

            # 判断是否相邻（边界接触）
            if poly_i.touches(poly_j):
                type_i = room_types[i]
                type_j = room_types[j]
                Avec[type_i] += 1
                Avec[type_j] += 1

    return Svec

def get_tas_by_id(pic_id):
    with open(f'E:/workdata/maskplan_tas/24/{pic_id}.txt', "r") as file:
        read = file.read()
        lines = read.split('\n')
        arr1,arr2,arr3=lines[0].split(','),lines[1].split(','),lines[2].split(',')
        pred_Tvec=np.zeros(8)
        pred_Avec=np.zeros(8)
        pred_Svec=np.zeros(8)
        for i in range(8):
            pred_Tvec[i]+=float(arr1[i])
            pred_Avec[i]+=float(arr2[i])
            pred_Svec[i]+=float(arr3[i])

    return pred_Tvec,pred_Avec,pred_Svec

if __name__ == '__main__':
    cnt = 0
    mse_T_all,mse_A_all,mse_S_all = 0,0,0
    base = f'E:/workspace/python/MaskPLAN/Inference/Inference/Base_Deep_cross/iteration/post_24'
    listdir = os.listdir(f'{base}')
    import time
    for file_name in listdir:
        try:
            pic_id=int(file_name[:-(len('.png'))])
            extrac_tas(pic_id)
        except Exception as ex:
            print(f'{file_name}提取失败:{ex}')
            continue

