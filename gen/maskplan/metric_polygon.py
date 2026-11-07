# 计算指标代码，输入多边形，参考cond_generation_new.py中的calculate_coverage_and_statistics
# 参数一：外墙多边形
# 参数二：内部房间的多边形
from typing import List

import numpy as np
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union


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

# 参数一：外墙多边形
# 参数二：内部房间的多边形
def calculate_coverage_and_statistics(wall_polygon: Polygon, room_polygons: List[Polygon], room_types: List[str]):
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

    # 统计相邻信息和重叠面积
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

    room_union = unary_union(room_polygons)

    covered_polygon = wall_polygon.intersection(room_union)

    covered_area = covered_polygon.area
    wall_area = wall_polygon.area

    coverage = covered_area / wall_area
    overlap_percent = total_overlap_area / (wall_area)
    outside_percent = total_outside_area / (wall_area + total_outside_area)

    return coverage, overlap_percent, outside_percent, Tvec, Avec, Svec



def metric_all(wall_polygon_gt, room_polygons_gt, room_types_gt, wall_polygon_pred, room_polygons_pred,
               room_types_pred):
    _, _, _, gt_Tvec, gt_Avec, gt_Svec = calculate_coverage_and_statistics(wall_polygon_gt, room_polygons_gt,
                                                                           room_types_gt)

    coverage, overlap_percent, outside_percent, pred_Tvec, pred_Avec, pred_Svec = calculate_coverage_and_statistics(
        wall_polygon_pred, room_polygons_pred, room_types_pred)

    mse_T = np.mean((pred_Tvec - gt_Tvec) ** 2)
    mse_A = np.mean((pred_Avec - gt_Avec) ** 2)
    mse_S = np.mean(((pred_Svec - gt_Svec) * (20 / 256) ** 2 * 4) ** 2)

    return mse_T, mse_A, mse_S, coverage, overlap_percent, outside_percent


