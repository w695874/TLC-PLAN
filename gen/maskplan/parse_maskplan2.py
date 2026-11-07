import sys
import os

import cv2
from shapely.geometry import Polygon, MultiPolygon, Point, LinearRing, MultiPoint, LineString
from shapely.ops import unary_union
from shapely import affinity
import numpy as np

from gen.maskplan.metric_polygon import metric_all
from gen.maskplan.parse_rplan import generate_data_by_id


# maskplan提取多边形边框代码
# 这个方法会提取成斜边
def get_bound_pt(img):
    _, thresh2 = cv2.threshold(img, 120, 255, 0)
    contours2, _ = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    epsilion = img.shape[0] / 128
    approxes = [cv2.approxPolyDP(cnt, epsilion, True) for cnt in contours2]

    return np.squeeze(approxes[0])


def extract_rooms(img_arr,site_id):
    # T_list = [[255, 255, 255, 255], [255, 255, 0, 255], [255, 0, 255, 255], [0, 255, 255, 255],
    #           [0, 0, 255, 255], [255, 0, 0, 255], [0, 255, 0, 255], [127, 127, 255, 255], [127, 255, 127, 255]]
    # 外墙，客厅living_room，浴室bathroom,储物间storage,卧室bedroom,厨房kitchen，dinner_room,阳台balcony,x
    T_list = [[79, 79, 79, 255], [229, 242, 244, 255], [252, 233, 205, 255], [189, 222, 249, 255],
              [171, 244, 253, 255], [214, 216, 234, 255], [0, 255, 0, 255], [135, 216, 208, 255],
              [127, 255, 127, 255]]

    # 每种类型房间都初始化一个128x128矩阵
    rooms = [
        np.zeros((128, 128), dtype=np.uint8),
        np.zeros((128, 128), dtype=np.uint8), np.zeros((128, 128), dtype=np.uint8),
        np.zeros((128, 128), dtype=np.uint8), np.zeros((128, 128), dtype=np.uint8),
        np.zeros((128, 128), dtype=np.uint8), np.zeros((128, 128), dtype=np.uint8),
        np.zeros((128, 128), dtype=np.uint8), np.zeros((128, 128), dtype=np.uint8)
    ]
    # 双循环遍历，把每个类型房间的边框的值变成255后面提取
    # rooms[i]就是类型i房间对应的矩阵，属于房间的像素值是255
    for i in range(128):
        for j in range(128):
            if img_arr[i][j][0] == T_list[0][0] and img_arr[i][j][0] == T_list[0][1] and img_arr[i][j][0] == T_list[0][2]:
                rooms[0][i][j] = 255
            if img_arr[i][j][0] == T_list[1][0] and img_arr[i][j][1] == T_list[1][1] and img_arr[i][j][2] == T_list[1][2]:
                rooms[1][i][j] = 255
            if img_arr[i][j][0] == T_list[2][0] and img_arr[i][j][1] == T_list[2][1] and img_arr[i][j][2] == T_list[2][2]:
                rooms[2][i][j] = 255
            if img_arr[i][j][0] == T_list[3][0] and img_arr[i][j][1] == T_list[3][1] and img_arr[i][j][2] == T_list[3][1]:
                rooms[3][i][j] = 255
            if img_arr[i][j][0] == T_list[4][0] and img_arr[i][j][1] == T_list[4][1] and img_arr[i][j][2] == T_list[4][1]:
                rooms[4][i][j] = 255
            if img_arr[i][j][0] == T_list[5][0] and img_arr[i][j][1] == T_list[5][1] and img_arr[i][j][2] == T_list[5][1]:
                rooms[5][i][j] = 255
            if img_arr[i][j][0] == T_list[6][0] and img_arr[i][j][1] == T_list[6][1] and img_arr[i][j][2] == T_list[6][1]:
                rooms[6][i][j] = 255
            if img_arr[i][j][0] == T_list[7][0] and img_arr[i][j][1] == T_list[7][1] and img_arr[i][j][2] == T_list[7][1]:
                rooms[7][i][j] = 255
            if img_arr[i][j][0] == T_list[8][0] and img_arr[i][j][1] == T_list[8][1] and img_arr[i][j][2] == T_list[8][1]:
                rooms[8][i][j] = 255

    # maskplan项目代码提取边框
    boundary = cv2.imread('E:/workspace/python/MaskPLAN/Processed_data/img_room_sqe/0/%d.png' % (site_id), cv2.IMREAD_UNCHANGED)[:, :, -1]
    boundary[np.where(boundary > 100)] = 255
    boundary_pt = get_bound_pt(boundary)
    if boundary_pt.ndim == 1:
        boundary_pt = [[0, 0], [0, 127], [127, 127], [127, 0]]
    boundary_line = LinearRing(boundary_pt)
    boundary_poly = Polygon(boundary_line)

    # 按照提取边框代码方式提取多边形
    inner_room_polygons = []
    inner_room_types = []
    for i in range(len(T_list)):
        if i == 0:
            continue

        arr = rooms[i]
        _, thresh2 = cv2.threshold(arr, 120, 255, 0)
        contours2, _ = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        epsilion = arr.shape[0] / 128
        approxes = [cv2.approxPolyDP(cnt, epsilion, True) for cnt in contours2]
        if len(approxes) == 0:
            continue

        for _, e in enumerate(approxes):
            pt = np.squeeze(e)
            line = LinearRing(pt)
            poly = Polygon(line)
            inner_room_polygons.append(poly)
            # 客厅living_room，浴室bathroom,储物间storage,卧室bedroom,厨房kitchen，dinner_room,阳台balcony,x
            if i == 1:
                inner_room_types.append('living_room')
            elif i == 2:
                inner_room_types.append('bathroom')
            elif i == 3:
                inner_room_types.append('storage')
            elif i == 4:
                inner_room_types.append('bedroom')
            elif i == 5:
                inner_room_types.append('kitchen')
            elif i == 6:
                inner_room_types.append('dinner_room')
            elif i == 7:
                inner_room_types.append('balcony')
            elif i == 8:
                inner_room_types.append('closet')


    return boundary_poly, inner_room_types, inner_room_polygons


# 存放maskplan生成文件的路径
base_path = f'E:/workspace/python/MaskPLAN/Inference/Inference/Base_Deep_cross/iteration/post_12'
gt_path = f'E:/workspace/python/MaskPLAN/Processed_data/img_gt'

def parse_by_id(pic_id):
    file_path = f'{base_path}/{pic_id}.png'
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    boundary_poly, types_predict, rooms_predict = extract_rooms(img)
    return boundary_poly, types_predict, rooms_predict

if __name__ == "__main__":
    coverage_sum, overlap_percent_sum, outside_percent_sum = 0.0, 0.0, 0.0
    mse_T_all, mse_A_all, mse_S_all = 0, 0, 0
    cnt=0
    dir_i = os.listdir(f'E:/workdata/mask_plan/post_13')
    sum_coverage, sum_overlap_percent, sum_outside_percent, sum_cnt = 0.0, 0.0, 0.0, 0.0
    for file_name in dir_i:
        try:
            pic_id = int(file_name[:-4])
            gt_file_path = os.path.join(gt_path, file_name)
            img2 = cv2.imread(gt_file_path, cv2.IMREAD_UNCHANGED)
            wall_polygon_gt, room_types_gt, room_polygons_gt = extract_rooms(img2, pic_id)


            for i in range(13,23,1):
                bp=f'E:/workdata/mask_plan/post_{i}'
                file_path = os.path.join(bp, file_name)
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                wall_polygon_pred, room_types_pred, room_polygons_pred = extract_rooms(img, pic_id)

                mse_T, mse_A, mse_S, coverage, overlap_percent, outside_percent = metric_all(wall_polygon_gt,
                                                                                             room_polygons_gt,
                                                                                             room_types_gt,
                                                                                             wall_polygon_pred,
                                                                                             room_polygons_pred,
                                                                                             room_types_pred)

                mse_T_all += mse_T
                mse_A_all += mse_A
                mse_S_all += mse_S
                coverage_sum += coverage
                overlap_percent_sum += overlap_percent
                outside_percent_sum += outside_percent
                cnt += 1


        except Exception as e:
            print(f'解析文件出错，文件名：{file_name},error：{e}')
            continue

    print(f'avg coverage : {1.0-(coverage_sum / cnt)}')
    print(f'avg overlap_percent : {overlap_percent_sum / cnt}')
    print(f'avg outside_percent : {outside_percent_sum / cnt}')
    print(f'avg mse_T : {mse_T_all / cnt}')
    print(f'avg mse_A : {mse_A_all / cnt}')
    print(f'avg mse_S : {mse_S_all / cnt}')