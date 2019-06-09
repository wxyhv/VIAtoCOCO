# coding: utf-8
import datetime
import json
import os
import math
import cv2

# 根路径，里面包含images(图片文件夹)，annos.txt(bbox标注)，classes.txt(类别标签),以及annotations文件夹(如果没有则会自动创建，用于保存最后的json)
root_path = 'F:/dell/ScientificReaserch/resized/'
# 用于创建训练集或验证集
phase = 'train'
# 训练集和验证集划分的界线
split = 60

# 打开类别标签
# with open(os.path.join(root_path, 'classes.txt')) as f:
#     classes = f.read().strip().split()

# 读取images文件夹的图片名称
_indexes = [f for f in os.listdir(os.path.join(root_path))]  # _indexes = [f for f in os.listdir(os.path.join(root_path, '0'))]


# # 判断是建立训练集还是验证集
# if phase == 'train':
#     indexes = [line for i, line in enumerate(_indexes) if i <= split]
# elif phase == 'val':
#     indexes = [line for i, line in enumerate(_indexes) if i > split]


def save_jason_in_cocoformat(dataset):
    # 保存结果的文件夹
    folder = os.path.join(root_path + 'VIAtoCOCO', 'annotations')
    if not os.path.exists(folder):
        os.makedirs(folder)
    json_name = os.path.join(root_path + 'VIAtoCOCO', 'annotations/{}.json'.format(phase))
    with open(json_name, 'w') as f:
        json.dump(dataset, f)


def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    image_info = {
        "id": image_id,
        "file_name": 'COCO_train2019_' + file_name,
        "width": image_size[1],
        "height": image_size[0],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url
    }

    return image_info


def create_annotation_info(annotation_id, image_id, category_id, is_crowd,
                           area, bounding_box, segmentation):
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": is_crowd,
        "area": area,  # float
        "bbox": bounding_box,  # [x,y,width,height]
        "segmentation": segmentation  # [polygon]
    }

    return annotation_info


def get_segmenation(coord_x, coord_y):
    seg = []
    for x, y in zip(coord_x, coord_y):
        seg.append(x)
        seg.append(y)
    return [seg]

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Area:
    def get_area_of_polygon(points_x, points_y):
        points = []
        for index in range(len(points_x)):
            points.append(Point(points_x[index], points_y[index]))
        area = 0
        if len(points) < 3:
            raise Exception("error")

        p1 = points[0]
        for i in range(1, len(points) - 1):
            p2 = points[1]
            p3 = points[2]
            vecp1p2 = Point(p2.x - p1.x, p2.y - p1.y)
            vecp2p3 = Point(p3.x - p2.x, p3.y - p2.y)
            vecMult = vecp1p2.x * vecp2p3.y - vecp1p2.y * vecp2p3.x
            sign = 0
            if vecMult > 0:
                sign = 1
            elif vecMult < 0:
                sign = -1

            triarea = get_area_of_triangle(p1, p2, p3) * sign
            area += triarea
        return abs(area)

    def get_area_of_triangle(p1, p2, p3):
        area = 0
        p1p2 = get_line_length(p1, p2)
        p2p3 = get_line_length(p2, p3)
        p3p1 = get_line_length(p3, p1)
        s = (p1p2 + p2p3 + p3p1) / 2
        area = s * (s - p1p2) * (s - p2p3) * (s - p3p1)
        area = math.sqrt(area)
        return area

    def get_line_length(p1, p2):
        length = math.pow((p1.x - p2.x), 2) + math.pow((p1.y - p2.y), 2)
        length = math.sqrt(length)
        return length

def convert(imgdir, annpath):
    '''
    :param imgdir: directory for your images
    :param annpath: path for your annotations
    :return: coco_output is a dictionary of coco style which you could dump it into a json file
    as for keywords 'info','licenses','categories',you should modify them manually
    '''
    info = {
        "description": "Example Dataset",
        "url": "https://github.com/wxyhv",
        "version": "0.2.0",
        "year": 2019,
        "contributor": "wxyhv",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }
    licenses = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "https://creativecommons.org/licenses/by-nc-sa/3.0/"
        }
    ]
    categories = [
        {
            'id': 1,
            'name': 'class1',
            'super_category': 'S_Category',
        },
        {
            'id': 2,
            'name': 'class2',
            'super_category': 'S_Category',
        },
        {
            'id': 3,
            'name': 'class3',
            'super_category': 'S_Category',
        },
        {
            'id': 4,
            'name': 'class4',
            'super_category': 'S_Category',
        },

    ]
    images = []
    annotations = []

    coco_output = {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": images,
        "annotations": annotations
    }

    ann = json.load(open(annpath))
    # annotations id start from zero
    ann_id = 0
    cat_id = 0
    # dict_keys(['_via_settings', '_via_attributes', '_via_img_metadata'])
    # in VIA annotations, keys are image name

    annkey = ann['_via_img_metadata'].keys()
    for img_id, key in enumerate(annkey):  # for img_id, key in enumerate(ann['_via_img_metadata']):

        # filename = dict_get(ann, 'filename', None)
        filename = ann['_via_img_metadata'][key]['filename']

        img = cv2.imread(imgdir + '/' + filename)
        # make image info and storage it in coco_output['images']
        image_info = create_image_info(img_id, os.path.basename(filename), img.shape[:2])
        # image_info = create_image_info(filename, os.path.basename(filename), img.shape[:2])
        coco_output['images'].append(image_info)
        regions = ann['_via_img_metadata'][key]["regions"]
        # for one image ,there are many regions,they share the same img id
        for region in regions:
            cat = region['region_attributes']['name']  # + '_' + region['region_attributes']['type']
            # assert cat in ['class1', 'class2', 'class3', 'class4']
            if cat in ['class1', 'class2', 'class3', 'class4']:
                if cat == 'class1':
                    cat_id = 1
                else:
                    if cat == 'class2':
                        cat_id = 2
                    else:
                        if cat == 'class3':
                            cat_id = 3
                        else:
                            if cat == 'class4':
                                cat_id = 4
                            else:
                                cat_id = 0
            iscrowd = 0
            x = region['shape_attributes']['x']
            y = region['shape_attributes']['y']
            width = region['shape_attributes']['width']
            height = region['shape_attributes']['height']
            x = float(x)
            y = float(y)
            width = float(width)
            height = float(height)

            points_x = [x, x, x + width, x + width]
            points_y = [y, y + height, y + height, y]
            area = width * height
            ##################VIA ploygon mode use code below：#######################
            # area = Area.get_area_of_polygon(points_x, points_y)
            # min_x = min(points_x)
            # max_x = max(points_x)
            # min_y = min(points_y)
            # max_y = max(points_y)
            ################################################################
            box = [x, y, width, height]
            ##################VIA ploygon mode use code below：#######################
            # box = [min_x, min_y, max_x-min_x, max_y-min_y]
            ################################################################
            segmentation = get_segmenation(points_x, points_y)
            # make annotations info and storage it in coco_output['annotations']
            # ann_info = create_annotation_info(ann_id, filename, cat_id, iscrowd, area, box, segmentation)
            ann_info = create_annotation_info(ann_id, img_id, cat_id, iscrowd, area, box, segmentation)
            coco_output['annotations'].append(ann_info)
            ann_id = ann_id + 1

    return coco_output



def main():
    imgdir = "F:/dell/ScientificReaserch/resized/resized_img/"  # image path
    annpath = "F:/dell/ScientificReaserch/resized/" \
              "AOI_resized_via_project_10Mar2019_15h52m.json"   # annotationfile path

    save_jason_in_cocoformat(convert(imgdir, annpath))

if __name__ == '__main__':
    main()
