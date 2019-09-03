
# =========================================================
# @purpose: Flip the images and generate the corresponding XMLs
# @date：   2019/9
# @version: v1.1
# @author： Xu Huasheng
# @github： https://github.com/xuhuasheng/images_flip
# =========================================================

import cv2
import os
import shutil
import xml.etree.ElementTree as ET

IMG_PATH = '/home/watson/Documents/THzDataset/train_img_rgb/'
XMLS_PATH = '/home/watson/Documents/THzDataset/annotations/train_xml'

FLIPPED_IMG_PATH = '/home/watson/Documents/THzDataset/flipped/imgs/'
FLIPPED_XMLS_PATH = '/home/watson/Documents/THzDataset/flipped/xmls/'

def get(root, name):
    vars = root.findall(name)
    return vars

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars

def img_augmentation():
    img_list = os.listdir(IMG_PATH)
    xmls_list = os.listdir(XMLS_PATH)

    cnt = 0
    for img_fileName in img_list:
        cnt += 1
        img_fullFileName = os.path.join(IMG_PATH + img_fileName) # 路径拼接，获得图片绝对路径
        img_name = img_fileName.split('.')[0] # 去掉后缀名，获得图片名字
        
        xml_name = img_name
        xml_fileName = xml_name + '.xml'

        # 判断对应xml文件是否存在
        if xml_fileName not in xmls_list:
            print('WARNING:' + xml_fileName + 'is not exist in' + XMLS_PATH)
            continue

        print('%d/%d' % (cnt, len(img_list)) + ' flipping... ' + img_fullFileName )
        origin_img = cv2.imread(img_fullFileName)   # 读取原图
        flipped_img = cv2.flip(origin_img, 1)       # 水平翻转

        # 保存翻转图片
        flipped_img_fullFileName = FLIPPED_IMG_PATH + img_name + '_flipped.jpg'
        cv2.imwrite(flipped_img_fullFileName, flipped_img)

        # 拷贝xml作为副本
        xml_fullFileName = os.path.join(XMLS_PATH, xml_fileName)
        flipped_xml_fullFileName = FLIPPED_XMLS_PATH + xml_name + '_flipped.xml'
        shutil.copyfile(xml_fullFileName, flipped_xml_fullFileName)

        # 解析xml副本
        tree = ET.parse(flipped_xml_fullFileName) # 解析xml元素树
        root = tree.getroot()                     # 获得树的根节点

        # 读取image: width & height
        size = get_and_check(root, 'size', 1)
        img_width = int(get_and_check(size, 'width', 1).text)
        img_height = int(get_and_check(size, 'height', 1).text)

        # 修改文件夹、文件名和路径
        get_and_check(root, 'folder', 1).text = FLIPPED_IMG_PATH.split('/')[-2]
        get_and_check(root, 'filename', 1).text = img_name + '_flipped.jpg'
        get_and_check(root, 'path', 1).text = FLIPPED_IMG_PATH + img_name + '_flipped.jpg'

        # 遍历目标
        for obj in get(root, 'object'):
            # 水平翻转前的bbox坐标
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) 
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) 
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            bbox_width = abs(xmax - xmin)
            bbox_height = abs(ymax - ymin)

            # 水平翻转后的bbox坐标
            flipped_xmin = img_width - xmin - bbox_width
            flipped_ymin = ymin
            flipped_xmax = img_width - xmax + bbox_width
            flipped_ymax = ymax

            # 修改xml副本
            get_and_check(bndbox, 'xmin', 1).text = str(flipped_xmin)
            get_and_check(bndbox, 'ymin', 1).text = str(flipped_ymin)
            get_and_check(bndbox, 'xmax', 1).text = str(flipped_xmax)
            get_and_check(bndbox, 'ymax', 1).text = str(flipped_ymax)

            # 显示bbox
            cv2.rectangle(origin_img, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
            cv2.rectangle(flipped_img, (flipped_xmin, flipped_ymin), (flipped_xmax, flipped_ymax), (0,255,0), 1)

        #保存修改的xml
        tree.write(flipped_xml_fullFileName) 
        # 显示
        cv2.imshow('origin_img', origin_img)
        cv2.imshow('flipped_img',flipped_img)
        cv2.waitKey(80)

if __name__ == "__main__":
    print('images augmentation start!')
    img_augmentation()
    print('images augmentation finished!')
        


    