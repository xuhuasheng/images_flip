
# =========================================================
# @purpose: Augmentation dataset: Flip the images and generate the corresponding XMLs
# @date：   2019/9
# @version: v1.2
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


'''
input：
    @root: 根节点  
    @childElementName: 字节点tag名称
output：
    @elements:根节点下所有符合的子元素对象    
''' 
def get_elements(root, childElementName):
    elements = root.findall(childElementName)
    return elements


'''
input：
    @root: 根节点  
    @childElementName: 字节点tag名称
output：
    @elements:根节点下第一个符合的子元素对象    
''' 
def get_element(root, childElementName):
    element = root.find(childElementName)
    return element


'''
@purpuse: 图像数据集增广，图片水平翻转，并生成对应的标注文件，  
''' 
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
        size = get_element(root, 'size')
        img_width = int(get_element(size, 'width').text)
        img_height = int(get_element(size, 'height').text)

        # 修改文件夹、文件名和路径
        get_element(root, 'folder').text = FLIPPED_IMG_PATH.split('/')[-2]
        get_element(root, 'filename').text = img_name + '_flipped.jpg'
        get_element(root, 'path').text = FLIPPED_IMG_PATH + img_name + '_flipped.jpg'

        # 遍历目标
        for obj in get_elements(root, 'object'):
            # 水平翻转前的bbox坐标
            bndbox = get_element(obj, 'bndbox')
            xmin = int(get_element(bndbox, 'xmin').text) 
            ymin = int(get_element(bndbox, 'ymin').text) 
            xmax = int(get_element(bndbox, 'xmax').text)
            ymax = int(get_element(bndbox, 'ymax').text)
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
            get_element(bndbox, 'xmin').text = str(flipped_xmin)
            get_element(bndbox, 'ymin').text = str(flipped_ymin)
            get_element(bndbox, 'xmax').text = str(flipped_xmax)
            get_element(bndbox, 'ymax').text = str(flipped_ymax)

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
        


    