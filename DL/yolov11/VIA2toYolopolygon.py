### For VGG Image Annotator 2 (2.0.12)
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

def VIA2YOLO_mask(annotations_path, savefolder, imagepath):
    annotations = json.load(open(annotations_path)) # dict
    # print(annotations)

    ### image id layer (key)
    # for key in annotations.keys():
    #     print(key)
    #     print(annotations[key])

    annotations = list(annotations.values())
    # print(annotations[0])
    # annotations = [a for a in annotations if a['regions']]
    # print(annotations)

    ### each image
    for a in annotations:
        # print(a)
        # print(type(a['regions']))
        # print(a['regions'])

        file = open(os.path.join(savefolder, str(a['filename'][:-4])+ ".txt"), "w")
        images = os.path.join(imagepath, a['filename'])
        image = cv2.imread(images)
        img_h, img_w = image.shape[:2]

        polygons=[]
        num_ids=[]

        ### Find each region's coordinate and label (polygon)
        for i in range(len(a['regions'])):
            # print(a['regions'][i])
            
            polygons.append(a['regions'][i]['shape_attributes'])
            # print(polygons)

            # print(a['regions'][i]['region_attributes'])
            n = a['regions'][i]['region_attributes']
            
            # sumn=0
            for k, v in n.items():
                if len(v) != 0:
                    # print(k)
                    try:
                        ### Change classes for your own dataset
                        if k == 'person':
                            num_ids.append(0)
                        elif k == 'balloon':
                            num_ids.append(1)
                        # elif k == 'goose':
                        #     num_ids.append(2)
                        # elif k == 'bovine':
                        #     num_ids.append(3)
                        # elif k == 'Chicken':
                        #     num_ids.append(4)
                        else:
                            print(i, "annotation lost")
                    except:
                        pass
            # print(num_ids)
            # sumn+=1

        idSUM = 0

        ### Change to yolov7-mask label
        if (num_ids != []):
            #f.write("dataset\\" + str(a['filename']) )
            for g in polygons:
                # print(g)
                
                norX = [x/img_w for x in g['all_points_x']]
                norY = [x/img_h for x in g['all_points_y']]
                newlist = [e for t in zip(norX, norY) for e in t]
                #print(newlist)

                file.write(str([num_ids][0][idSUM])+ ' ' + str(newlist)[1:-1].replace(',', '') )
                file.write('\n')
                # f.write('\r\n')

                idSUM += 1
        file.close()
        print(a['filename'], num_ids, " region_number:", len(num_ids))
    return num_ids

if __name__=='__main__':
    # annotations_path = "C:/Users/calmnight/python3/VIA2Yolov7/via_region_data.json"
    annotations_path = "C:/Users/calmnight/python3/10402AI_course/DL/yolov11/dataset/balloon_dataset/segmentation/val.json"
    savefolder = 'C:/Users/calmnight/python3/10402AI_course/DL/yolov11/dataset/balloon_dataset/segmentation/temp_annotation/'
    imagepath = "C:/Users/calmnight/python3/10402AI_course/DL/yolov11/dataset/balloon_dataset/segmentation/val/"
    num_ids = VIA2YOLO_mask(annotations_path, savefolder, imagepath)
    