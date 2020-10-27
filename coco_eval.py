"""eval_ssd.py
This script is for evaluating mAP (accuracy) of SSD models.  The
model being evaluated could be either a TensorFlow frozen inference
graph (pb) or a TensorRT engine.
"""


import os
import sys
import json
import argparse
import numpy as np

import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from progressbar import progressbar

from data import COCO_CLASSES, COCOAnnotationTransform, COCODetection,COCO_ROOT
import torch.utils.data as data
from ssd import build_ssd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable

COCO_change_category = ['0','1','2','3','4','5','6','7','8','9','10','11','13','14','15','16','17','18','19','20',
'21','22','23','24','25','26','27','28','31','32','33','34','35','36','37','38','39','40',
'41','42','43','44','46','47','48','49','50','51','52','53','54','55','56','57','58','59',
'60','61','62','63','64','65','67','70','72','73','74','75','76','77','78','79','80','81',
'82','84','85','86','87','88','89','90']

INPUT_HW = (300, 300)


VAL_IMGS_DIR = '/home/data/coco/images/val2014'
VAL_ANNOTATIONS =  '/home/data/coco/annotations/instances_val2014.json'
def parse_args():
    """Parse input arguments."""
    desc = 'Evaluate mAP of SSD model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--mode', type=str, default='trt',
                        choices=['tf', 'trt'])
    parser.add_argument('--imgs_dir', type=str, default=VAL_IMGS_DIR,
                        help='directory of validation images [%s]' % VAL_IMGS_DIR)
    parser.add_argument('--annotations', type=str, default=VAL_ANNOTATIONS,
                        help='groundtruth annotations [%s]' % VAL_ANNOTATIONS)
    parser.add_argument('--trained_model', default='weights/ssd300_COCO_best.pth',type=str, help='Trained state_dict file path to open')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--visual_threshold', default=0.25, type=float, help='Final confidence threshold')
    args = parser.parse_args()
    return args


def check_args(args):
    """Check and make sure command-line arguments are valid."""
    if not os.path.isdir(args.imgs_dir):
        sys.exit('%s is not a valid directory' % args.imgs_dir)
    if not os.path.isfile(args.annotations):
        sys.exit('%s is not a valid file' % args.annotations)

def _preprocess_trt(img, shape=(300, 300)):
    """Preprocess an image before TRT SSD inferencing."""
    img = cv2.resize(img, shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img *= (2.0/255.0)
    img -= 1.0
    return img

def generate_results(model, imgs_dir, jpgs, results_file,thresh,testset):
    """Run detection on each jpg and write results to file."""
    results = []
    # for jpg in progressbar(jpgs):
    #     img = cv2.imread(os.path.join(imgs_dir, jpg))
    #     image_id = int(jpg.split('.')[0].split('_')[-1])
    #     boxes, confs, clss = ssd.detect(img, conf_th=1e-2)
    #     for box, conf, cls in zip(boxes, confs, clss):
    #         x = float(box[0])
    #         y = float(box[1])
    #         w = float(box[2] - box[0] + 1)
    #         h = float(box[3] - box[1] + 1)
    #         results.append({'image_id': image_id,
    #                         'category_id': int(cls),
    #                         'bbox': [x, y, w, h],
    #                         'score': float(conf)})
    # with open(results_file, 'w') as f:
    #     f.write(json.dumps(results, indent=4))

    for idx,jpg in enumerate(jpgs):
        print(f'{idx}/{len(jpgs)}')
        img = cv2.imread(os.path.join(imgs_dir, jpg))
        image_id = int(jpg.split('.')[0].split('_')[-1])
        x = _preprocess_trt(img)
        x = torch.from_numpy(x)
        x = Variable(x.unsqueeze(0))
        x = x.cuda()
        
        with torch.no_grad():
            y = model(x)
        
        detections = y.data
        scale = torch.Tensor([img.shape[1], img.shape[0],img.shape[1], img.shape[0]])
        results = []
        for ii in range(detections.size(1)):
            j = 0
            while detections[0, ii, j, 0] >= thresh:

                score = detections[0, ii, j, 0].cpu().data.numpy()
                pt = (detections[0, ii, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])

                results.append({'image_id': str(testset.pull_anno(idx)[0]['image_id']),
                            'category_id': int(COCO_change_category[ii]),
                            'bbox': [','.join(str(c) for c in coords)],
                            'score': float(score)})
             
                    # you need to delete the last ',' of the last image output of test image
                j += 1
    with open(results_file, 'w') as f:
        f.write(json.dumps(results, indent=4))
def main():
    args = parse_args()
    check_args(args)
    
    if args.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    results_file = 'results_ssd_map.json'
    num_classes = 82 # change
    net = build_ssd('test', 300, num_classes)
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()

    testset = COCODetection(COCO_ROOT, 'val2014', None, COCOAnnotationTransform)
    if args.cuda:
        net = net.cuda()
    
    jpgs = [j for j in os.listdir(args.imgs_dir) if j.endswith('.jpg')]
    generate_results(net, args.imgs_dir, jpgs, results_file,args.visual_threshold,testset)

    # Run COCO mAP evaluation
    # Reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    cocoGt = COCO(args.annotations)
    cocoDt = cocoGt.loadRes(results_file)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    print(cocoEval.summarize())


if __name__ == '__main__':
    main()
