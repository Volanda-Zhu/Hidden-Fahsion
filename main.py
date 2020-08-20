import sys
sys.dont_write_bytecode = True
import os
import numpy as np
import skimage.draw
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from config import Config
from model import MaskRCNN
import utils
import argparse
import time
import Dataset

class DeepFashion2Config(Config):


    NAME = "deepfashion2"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 14
    USE_MINI_MASK = True
    train_img_dir = "/home/ubuntu/dl/final/match_rcnn-master/train/image"
    train_json_path = "/home/ubuntu/dl/final/match_rcnn-master/tools/train.json"
    valid_img_dir = "/home/ubuntu/dl/final/match_rcnn-master/validation/image"
    valid_json_path = "/home/ubuntu/dl/final/match_rcnn-master/tools/valid.json"


class DeepFashion2(Dataset):
    def load_coco(self, image_dir, json_path, class_ids=None, class_map=None, return_coco=False):

        coco = COCO(json_path)
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            image_ids = list(set(image_ids))
        else:
            class_ids = sorted(coco.getCatIds())
            image_ids = list(coco.imgs.keys())

        for i in class_ids:
            self.add_class("deepfashion2", i, coco.loadCats(i)[0]["name"])
        for i in image_ids:
            self.add_image(
                "deepfashion2", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def load_keypoint(self, image_id):

        image_info = self.image_info[image_id]
        if image_info["source"] != "deepfashion2":
            return super(DeepFashion2, self).load_mask(image_id)

        instance_keypoints = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        for annotation in annotations:
            class_id = self.map_source_class_id(
                "deepfashion2.{}".format(annotation['category_id']))
            if class_id:
                keypoint = annotation['keypoints']

                instance_keypoints.append(keypoint)
                class_ids.append(class_id)

        keypoints = np.stack(instance_keypoints, axis=1)
        class_ids = np.array(class_ids, dtype=np.int32)
        return keypoints, class_ids
            
    def load_mask(self, image_id):

        image_info = self.image_info[image_id]
        if image_info["source"] != "deepfashion2":
            return super(DeepFashion2, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        for annotation in annotations:
            class_id = self.map_source_class_id(
                "deepfashion2.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                if m.max() < 1:
                    continue
                if annotation['iscrowd']:
                    class_id *= -1

                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        if not class_ids:
            return super(DeepFashion2, self).load_mask(image_id)
        else:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids




    def image_reference(self, image_id):
        super(DeepFashion2, self).image_reference(image_id)

    def annToRLE(self, ann, height, width):

        segm = ann['segmentation']
        if isinstance(segm, list):
            rle = maskUtils.merge(maskUtils.frPyObjects(segm, height, width))
        elif isinstance(segm['counts'], list):
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):

        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


def train(model, config):
    dataset_train = DeepFashion2()
    dataset_valid = DeepFashion2()

    dataset_train.load_coco(config.train_img_dir, config.train_json_path)
    dataset_valid.load_coco(config.valid_img_dir, config.valid_json_path)

    dataset_train.prepare()
    dataset_valid.prepare()

    model.train(dataset_train, dataset_valid,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads')
def color_splash(image, mask):

    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    if mask.shape[-1] > 0:
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def detect_and_color_splash(model, image_path=None):
    if image_path:

        image = skimage.io.imread(args.image)
        r = model.detect([image], verbose=1)[0]
        splash = color_splash(image, r['masks'])
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(time.time())
        skimage.io.imsave(file_name, splash)

    print("Saved to ", file_name)

if __name__ == "__main__":
    ROOT_DIR = os.path.abspath("./")
    DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


    parser = argparse.ArgumentParser(
        description='Train Match R-CNN for DeepFashion.')
    parser.add_argument("--command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')

    args = parser.parse_args()

    print("Weights: ", args.weights)
    print("Logs: ", args.logs)



    if args.command == "train":
        config = DeepFashion2Config()
    else:
        class InferenceConfig(DeepFashion2Config):

            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    if args.command == "train":
        model = MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)


    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH

        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":

        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights


    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":

        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    if args.command == "train":
        train(model, config)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command)) 


