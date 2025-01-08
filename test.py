import cv2
import time
import argparse
import os
import torch

import posenet
from posenet.decode_multi import *

# run model
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args, unknown = parser.parse_known_args()

## 调试用
def process_single(model,source_image,input_image):
    # input_image -> tensor -> model -> decode -> keypoints
    # source_image + keypoits -> result
    
    with torch.no_grad():
        input_image = torch.tensor(input_image).cuda()
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)
        pose_scores, keypoint_scores, keypoint_coords, pose_offsets = decode_multiple_poses(
            heatmaps_result.squeeze(0),
            offsets_result.squeeze(0),
            displacement_fwd_result.squeeze(0),
            displacement_bwd_result.squeeze(0),
            output_stride=model.output_stride,
            max_pose_detections=10,
            min_pose_score=0.25)
        keypoint_coords *= 1
    draw_image = posenet.draw_skel_and_kp(
                source_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.25, min_part_score=0.25)
    return draw_image



## demo
def process():
    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    filenames = [
        f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

    start = time.time()
    for f in filenames:
        input_image, draw_image, output_scale = posenet.read_imgfile(
            f, scale_factor=args.scale_factor, output_stride=output_stride)

        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)
            pose_scores, keypoint_scores, keypoint_coords, pose_offsets = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)

        keypoint_coords *= output_scale

        if args.output_dir:
            draw_image = posenet.draw_skel_and_kp(
                draw_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.25, min_part_score=0.25)

            cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)

        if not args.notxt:
                print()
                print("Results for image: %s" % f)
                for pi in range(len(pose_scores)):
                    if pose_scores[pi] == 0.:
                        break
                    print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                    for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                        print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))

    print('Average FPS:', len(filenames) / (time.time() - start))

 
    
if __name__ == "__main__":
    process()