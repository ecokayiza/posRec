import torch
import cv2
import time
import posenet
import posenet.decode_multi

# 使用摄像头进行实时估计

model_id=101
CAM_ID = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720
SCALE_FACTOR = 0.7125


def cam():
    model = posenet.load_model(model_id)
    model = model.cuda()
    output_stride = model.output_stride
    
    try:
        cap = cv2.VideoCapture(0)  # 0 表示默认摄像头
        if not cap.isOpened():
            raise IOError("无法打开摄像头")
        cap.set(3, CAM_WIDTH)
        cap.set(4, CAM_HEIGHT)
    except:
        return

    start = time.time()
    frame_count = 0
    while True:
        input_image, display_image, output_scale = posenet.read_cap(
            cap, scale_factor=SCALE_FACTOR, output_stride=output_stride)

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
                min_pose_score=0.15)
        keypoint_coords *= output_scale

        overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=0.1)

        cv2.imshow('posenet', overlay_image)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break

    print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    cam()