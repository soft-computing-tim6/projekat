import cv2


def display_pose(image, pose):
    colors = [
        [255, 0, 0],   # right ankle
        [255, 85, 0],  # right knee
        [255, 170, 0], # right hip
        [255, 255, 0], # left hip
        [170, 255, 0], # left knee
        [85, 255, 0],  # left ankle
        [0, 255, 0],   # pelvis
        [0, 255, 85],  # thorax
        [0, 255, 170], # upper neck
        [0, 255, 255], # head top
        [0, 170, 255], # right wrist
        [0, 85, 255],  # right elbow
        [0, 0, 255],   # right shoulder
        [85, 0, 255],  # left shoulder
        [170, 0, 255], # left elbow
        [255, 0, 255]] # left wrist

    pairs = [
        [8, 9],   # upper neck and head top
        [11, 12], # right elbow and right shoulder
        [11, 10], # right elbow and right wrist
        [2, 1],   # right hip and right knee
        [1, 0],   # right knee and right ankle
        [13, 14], # left shoulder and left elbow
        [14, 15], # left elbow and left wrist
        [3, 4],   # left hip and left knee
        [4, 5],   # left knee and left ankle
        [8, 7],   # upper neck and thorax
        [7, 6],   # thorax and pelvis
        [6, 2],   # pelvis and right hip
        [6, 3],   # pelvis and left hip
        [8, 12],  # upper neck and right shoulder
        [8, 13]]  # upper neck and left shoulder

    colors_skeleton = [
        [255, 0, 0],   # upper neck and head top
        [255, 85, 0],  # right elbow and right shoulder
        [255, 170, 0], # right elbow and right wrist
        [255, 255, 0], # right hip and right knee
        [170, 255, 0], # right knee and right ankle
        [85, 255, 0],  # left shoulder and left elbow
        [0, 255, 0],   # left elbow and left wrist
        [0, 255, 85],  # left hip and left knee
        [0, 255, 170], # left knee and left ankle
        [0, 255, 255], # upper neck and thorax
        [0, 170, 255], # thorax and pelvis
        [0, 85, 255],  # pelvis and right hip
        [0, 0, 255],   # pelvis and left hip
        [85, 0, 255],  # upper neck and right shoulder
        [170, 0, 255]] # upper neck and left shoulder

    for joint_id in range(len(pose)):
        cv2.circle(image, (pose[joint_id, 0], pose[joint_id, 1]), 3, colors[joint_id], thickness=3, lineType=8, shift=0)

    for connection_id in range(len(colors_skeleton)):
        image = cv2.line(image,
         (pose[pairs[connection_id][0],0], pose[pairs[connection_id][0],1]), # start point x, y
         (pose[pairs[connection_id][1],0], pose[pairs[connection_id][1],1]), # end point x, y
         colors_skeleton[connection_id], 3) #color and thickness

    return image