import os
import queue
import threading
import time
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import subprocess as sp
import dlib
import numpy as np
import pandas as pd


class Live(object):
    def __init__(self):
        self.frame_queue = queue.Queue()
        self.command = ""
        # 自行设置
        # self.rtmpUrl = "rtmp://10.168.172.44:1935/hls/pano"
        # self.rtmpUrl = "rtmp://10.168.172.44:1935/hls/robot"
        # self.rtmpUrl = "rtmp://47.103.2.122:1935/hls/www"
        self.rtmpUrl = "rtmp://47.103.2.122:1935/live/robot"
        # self.rtmpUrl = "rtmp://10.168.172.44:1935/live/robot3"
        # self.camera_path = "rtmp://36.153.37.242:21935/live/live"
        # self.camera_path = "rtmp://10.20.64.6:1935/oflaDemo/hello"
        # self.camera_path = "rtmp://10.168.172.44:1935/hls/robot"
        # self.camera_path = "rtmp://10.168.172.44:1935/live/robot"
        # self.camera_path = "rtmp://58.200.131.2:1935/livetv/hunantv"
        # self.camera_path = "rtsp://admin:admin@192.168.8.103:8554/live"
        self.camera_path = "/home/user/Videos/vlc1.mp4"
        # self.camera_path = "rtmp://202.69.69.180:443/webcast/bshdlive-pc"
        # self.camera_path = "rtmp://media3.sinovision.net:1935/live/livestream"
        # self.camera_path = "rtmp://mobliestream.c3tv.com:554/live/goodtv.sdp"
        # self.camera_path = "data/movies/1.mp4"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
        self.features_known_arr = read_csv()
        self.facerec = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

    def read_frame(self):
        print("开启推流")
        cap = cv2.VideoCapture(self.camera_path)
        # Get video information
        # fps = int(cap.get(cv2.CAP_PROP_FPS))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # fps = 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("width==", width)
        print("weight==", height)
        """
        ffmpeg -y -hwaccel cuvid -c:v h264_cuvid  -i 1.mp4  -vf scale_npp=480:360   -vcodec h264_nvenc -preset slow -f flv -profile:v high rtmp://47.103.2.122:1935/live/robot
        ffmpeg -y -hwaccel cuvid -c:v h264_cuvid  -i rtmp://10.20.64.6:1935/oflaDemo/hello   -vcodec h264_nvenc -preset slow -f flv -profile:v high rtmp://47.103.2.122:1935/live/robot

        """
        self.command = ['ffmpeg',
                        # '-use_wallclock_as_timestamps',
                        # '1',
                        #          '-re',
                        '-y',
                        # '-hwaccel', 'cuvid',
                        '-hwaccel_device', "cuvid",
                        # '-c:v', "h264_cuvid",
                        '-f', 'rawvideo',
                        # '-f',"mpeg1video",
                        '-hwaccel', 'cuvid',
                        # '-f', 'dshow ',
                        '-c:v', "h264_cuvid",
                        # '-hwaccel_device', "cuvid",
                        '-vcodec', 'rawvideo',
                        # '-vcodec', 'copy',
                        # '-vcodec', 'h264_cuvid',
                        #    '-vcodec', 'h264_nvenc',
                        '-pix_fmt', 'bgr24',
                        # '-pix_fmt', 'libx264',
                        # '-pix_fmt', 'yuv420p',
                        '-s', "{}x{}".format(width, height),
                        # '-r', str(fps),

                        '-i', '-',
                        # '-f', 'rawvideo',
                        # '-hwaccel', 'cuvid',
                        # '-c:v', 'h264_cuvid',
                        # '-acodec','aac',
                        '-c:v', 'h264_nvenc',
                        # '-a:v', 'aac',
                        # '-c:v', 'h264_nvenc',
                        # '-vcodec', 'h264_nvenc',

                        # '-c:v', 'libx264',
                        # '-pix_fmt', 'yuv420p',
                        # '-pix_fmt', 'libx264',

                        # '-b:v', '800K',
                        '-pix_fmt', 'bgr24',
                        # '-pix_fmt', 'yuv420p',
                        # '-preset', 'slow',

                        # '-preset', 'ultrafast',
                        # '-preset', 'fast',
                        # '-preset', 'slow',
                        # '-preset', 'faster',
                        # '-f', 'hls',

                        '-f', 'flv',
                        # '-f', 'hevc',
                        #             '-profile:v','high',
                        #             '-aspect', '16:9',
                        # '-f', 'h264',
                        # '-hls_list_size', '9',
                        # '-hls_time','2',
                        # '/home/robotech/video/hls/www/robot/',
                        self.rtmpUrl]
        while not cap.isOpened():
            ret, frame = cap.read("data/pano_error.png")
            self.frame_queue.put(frame)

        start_time = time.time()
        while (cap.isOpened()):
            if time.time() - start_time > 60:
                cap = cv2.VideoCapture(self.camera_path)
            ret, frame = cap.read()
            if not ret:
                cap = cv2.VideoCapture(self.camera_path)
                print("Opening camera is failed")
                continue

            self.frame_queue.put(frame)


    def push_frame(self):
        # 防止多线程时 command 未被设置
        while True:
            if len(self.command) > 0:
                # 管道配置
                p = sp.Popen(self.command, stdin=sp.PIPE)
                break
        _index = 0
        face_count = 0
        faces = []
        t_start = time.time()
        while True:

            if self.frame_queue.empty() != True:
                _a = time.time()
                frame = self.frame_queue.get()


                if time.time() - t_start > 0.4:  # 因为识别速度和视频帧率相差过大，为了使输出图像与摄像头输入保持同步，所以每两秒输出一次识别结果。该参数可以根据计算性能加以调整
                    t_start = time.time()
                    faces = self.detector(frame, 0)
                    if len(faces) != 0:
                        frame = self.analysis_frame(faces, frame, self.predictor)


                p.stdin.write(frame.tostring())
                _b = time.time()
                _index = _index + 1

    def analysis_frame(self, faces, img_rd, predictor):
        """
        对获取到的人脸数据进行分析
        :return:
        """
        # detector, predictor = load_face_reco()
        # features_known_arr = read_csv()
        # print("开始分析人脸")
        # _a = time.time()
        font = cv2.FONT_ITALIC
        pos_namelist = []
        name_namelist = []
        if len(faces) != 0:
            # 4. 获取当前捕获到的图像的所有人脸的特征，存储到 features_cap_arr
            # 4. Get the features captured and save into features_cap_arr
            features_cap_arr = []
            for i in range(len(faces)):
                shape = predictor(img_rd, faces[i])
                features_cap_arr.append(self.facerec.compute_face_descriptor(img_rd, shape))

            # 5. 遍历捕获到的图像中所有的人脸
            # 5. Traversal all the faces in the database
            for k in range(len(faces)):
                # print("##### camera person", k + 1, "#####")
                # 让人名跟随在矩形框的下方
                # 确定人名的位置坐标
                # 先默认所有人不认识，是 unknown
                # Set the default names of faces with "unknown"
                # name_namelist.append("unknown")
                name_namelist.append("VIP")

                # 每个捕获人脸的名字坐标 the positions of faces captured
                pos_namelist.append(
                    tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                # # 对于某张人脸，遍历所有存储的人脸特征
                # # For every faces detected, compare the faces in the database
                e_distance_list = []
                for i in range(len(self.features_known_arr)):
                    # 如果 person_X 数据不为空
                    if str(self.features_known_arr[i][0]) != '0.0':
                        # print("with person", str(i + 1), "the e distance: ", end='')
                        e_distance_tmp = return_euclidean_distance(features_cap_arr[k], self.features_known_arr[i])
                        # print(e_distance_tmp)
                        e_distance_list.append(e_distance_tmp)
                    else:
                        # 空数据 person_X
                        e_distance_list.append(999999999)
                # # Find the one with minimum e distance
                similar_person_num = e_distance_list.index(min(e_distance_list))
                # print("Minimum e distance with person", int(similar_person_num) + 1)
                #
                if min(e_distance_list) < 0.4:
                    ####### 在这里修改 person_1, person_2 ... 的名字 ########
                    # 可以在这里改称 Jack, Tom and others
                    # Here you can modify the names shown on the camera
                    name_namelist[k] = "Person " + str(int(similar_person_num) + 1)
                    # print("May be person " + str(int(similar_person_num) + 1))
                else:
                    # print("Unknown person")
                    pass
            for kk, d in enumerate(faces):
                # 绘制矩形框
                cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 0),
                              2)
                # print('\n')

            # 6. 在人脸框下面写人脸名字
            # 6. write names under rectangle
            for i in range(len(faces)):
                # cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 1, (255, 255, 0), 1, 25)

        # cv2.putText(img_rd, "Press 'q': Quit", (20, 450), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
        # cv2.putText(img_rd, "Face Recognition", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        # _b = time.time()
        # print("人脸识别完成", _b - _a)
        return img_rd

    def run(self):
        threads = [
            threading.Thread(target=Live.read_frame, args=(self,)),
            threading.Thread(target=Live.push_frame, args=(self,))
        ]
        [thread.start() for thread in threads]
        [thread.setDaemon(True) for thread in threads]

def read_csv():
    """
    只需要加载一次，读取人脸数据csv（csv因为人员缘故，一般不会改变）
    :return:
    """
    if os.path.exists("data/features_all.csv"):
        path_features_known_csv = "data/features_all.csv"
        csv_rd = pd.read_csv(path_features_known_csv, header=None)

        # 用来存放所有录入人脸特征的数组
        # The array to save the features of faces in the database
        features_known_arr = []

        # 2. 读取已知人脸数据
        # Print known faces
        for i in range(csv_rd.shape[0]):
            features_someone_arr = []
            for j in range(0, len(csv_rd.ix[i, :])):
                features_someone_arr.append(csv_rd.ix[i, :][j])
            features_known_arr.append(features_someone_arr)
        # print("Faces in Database：", len(features_known_arr))
        return features_known_arr

# 计算两个128D向量间的欧式距离
# Compute the e-distance between two 128D features
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return dist


if __name__ == "__main__":
    live = Live()
    live.run()
