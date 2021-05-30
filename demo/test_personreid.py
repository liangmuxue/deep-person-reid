from copy import deepcopy

import chainer
import torch, os
import cv2
import argparse
from PIL import Image
from torch import nn, optim
import base64
from mmfashion.utils.image import get_img_file_tensor
from mmfashion.utils.checkpoint import load_checkpoint
from mmfashion.models import build_retriever
from alphapose_part.infer import load_pose_model
from scipy.spatial.distance import cosine
from detection_visualizer import append_letter_img
from detector.apis import get_detector
from alphapose_part.infer import inference_keypoints_alp_single
from keypoint_visualizer import build_joint_connections_line, add_joint_connections_with_lines
from model import PCB, ClassBlock
from predict import load_config, estimate
from test_retriever import get_q_feature
from scripts.realtime_tracker import RealtimeTracker
from utils_pose import rebuild_pose_cfg, get_scale_rate, xywh_to_x1y1x2y2, x1y1x2y2_to_xywh
from utils_transform import resize_and_pad
from vis import *
from query_datamanager import getScoreByDismat
from scripts import realtime_tracker
from person_reid.query_datamanager import QueryDataManager, dis_compute
from person_reid.reid_data_service import ReidDataService
from vis import getTime
from mmcv import Config
import random
from visualize_pose_matching import Pose_Matcher
from ymlconfig import update_config
from detector.yolo_api import YOLODetector

from chainer_pose.model import PoseProposalNet
from chainer_pose.utils import parse_size
from chainer_pose.predict import create_model, load_config, get_feature, get_humans_by_feature

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
# parser.add_argument('--pose_cfg', type=str, required=True,
#                     help='pose inference configure file name')
args = parser.parse_args()

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMFashion In-shop Clothes Retriever Demo')
    parser.add_argument(
        '--input',
        type=str,
        help='input image path',
        default='demo/imgs/ai_fashion/out1.png')
    parser.add_argument(
        '--topk', type=int, default=5, help='retrieve topk items')
    parser.add_argument(
        '--config',
        help='train config file path',
        default='mmfashion/configs/retriever_in_shop/global_retriever_vgg_loss_id.py')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='mmfashion/checkpoint/InshopRetrieval/vgg16_gp.pth',
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--use_cuda', type=bool, default=True, help='use gpu or not')
    args = parser.parse_args()
    return args


def test_load_disk_data():
    from detector.yolo_api import YOLODetector
    from detector.yolo_cfg import cfg
    cfg.gal_filepath = "data/torchreid"
    det = get_detector(cfg)
    pose_model, p_cfg = _load_pose_model()
    datamanager = QueryDataManager(
        cfg=cfg,
        detector=det,
        pose_model=pose_model["model"]
    )
    datamanager.load_gallery_from_disk()


def test_add_data():
    pose_mode = "alp"
    args = parser.parse_args()
    cfg = update_config(args.cfg)
    cfg.detector = "yolo"
    det = get_detector(cfg)
    pose_model, p_cfg = _load_pose_model(pose_mode=pose_mode)
    datamanager = QueryDataManager(
        cfg=cfg,
        detector=det,
        pose_model=pose_model,
        pose_mode=pose_mode,
        result_dir='./person_reid/train/result_newds2',
        train_classnum=10,
        which_epoch=59,
    )
    feature_persist_path = "home/bavon/data/torchreid/storage/feature_tensors.h5"
    reid_service = ReidDataService(datamanager, feature_persist_path)
    reid_service.load_persons()
    # reid_service.load_features_data()
    img_list = []
    img_list.append("/home/bavon/model/datasets/ourds/data/E0001/E0001C10T0001F005.png")
    img_list.append("/home/bavon/model/datasets/ourds/data/E0002/E0002C10T0002F008.png")
    img_list.append("/home/bavon/model/datasets/ourds/data/E0003/E0003C10T0003F002.png")
    img_list.append("/home/bavon/model/datasets/ourds/data/E0004/E0004C10T0004F002.png")
    img_list.append("/home/bavon/model/datasets/ourds/data/E0005/E0005C10T0005F002.png")
    img_list.append("/home/bavon/model/datasets/ourds/data/E0006/E0006C10T0006F002.png")
    img_list.append("/home/bavon/model/datasets/ourds/data/E0007/E0007C10T0007F001.png")
    img_list.append("/home/bavon/model/datasets/ourds/data/E0008/E0008C10T0008F002.png")
    img_list.append("/home/bavon/model/datasets/ourds/data/E0009/E0009C10T0009F003.png")
    img_list.append("/home/bavon/model/datasets/ourds/data/E0010/E0010C10T0010F003.png")
    for index, img in enumerate(img_list):
        reid_service.add_gallery_imagedata(img, index + 1)
    # reid_service.add_gallery_imagedata(img3, 3)


def test_load_data():
    pose_mode = "chainer"
    from detector.yolo_cfg import cfg
    cfg.gal_filepath = "data/torchreid"
    det = get_detector(cfg)
    pose_model, p_cfg = _load_pose_model()
    datamanager = QueryDataManager(
        cfg=cfg,
        detector=det,
        pose_model=pose_model,
        pose_mode=pose_mode,
        result_dir='/home/bavon/project/ai-demo/person_reid/train/result_newds58fc',
        train_classnum=48,
        which_epoch=59,
    )
    feature_persist_path = "data/torchreid/storage/feature_tensors"
    reid_service = ReidDataService(datamanager, feature_persist_path)
    # reid_service.load_persons()
    # reid_service.load_person_images()
    reid_service.load_features_data()
    print("load over")


def test_img_match():
    pose_mode = "alp"
    args = parser.parse_args()
    cfg = update_config(args.cfg)
    cfg.detector = "yolo"
    det = get_detector(cfg)
    pose_model, p_cfg = _load_pose_model(pose_mode=pose_mode)
    datamanager = QueryDataManager(
        cfg=cfg,
        detector=det,
        pose_model=pose_model,
        pose_mode=pose_mode,
        result_dir='/home/bavon/project/ai-demo/person_reid/train/result_newds58fc',
        train_classnum=58,
        which_epoch=59,
        # outer_model=(model, global_classifier, PCB_classifier)
    )
    feature_persist_path = "data/torchreid/storage/feature_tensors"
    reid_service = ReidDataService(datamanager, feature_persist_path)
    reid_service.load_persons()
    reid_service.load_person_images()
    reid_service.load_features_data()
    print("load over")
    q_img_filepath = "/home/bavon/project/ai-demo/data/torchreid/query_test/lcy_crop.png"
    # q_img_filepath = "/home/bavon/model/datasets/ourds/data/E0002/E0002C10T0002F002.png"
    # q_img_filepath = "/home/bavon/model/datasets/ourds/data/E0002/E0002C11T0012F027.png"
    img_data = cv2.imread(q_img_filepath)
    top_index, top_distmat = reid_service.match_person_image(img_data)
    print("top_index:{}, top_distmat:{}".format(top_index, top_distmat))


def test_pose_get(image=None):
    from detector.yolo_api import YOLODetector
    from detector.yolo_cfg import cfg
    cfg.gal_filepath = "data/torchreid"
    det = get_detector(cfg)
    pose_model, p_cfg = _load_pose_model()
    datamanager = QueryDataManager(
        cfg=cfg,
        detector=det,
        pose_model=pose_model["model"]
    )
    for i in range(10):
        ckpt_time = getTime()
        if image is None:
            q_img_filepath = "/home/bavon/project/ai-demo/data/torchreid/query_test/weimin2.png"
            # q_img_filepath = "/home/bavon/project/ai-demo/data/torchreid/query_test/0001_c2s1_000301_00.jpg"
            f = open(q_img_filepath, mode='rb');
            img_bas64data = f.read()
            global_features, partical_features = datamanager.get_image_feature(img_bas64data, mode=1)
        else:
            global_features, partical_features = datamanager.get_image_feature(image, mode=3)
        ckpt_time, pose_time = getTime(ckpt_time)
        print("feature get time:{}".format(pose_time))
    # print("global_features:{}, partical_features:{}".format(global_features, partical_features))


def _load_pose_model(pose_mode="alp"):
    cfg = update_config(args.cfg)
    config = load_config(cfg)
    if pose_mode == 'alp':
        g_pose_model = load_pose_model()
        g_pose_model = {"model": g_pose_model}
    else:
        g_pose_model = create_model(cfg, config)
    p_cfg = {"detection_thresh": cfg.pose_match.detection_thresh,
             "high_detection_thresh": cfg.pose_match.high_detection_thresh,
             "min_num_hs_keypoints": cfg.pose_match.min_num_hs_keypoints,
             "min_num_keypoints": cfg.pose_match.min_num_keypoints,
             "img_scale_size_x": cfg.img_scale_size_x, "img_scale_size_y": cfg.img_scale_size_y}
    rebuild_pose_cfg(p_cfg)
    return g_pose_model, p_cfg

def test_compare_img():
    pose_mode = "alp"
    args = parser.parse_args()
    cfg = update_config(args.cfg)
    cfg.detector = "yolo"
    det = get_detector(cfg)
    pose_model, p_cfg = _load_pose_model(pose_mode=pose_mode)
    # model, global_classifier, PCB_classifier, optimizer = init_network()
    # model, global_classifier, PCB_classifier = pre_load_network(model, global_classifier, PCB_classifier)

    datamanager = QueryDataManager(
        cfg=cfg,
        detector=det,
        pose_model=pose_model,
        pose_mode=pose_mode,
        #result_dir='/home/bavon/project/deepface-reid-remote/person_reid/train/person_reid/train/2020-0316-pf-0.8',
        result_dir='/home/bavon/model/market_ckp',
        train_classnum=751,
        which_epoch=59,
        # outer_model=(model, global_classifier, PCB_classifier
    )

    # img1_filepath = "/home/bavon/project/ai-demo/data/torchreid/query_test/jiashuhan1.jpg"
    # img1_filepath = "/home/bavon/model/datasets/mars/bbox_train/E0101/E0101C12T0021F016.jpg"
    # img1_filepath = "/home/bavon/project/ai-demo/data/torchreid/query_test/wangshuai1.jpg"
    # img1_filepath = "/home/bavon/model/datasets/mars_ext/bbox_train/E0001/E0001C10T0001F003.jpg"
    # img1_filepath = "/home/bavon/model/datasets/ourds/bbox_train/E0001/E0001C10T0001F003.jpg"
    # img1_filepath = "/home/bavon/face_test/reid/20201222-160745-642261186-1273073-full.jpg"
    # img1_filepath = "/home/bavon/face_test/reid/wangshuai2.jpg"
    # img1_filepath = "/home/bavon/model/datasets/ziming.jpg"
    # img1_filepath = "/home/bavon/model/datasets/miaozhuang1.jpg"
    # img1_filepath = "/home/bavon/model/datasets/yangyi1.jpg"
    img1_filepath = "/home/bavon/model/datasets/duibi/cn.jpg"
    # img1_filepath = "/home/bavon/model/datasets/haoran1.jpg"
    # img1_filepath = "/home/bavon/project/ai-demo/data/torchreid/query_test/jiashuhan1.jpg"
    # img1_filepath = "/home/bavon/model/datasets/ourds/data/E0002/E0002C11T0012F027.png"
    # img1_filepath = "/home/bavon/project/ai-demo/data/torchreid/query_test/wangshuai1.jpg"
    # img1_filepath = "/home/bavon/model/datasets/mars_ext/bbox_train/E0001/E0001C10T0001F003.jpg"
    # img1_filepath = "/home/bavon/model/datasets/ourds/bbox_train/E0001/E0001C10T0001F003.jpg"
    # img1_filepath = "/home/bavon/model/datasets/mars/bbox_train/E0002/E0002C11T0012F027.jpg"
    # img1_filepath = "/home/bavon/model/datasets/mars_ext/bbox_train/E0003/E0003C10T0003F001.jpg"
    # img1_filepath = '/home/bavon/model/datasets/mars/bbox_train/0001/0001C1T0001F001.jpg'
    # img1_filepath = '/home/bavon/model/datasets/mars/bbox_test/0006/0006C2T0002F096.jpg'
    # img1_filepath = '/home/bavon/model/datasets/Market-1501-v15.09.15/bounding_box_test/1498_c6s3_088642_01.jpg'
    # img1_filepath = "/home/bavon/project/ai-demo/data/torchreid/query_test/zhangao_crop.png"
    # img1_filepath = "/home/bavon/project/ai-demo/data/torchreid/query_test/lishuangchao.jpg"
    # img1_filepath = "/home/bavon/project/ai-demo/data/torchreid/query_test/mengge.jpg"
    # img1_filepath = "/home/bavon/project/ai-demo/data/torchreid/query_test/weimin1.png"
    # img2_filepath = "/home/bavon/model/datasets/zimingrenti.jpg"
    # img2_filepath = "/home/bavon/model/datasets/dingchangrenti.jpg"
    # img2_filepath = "/home/bavon/model/datasets/zhangao2.jpg"
    # img2_filepath = "/home/bavon/model/datasets/yangyi2.jpg"
    img2_filepath = "/home/bavon/model/datasets/duibi/miaozhuang88.jpg"
    # img2_filepath = "/home/bavon/model/datasetshaoran2.jpg"
    # img2_filepath = "/home/bavon/model/datasets/nv2.jpg"
    # img2_filepath = "/home/bavon/model/datasets/shuangchao2.jpg"
    # img2_filepath = "/home/bavon/face_test/reid/854_1.jpg"
    # img2_filepath = "/home/bavon/model/datasets/ourds/bbox_train/E0002/E0002C10T0002F021.jpg"
    # img2_filepath = "/home/bavon/model/datasets/mars/bbox_train/E0101/E0101C12T0021F002.jpg"
    # img2_filepath = "/home/bavon/model/datasets/mars/bbox_train/E0002/E0002C11T0012F008.jpg"
    # img2_filepath = "/home/bavon/model/datasets/ourds_gan/bbox_train/E0002/G0_E0002C10T0002F003.jpg"
    # img2_filepath = "/home/bavon/model/datasets/mars_ext/bbox_train/E0003/E0003C10T0003F005.jpg"
    # img2_filepath = "/home/bavon/project/ai-demo/data/torchreid/query_test/jiashuhan2.jpg"
    # img2_filepath = "/home/bavon/project/ai-demo/data/torchreid/query_test/wangshuai-standard.jpg"

    img1 = cv2.imread(img1_filepath)
    with torch.no_grad():
        gf1, pf1, scores, keypoints = inference_keypoints_alp_single(datamanager, img1, keypoint=None)
    # img1 = resize_and_pad(img1, (256, 128), 127)
    # img2 = resize_and_pad(img2, (256, 128), 127)
    # vis_data(img1,None)
    directory_name = '/home/bavon/model/datasets/test_20'
    for filename in os.listdir(directory_name):
        # print(filename) #just for test
        # img is used to store the image data
        img2 = cv2.imread(directory_name + "/" + filename)
        with torch.no_grad():
            gf2, pf2, scores, keypoints = inference_keypoints_alp_single(datamanager, img2, keypoint=None)
        distmat_1 = dis_compute(pf1, gf1, pf2, gf2)
        print("dismat:{}".format(distmat_1))
    # gf2_fea = gf2.squeeze(0).cpu().numpy()
    # # show_features_single(gf2_fea, name="test_comp_feature")
    # # pf2_fea = pf2.squeeze(0).cpu().numpy().transpose(1,0)
    # # show_features(pf2_fea,name="test_comp_feature")



def image_test():
    from PIL import Image
    from mmfashion.utils.image import get_img_tensor
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # img_path = "/home/bavon/model/datasets/duibi/1120.jpg"
    # img = Image.open(img_path)
    img_b64encode = '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'
    img_b64decode = base64.b64decode(img_b64encode)
    img_array = np.fromstring(img_b64decode, np.uint8)
    raw_img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)
    raw_img2 = Image.fromarray(np.uint8(raw_img))
    tensor =get_img_tensor(raw_img2, True, get_size=False)
    print(tensor)

def test_keypoints_get():
    pose_mode = "alp"
    args = parser.parse_args()
    cfg = update_config(args.cfg)
    cfg.detector = "yolo"
    det = get_detector(cfg)
    pose_model, p_cfg = _load_pose_model(pose_mode=pose_mode)
    # model, global_classifier, PCB_classifier, optimizer = init_network()
    # model, global_classifier, PCB_classifier = pre_load_network(model, global_classifier, PCB_classifier)

    datamanager = QueryDataManager(
        cfg=cfg,
        detector=det,
        pose_model=pose_model,
        pose_mode=pose_mode,
        result_dir='/home/bavon/model/market_ckp',
        train_classnum=6,
        which_epoch=59,
        # outer_model=(model, global_classifier, PCB_classifier)
    )
    img_filepath = "/home/bavon/face_test/reid/20201222-160745-642261186-1273073-full.jpg"
    img_filepath = "/home/bavon/model/datasets/duibi/1708.jpg"
    # img_filepath = "/home/bavon/project/ai-demo/data/torchreid/query_test/jiashuhan1.jpg"
    # img_filepath = "/home/bavon/face_test/reid/wangshuai2.jpg"
    img = cv2.imread(img_filepath)
    keypoint_result, preds_scores, bboxes = datamanager.get_keypoints_alp_multi(img)
    print("preds_scores len:{}".format(len(preds_scores)))
    if bboxes is None:
        print("get data none")
        return
    for i, preds_score in enumerate(preds_scores):
        preds_score = preds_score.squeeze(-1).numpy().tolist()
        flag = RealtimeTracker(data_manager=datamanager, cfg=cfg).keypoint_score_fit(preds_score, 5, 0.2)
        body = crop_candicates(img, bboxes[i])
        visdom_img_data(body, title="det_{}".format(i), cap="flag:{},{}".format(flag,preds_score))


def test_pose_crop_get():
    image = test_cv_crop()
    test_pose_get(image)


def test_cv_crop():
    input_source = "/home/bavon/project/ai-demo/data/demo/real_0814.mp4"
    stream = cv2.VideoCapture(input_source)
    flag, img = stream.read()
    # vis_data(img,None)
    img1 = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
    # vis_data(img1, None)
    return img1


def test_img_letter():
    img1_filepath = ""
    img1 = cv2.imread(img1_filepath)
    letter = "zhangao,bend"
    position = (120, 100)
    append_letter_img(img1, position, letter, letter_dict=None)


################# pose relation #################
def test_chainer_keypoints_get():
    img1_filepath = "/home/bavon/project/ai-demo/data/torchreid/query_test/jiashuhan_front.jpg"
    # img1_filepath = "/home/bavon/face_test/full_img/192.168.0.122_01_20201225141753283_TIMING.jpg"
    img1_filepath = "/home/vsftp_pas/20201035/192.168.0.122_01_20210105110254595_TIMING.jpg"
    img1_filepath = "/home/bavon/model/datasets/tracker_full/frame-051.jpeg"
    img1_filepath = "/home/bavon/face_test/reid/por_t1.jpg"
    img1 = cv2.imread(img1_filepath)
    height, width, ch = img1.shape

    img_scale_size_x = img_scale_size_y = 384

    from utils_pose import scale_transfer, rebuild_pose_cfg
    p_cfg = {"real_width": width, "real_height": height, "img_scale_size_x": img_scale_size_x,
             "img_scale_size_y": img_scale_size_y}
    rebuild_pose_cfg(p_cfg)

    small_img = cv2.resize(img1, (img_scale_size_x, img_scale_size_y),
                           interpolation=cv2.INTER_CUBIC)
    image = small_img.transpose(2, 0, 1).astype(np.float32)
    pose_model, config = _load_pose_model(pose_mode="chainer")
    humans, humans_points, preds_scores = estimate(pose_model, image)
    for i, human in enumerate(humans):
        human_wh = x1y1x2y2_to_xywh(human)
        vis_data(small_img, [human_wh])
        keypoints = humans_points[i]
        vis_keypoints_data(deepcopy(small_img), keypoints, keypoints_mode="chainer")

    img_single = crop_candicates(small_img, humans[0])
    vis_data(img_single, None)
    image = cv2.imencode('.jpg', img_single)[1]
    base64_data = str(base64.b64encode(image))[2:-1]
    # print(len(humans),base64_data)
    keypoints = scale_transfer(humans_points[0], type=3, mode=2)
    jc = build_joint_connections_line(keypoints, keypoints_mode="chainer")
    img = add_joint_connections_with_lines(img1, jc)
    vis_keypoints_data(img, keypoints, keypoints_mode="chainer")


def test_alp_keypoints_get():
    cfg = update_config(args.cfg)
    img1_filepath = "/home/bavon/project/ai-demo/data/torchreid/query_test/jiashuhan_front.jpg"
    img1_filepath = "/home/bavon/face_test/reid/qingyang.jpg"
    img1_filepath = "/home/bavon/face_test/reid/por.jpg"
    #img1_filepath = "/home/bavon/face_test/reid/sm_q1.jpg"
    # img1_filepath = "/home/bavon/face_test/reid/sm_q1_s.jpg"
    img1_filepath = "/home/bavon/model/datasets/duibi/1428.jpg"
    img1 = cv2.imread(img1_filepath)
    height, width, ch = img1.shape
    img_scale_size_x = img_scale_size_y = 384
    from utils_pose import scale_transfer, rebuild_pose_cfg
    p_cfg = {"real_width": width, "real_height": height, "img_scale_size_x": img_scale_size_x,
             "img_scale_size_y": img_scale_size_y}
    rebuild_pose_cfg(p_cfg)
    small_img = cv2.resize(img1, (width, height))
    # small_img = cv2.resize(img1, (img_scale_size_x, img_scale_size_y),
    #                        interpolation=cv2.INTER_CUBIC)
    # small_img = small_img.transpose(2, 0, 1).astype(np.float32)
    pose_model, config = _load_pose_model(pose_mode="alp")
    cfg.detector = "yolo"
    det = get_detector(cfg)
    data_manager = QueryDataManager(
        cfg=cfg,
        detector=det,
        pose_model=pose_model,
        pose_mode=cfg.realtime_tracking.pose_mode,
        not_init_feature=True
    )
    # visdom_img_data(small_img,None)
    # human_point, preds_score = data_manager.get_keypoints_alp(small_img)
    (bboxes, scores, other_candidates) = data_manager.detector.inference_yolo_from_img(img1)
    humans = []
    humans_points = []
    preds_scores = []
    for i in range(0, len(bboxes)):  # ????????????base64?????
        box = bboxes[i]
        x = box[0]
        y = box[1]
        body = crop_candicates(small_img, box)
        visdom_img_data(body, title="body before:{}".format(i), cap="score...")
        human_point, preds_score = data_manager.get_keypoints_alp(body)

        if preds_score is None:
            continue
        # get real point
        try:
            hp = human_point[0]['pose_keypoints_2d']
            # visdom_img_data(body)
            visdom_img_data(body, title="body after:{}".format(i), cap="preds_score:{}".format(preds_score))
        except:
            continue

        else:
            length = len(hp)
            for i in range(0, length):
                if i % 3 == 0:
                    hp[i] = hp[i] + x
                if i % 3 == 1:
                    hp[i] = hp[i] + y
            preds_score = torch.squeeze(preds_score).cpu().numpy()
            if preds_score.ndim == 2:
                preds_score = preds_score[0]
            preds_score = preds_score.tolist()
            humans.append(xywh_to_x1y1x2y2(box))
            humans_points.append(human_point)
            preds_scores.append(preds_score)

    for i, human in enumerate(humans):
        human_wh = x1y1x2y2_to_xywh(human)
        keypoints = humans_points[i]
        kp = keypoints[0]["pose_keypoints_2d"]
        kp = np.array(kp).reshape(18, 3)
        vis_keypoints_data(small_img, kp)


def get_humans_keypoints(data_manager, img):
    (bboxes, scores, other_candidates) = data_manager.detector.inference_yolo_from_img(img)
    humans = []
    humans_points = []
    preds_scores = []
    for i in range(0, len(bboxes)):  # ????????????base64?????
        box = bboxes[i]
        x = box[0]
        y = box[1]
        body = crop_candicates(img, box)
        human_point, preds_score = data_manager.get_keypoints_alp(body)
        if preds_score is None:
            continue
        # get real point
        hp = human_point[0]['pose_keypoints_2d']
        length = len(hp)
        for i in range(0, length):
            if i % 3 == 0:
                hp[i] = hp[i] + x
            if i % 3 == 1:
                hp[i] = hp[i] + y
        preds_score = torch.squeeze(preds_score).cpu().numpy().tolist()
        humans.append(xywh_to_x1y1x2y2(box))
        humans_points.append(human_point)
        preds_scores.append(preds_score)
    return humans, humans_points, preds_scores


def create_chainer_pose_model(cfg, config):
    dataset_type = config.get('dataset', 'type')

    if dataset_type == 'mpii':
        import mpii_dataset as x_dataset
    elif dataset_type == 'coco':
        import coco_dataset as x_dataset
    else:
        raise Exception('Unknown dataset {}'.format(dataset_type))

    KEYPOINT_NAMES = x_dataset.KEYPOINT_NAMES
    EDGES = x_dataset.EDGES
    DIRECTED_GRAPHS = x_dataset.DIRECTED_GRAPHS
    COLOR_MAP = x_dataset.COLOR_MAP

    model = PoseProposalNet(
        model_name=config.get('model_param', 'model_name'),
        insize=parse_size(config.get('model_param', 'insize')),
        keypoint_names=KEYPOINT_NAMES,
        edges=np.array(EDGES),
        local_grid_size=parse_size(config.get('model_param', 'local_grid_size')),
        parts_scale=parse_size(config.get(dataset_type, 'parts_scale')),
        instance_scale=parse_size(config.get(dataset_type, 'instance_scale')),
        width_multiplier=config.getfloat('model_param', 'width_multiplier'),
    )

    result_dir = cfg.pose_rel.model
    chainer.serializers.load_npz(
        os.path.join(result_dir, 'bestmodel.npz'),
        model
    )

    if chainer.backends.cuda.available:
        model.to_gpu()
    elif chainer.backends.intel64.is_ideep_available():
        model.to_intel64()
    return {"model": model, "DIRECTED_GRAPHS": DIRECTED_GRAPHS, "COLOR_MAP": COLOR_MAP}


def check_pth_info():
    save_path = "/home/bavon/project/ai-demo/person_reid/train/result/PGFA/global_39.pth"
    net = torch.load(save_path, map_location=torch.device('cpu'))
    print(len(net))


def init_network():
    lr = 0.1
    class_num = 39
    model = PCB(class_num)
    print(model)
    print('*' * 20)
    global_classifier = ClassBlock(4096, class_num, True, False, 256)
    PCB_classifier = {}
    for i in range(3):
        PCB_classifier[i] = ClassBlock(2048, class_num, True, False, 256)

    model = model.cuda()
    global_classifier = global_classifier.cuda()
    ppp = model.parameters()
    print("model param:{}".format(ppp))
    for i in range(3):
        PCB_classifier[i].cuda()

    param_groups = [{'params': model.parameters(), 'lr': lr * 0.1},
                    {'params': global_classifier.parameters()}]
    weight_decay = 5e-4
    for i in range(3):
        param_groups.append({'params': PCB_classifier[i].parameters()})
    optimizer = optim.SGD(param_groups, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    return model, global_classifier, PCB_classifier, optimizer


def load_network(name_, network):
    name = "PGFA"
    opt = {"last_epoch": 29, "result_dir": "./person_reid/train/result"}
    save_path = os.path.join(opt["result_dir"], name, '%s_%s.pth' % (name_, opt["last_epoch"]))
    checkpoint = torch.load(save_path)
    network.load_state_dict(checkpoint['net'])
    return network


def pre_load_network(model_structure, global_model_structure, part_model_structure):
    opt = {"part_num": 3}
    model = load_network('net', model_structure)
    global_model = load_network('global', global_model_structure)
    global_model.classifier = nn.Sequential()
    partial_model = {}
    for i in range(opt["part_num"]):
        part_model_ = part_model_structure[i]
        partial_model[i] = load_network('partial' + str(i), part_model_)
        partial_model[i].classifier = nn.Sequential()
        partial_model[i].eval()
        partial_model[i] = partial_model[i].cuda()

    # Change to test mode
    model.eval()
    global_model.eval()
    model = model.cuda()
    global_model = global_model.cuda()
    return model, global_model, partial_model


def test_keypoints_match():
    cfg = update_config(args.cfg)
    config = load_config(cfg)
    pose_cfg = update_config(args.pose_cfg)
    pose_matcher = Pose_Matcher(args=pose_cfg)
    p_cfg = {"detection_thresh": cfg.pose_match.detection_thresh,
             "high_detection_thresh": cfg.pose_match.high_detection_thresh,
             "min_num_hs_keypoints": cfg.pose_match.min_num_hs_keypoints,
             "min_num_keypoints": cfg.pose_match.min_num_keypoints,
             "img_scale_size_x": cfg.img_scale_size_x, "img_scale_size_y": cfg.img_scale_size_y}
    rebuild_pose_cfg(p_cfg)
    pose_model = create_model(cfg, config)

    realtime_tracker = RealtimeTracker(device_id="0", det_loader=None, pose_model=pose_model,
                                       pose_matcher=pose_matcher, track_service=None, cfg=cfg)

    # keypointsA = [[225.42, 81.999, 0.43558], [215.43, 57.913, 0.0090646], [214.02, 55.098, 0.012487],
    #               [216.55, 54.931, 0.005677], [215.72, 56.15, 0.28424], [220.46, 55.146, 0.057287],
    #               [207.68, 83.147, 0.61158], [243.01, 81.348, 0.69751], [205.47, 132.1, 0.73187],
    #               [253.96, 119.54, 0.40091], [204.74, 121.25, 0.087091], [248.94, 139.52, 0.10407],
    #               [221.28, 170.37, 0.54386], [239.98, 171.2, 0.64372], [221.73, 230.35, 0.27566],
    #               [236.92, 232.88, 0.6709], [218.08, 293.99, 0.044804], [230.04, 297.31, 0.35274]]
    #
    # keypointsB = [[231.13, 123.85, 0.50556], [207.52, 107.71, 0.041185], [210.57, 105.02, 0.036979],
    #               [206.4, 106.48, 0.0077874], [215.83, 101.41, 0.11784], [208.63, 102.24, 0.022938],
    #               [219.9, 124.05, 0.15942], [244.53, 122.87, 0.48238], [213.41, 168.75, 0.59143],
    #               [251.61, 168.97, 0.23981], [207.22, 201.98, 0.21012], [246.9, 202.08, 0.048938],
    #               [221.49, 207.59, 0.58958], [239.78, 205.87, 0.62116], [218.42, 260.59, 0.23733],
    #               [236.94, 250.2, 0.46381], [214.71, 304.26, 0.60791], [236.87, 298.13, 0.52875]]
    #
    # keypointsA = np.array(keypointsA)
    # keypointsB = np.array(keypointsB)
    # bboxA = [195.47027587890625, 1.4881744384765625, 66, 350]
    # bboxB = [194.30747985839844, 49.71630096435547, 67, 302]
    imgA_path = "/home/bavon/face_test/full_img/192.168.0.122_01_20210108170226193_TIMING.jpg"
    imgB_path = "/home/vsftp_pas/20201035/192.168.0.122_01_20201226112603184_TIMING.jpg"
    imgA = cv2.imread(imgA_path)
    imgA = cv2.resize(imgA, (384, 384))
    imgB = cv2.imread(imgB_path)
    imgB = cv2.resize(imgB, (384, 384))
    bboxA, keypointsA = realtime_tracker.get_bbox_and_keypoints(imgA)
    vis_data(imgA, [bboxA], box_mode=2)
    bboxB, keypointsB = realtime_tracker.get_bbox_and_keypoints(imgB)
    vis_data(imgB, [bboxB], box_mode=2)
    pose_matching_score = realtime_tracker.get_pose_matching_score(keypointsA, keypointsB, bboxA, bboxB)
    print("pose_matching_score is:{}".format(pose_matching_score))

def test_openpose_net():
    protoFile = 'openpose/models/caffe_models/pose/body_25/pose_deploy.prototxt'
    weightsFile = 'openpose/models/caffe_models/pose/body_25/pose_iter_584000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    img = cv2.imread('/home/bavon/model/datasets/duibi/ziming1031.jpg')
    frameWidth = img.shape[1]
    frameHeight = img.shape[0]
    inHeight = 368
    inWidth = int((inHeight/frameHeight)*frameWidth)
    inBlob = cv2.dnn.blobFromImage(img,1.0/255.0,(224,224),(0,0,0),swapRB=False,crop=False)
    net.setInput(inBlob)
    output=net.forward()
    print("out shape:{}".format(output.shape))#(1, 57, 46, 60)


def test_zengqiang():
    img = cv2.imread("/home/bavon/model/datasets/duibi/1613.jpg")
    # clahe = cv2.createCLAHE(clipLimit=600.0, tileGridSize=(1, 1))
    # img_clahe = clahe.apply(img)
    # img = cv2.imread("/home/bavon/model/datasets/duibi/wyface.jpg", 0)
    # cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    # lut = np.zeros(256, dtype=img.dtype)
    # hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    # cdf = hist.cumsum()
    # cdf_m = np.ma.masked_equal(cdf, 0)
    # cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    # cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    # res = cdf[img]
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #
    # lab_planes = cv2.split(lab)
    #
    # clahe = cv2.createCLAHE(clipLimit=600.0, tileGridSize=(1, 1))
    #
    # lab_planes[0] = clahe.apply(lab_planes[0])
    #
    # lab = cv2.merge(lab_planes)

    # bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # im = Image.fromarray(bgr)
    score = cv2.Laplacian(img, cv2.CV_64F).var()
    print(score)
    # im.save("test5.jpg")
    # print('save')


def test_pid():
    import psutil
    process_list = list(psutil.process_iter())
    print(process_list)
    for i in range(len(process_list)):
        print(process_list[i])

def test_sqllite():
    import sqlite3
    cu = sqlite3.connect("/home/bavon/model/sqllite/device.db")
    cur = cu.cursor()
    # delete_sql = 'delete from device where deviceNO = 20210130'
    # cu.execute(delete_sql)
    #cu.execute('create table time (id int,lastSyncTime int)')
    #cu.execute("insert into time values(1,0)")
    # # cu.execute("insert into device values(20210130, 'rtsp://admin:zkrh2019@192.168.0.159:554/h264/ch1/main/av_stream', 0,0,0)")
    # # cu.commit()
    # cur.execute("UPDATE device SET falg=1  WHERE id=1")
    # cu.commit()
    # cur.execute("PRAGMA table_info(time)")
    # print(cur.fetchall())
    # bian_ma = "A123456"
    # up_sql = "delete from time where boxId = ?"
    # cur.execute(up_sql,(bian_ma,))
    # devices = [{'id': 1, 'deviceName': '测试设备2', 'deviceBrand': 1, 'deviceType': 1, 'deviceId': '2021002', 'ip': '192.168.0.159', 'port': 554, 'loginName': 'admin', 'password': 'admin123', 'mainStream': 'rtsp://admin:zkrh2019@192.168.0.159:554/h264/ch1/main/av_stream', 'subStream': 'rtsp://admin:zkrh2019@192.168.0.159:554/h264/ch1/main/av_stream', 'addressCode': '210203', 'address': '地址2', 'inOutFlag': 0, 'deviceState': None, 'streamState': None, 'state': 1, 'deleteFlag': None, 'boxId': 'A123456', 'createTime': None, 'updateTime': None, 'deleteTime': None, 'deviceBrandName': None, 'deviceTypeName': None, 'bCreateTime': None, 'eCreateTime': None, 'bUpdateTime': None, 'eUpdateTime': None, 'bDeleteTime': None, 'eDeleteTime': None, 'dataType': 1, 'ids': None}]
    # deldevice = devices[0]['id']
    # delete_sql = 'delete from device where id = '+str(deldevice)+''
    # cur.execute(delete_sql)
    # cu.commit()
    #cur.execute("select * from device ")
    # cur.execute("select mainStream from device where id = 30 ")
    # device = []
    # for row in cur.fetchall():
    #     print(row)
    #     device.append(row)
    select_isql = "select deviceName,deviceType,inOutFlag,ip,port,loginName,password,mainstream,state,platformTenantId,boxId,id from device where insertTime>?"
    cur.execute(select_isql, (0,))
    print(cur.fetchall())

def test_ping(ip):
    import subprocess
    if subprocess.call(["ping", "-c", "2", ip]) == 0:  # 只发送两个ECHO_REQUEST包
        return 0
    else:
        return 1


if __name__ == "__main__":
    #test_openpose_net()
    # test_query_data()
    # test_compare_img()
    # image_test()
    #test_keypoints_get()
    # test_add_data()
    # test_load_data()
    # test_img_match()
    # test_img_match()
    # test_pose_get()
    # test_pose_crop_get()
    # test_cv_crop()
    # test_img_letter()
    test_alp_keypoints_get()
    # test_chainer_keypoints_get()
    # check_pth_info()
    # test_keypoints_match()
    # test_yolo_alp_get_human_features()
    #test_zengqiang()
    #test_pid()
    #test_sqllite()
    #test_ping(ip='192.168.0.1')

