import torchreid
from torchreid import metrics

from torchreid.utils import FeatureExtractor

extractor = FeatureExtractor(
    # model_name='resnet50',
    model_name='osnet_ain_x1_0',
    model_path='/home/bavon/model/personreid/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth',
    # model_path='scripts/log/model/model.pth.tar-60',
    device='cuda'
)

# image_list1 = [
#     '/home/bavon/model/datasets/duibi/mz11.jpg',
#     # '/home/bavon/model/datasets/test_20/1f85d35e6d9430e64624e80f25a1306d_capture20210131_1621128953108_body.jpg',
#     # '/home/bavon/model/datasets/test_20/25d02e63dab775c66b56494be23ff55b_capture20210131_1621127598358_body.jpg',
#     # '/home/bavon/model/datasets/test_20/2f86a78a324dd9551720b5a0a06058f8_capture20210131_1621128499442_body.jpg',
#     # '/home/bavon/model/datasets/test_20/3bb8f3ddddb5aa79a742e5f5d0fe1eb8_capture20210131_1621131210183_body.jpg',
#     # '/home/bavon/model/datasets/test_20/49a2d2b995c7ee1417293725f719b41f_capture20210131_1621128294968_body.jpg'
# ]
# image_list2 = [
#     '/home/bavon/model/datasets/duibi/mzz.png'
# ]


image_list1 = [
    '/home/bavon/app/MODNet/demo/output/yuexin11.jpg'
]
image_list1 = [
    '/home/bavon/face_test/reid/d1.jpg'
]
image_list2 = [
    # '/home/bavon/app/MODNet/demo/output/yuexin12.jpg',
    # '/home/bavon/app/MODNet/demo/output/yuexin2.jpg',
    # '/home/bavon/app/MODNet/demo/output/yuexin3.jpg',
    '/home/bavon/face_test/reid/yuexin12.jpg',
    '/home/bavon/face_test/reid/d2.jpg',
    '/home/bavon/model/datasets/duibi/yy2.jpg',
    # '/home/bavon/face_test/reid/d1.jpg'
]

features1 = extractor(image_list1)
features2 = extractor(image_list2)

distmat = metrics.compute_distance_matrix(features1, features2,metric='cosine')
# distmat = metrics.compute_distance_matrix(features1, features2)
print("distmat is:",distmat)