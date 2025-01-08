from posenet.constants import *
# from posenet.decode_multi import decode_multiple_poses
from posenet import decode
from posenet.models.model_factory import load_model
from posenet.models import MobileNetV1, MOBILENET_V1_CHECKPOINTS
from posenet.utils import *

# posenet.model.mobilenet_v1 -> posenet.model -> posenet 向外封装
# posenet 提供 models、load_model、decode、一些uitls和常量的接口