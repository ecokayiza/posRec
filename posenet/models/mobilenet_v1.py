import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


####模型结构 ！




# 3.定义模型结构和输出

def _get_padding(kernel_size, stride, dilation):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

# 输入卷积层
class InputConv(nn.Module):
    def __init__(self, inp, outp, k=3, stride=1, dilation=1):
        super(InputConv, self).__init__()
        
        # 输入经过一层卷积
        self.conv = nn.Conv2d(
            inp, outp, k, stride, padding=_get_padding(k, stride, dilation), dilation=dilation)

    def forward(self, x):
        return F.relu6(self.conv(x)) #limit to 0-6

# 深度可分离卷积
class SeperableConv(nn.Module):
    def __init__(self, inp, outp, k=3, stride=1, dilation=1):
        super(SeperableConv, self).__init__()
        
        # 深度卷积
        # 输出通道与输入通道数相同
        self.depthwise = nn.Conv2d(
            inp, inp, k, stride,
            padding=_get_padding(k, stride, dilation), dilation=dilation, groups=inp)
        
        # 点积
        self.pointwise = nn.Conv2d(inp, outp, 1, 1)

    def forward(self, x):
        x = F.relu6(self.depthwise(x))
        x = F.relu6(self.pointwise(x))
        return x
        

# 根据网络结构自动调整步幅和膨胀率
def _to_output_strided_layers(convolution_def, output_stride):
    current_stride = 1
    rate = 1
    block_id = 0
    buff = []
    for c in convolution_def:
        conv_type = c[0]
        inp = c[1]
        outp = c[2]
        stride = c[3]

        if current_stride == output_stride:  #达到累计目标步幅则保持步幅为1但膨胀率改变
            layer_stride = 1
            layer_rate = rate
            rate *= stride
        else:                            #未达到累计目标步幅则不变
            layer_stride = stride     
            layer_rate = 1
            
            current_stride *= stride

        buff.append({
            'block_id': block_id,
            'conv_type': conv_type,
            'inp': inp,
            'outp': outp,
            'stride': layer_stride,
            'rate': layer_rate,
            'output_stride': current_stride
        })
        block_id += 1

    return buff

# 网络结构
MOBILENET_V1_CHECKPOINTS = {
    50: 'mobilenet_v1_050',
    75: 'mobilenet_v1_075',
    100: 'mobilenet_v1_100',
    101: 'mobilenet_v1_101',
    0: 'mobilenet_my_v0'
}

MOBILE_NET_V1_100 = [
    (InputConv, 3, 32, 2),        # (输入通道、输出通道、步幅)
    (SeperableConv, 32, 64, 1),
    (SeperableConv, 64, 128, 2),
    (SeperableConv, 128, 128, 1),
    (SeperableConv, 128, 256, 2),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 512, 2),    #累计步幅达到16，膨胀为1 (一般output_stride=16)
    (SeperableConv, 512, 512, 1),    
    (SeperableConv, 512, 512, 1),
    (SeperableConv, 512, 512, 1),
    (SeperableConv, 512, 512, 1),
    (SeperableConv, 512, 512, 1),
    (SeperableConv, 512, 1024, 2),   #膨胀率变为2
    (SeperableConv, 1024, 1024, 1)          
]

MOBILE_NET_V1_75 = [
    (InputConv, 3, 24, 2),
    (SeperableConv, 24, 48, 1),
    (SeperableConv, 48, 96, 2),
    (SeperableConv, 96, 96, 1),
    (SeperableConv, 96, 192, 2),
    (SeperableConv, 192, 192, 1),
    (SeperableConv, 192, 384, 2),
    (SeperableConv, 384, 384, 1),
    (SeperableConv, 384, 384, 1),
    (SeperableConv, 384, 384, 1),
    (SeperableConv, 384, 384, 1),
    (SeperableConv, 384, 384, 1),
    (SeperableConv, 384, 384, 1),
    (SeperableConv, 384, 384, 1)
]

MOBILE_NET_V1_50 = [
    (InputConv, 3, 16, 2),
    (SeperableConv, 16, 32, 1),
    (SeperableConv, 32, 64, 2),
    (SeperableConv, 64, 64, 1),
    (SeperableConv, 64, 128, 2),
    (SeperableConv, 128, 128, 1),
    (SeperableConv, 128, 256, 2),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 256, 1)
]



class MobileNetV1(nn.Module):
    def __init__(self, model_id, output_stride=16):
        super(MobileNetV1, self).__init__()

        assert model_id in MOBILENET_V1_CHECKPOINTS.keys()
        self.output_stride = output_stride

        if model_id == 50:
            arch = MOBILE_NET_V1_50
        elif model_id == 75:
            arch = MOBILE_NET_V1_75
        else:
            arch = MOBILE_NET_V1_100

        conv_def = _to_output_strided_layers(arch, output_stride)
        conv_list = [
            ('conv%d' % c['block_id'], c['conv_type'](
            c['inp'], c['outp'], 3, stride=c['stride'], dilation=c['rate'])
            )
            for c in conv_def]
        # {
        # 'block_id': block_id,
        # 'conv_type': conv_type,
        # 'inp': inp,
        # 'outp': outp,
        # 'stride': layer_stride,
        # 'rate': layer_rate,
        # 'output_stride': current_stride
        # }
        
        self.id = model_id
        
        # 输出通道数(1024)
        last_depth = conv_def[-1]['outp']
        
        # 从arch中定义的网络结构
        self.features = nn.Sequential(OrderedDict(conv_list))
        
        
        # tunning
        if(self.id == 0):
            self.new1 = SeperableConv(last_depth, last_depth, 3, 1, 1)
            self.new2 = SeperableConv(last_depth, last_depth, 3, 1, 1)
            self.new3 = SeperableConv(last_depth, last_depth, 3, 1, 1)

        
        #（用于生成关键点特征的卷积）
        # 用于生成关键点的热图
        self.heatmap = nn.Conv2d(last_depth, 17, 1, 1) #17个关键点
        # 关键点偏移量
        self.offset = nn.Conv2d(last_depth, 34, 1, 1) #每个关键点有两个偏移量x,y
        # 用于连接骨架
        self.displacement_fwd = nn.Conv2d(last_depth, 32, 1, 1) #每个关键点有16个偏移 *2
        self.displacement_bwd = nn.Conv2d(last_depth, 32, 1, 1)


    def forward(self, x):
        # 输入并经过网络
        x = self.features(x)
        # (1,1024,33,33)
        x1 = x
        if(self.id == 0):
            x = self.new1(x)
            x = self.new2(x)
            x1 = self.new3(x)
        
        # 网络输出再经过特定卷积获得特征图
        heatmap = torch.sigmoid(self.heatmap(x1))
        offset = self.offset(x)
        displacement_fwd = self.displacement_fwd(x)
        displacement_bwd = self.displacement_bwd(x)
        
        return heatmap, offset, displacement_fwd, displacement_bwd
