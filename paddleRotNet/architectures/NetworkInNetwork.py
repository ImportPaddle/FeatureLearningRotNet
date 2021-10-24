import math
import paddle
from paddle import nn, reshape
from paddle.fluid import Variable


class BasicBlock(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(BasicBlock, self).__init__()
        padding = int((kernel_size - 1) / 2)  ##
        # print('padding',padding)
        self.layers = nn.Sequential()
        self.layers.add_sublayer('Conv',
                                 nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding,
                                           bias_attr=False))
        self.layers.add_sublayer('BatchNorm', nn.BatchNorm2D(out_planes))
        self.layers.add_sublayer('ReLU', nn.ReLU())

    def forward(self, x):
        # print(type(x))
        # print(x.shape)
        # print(next(self.layers.parameters()).device)
        return self.layers(x)


class GlobalAveragePooling(nn.Layer):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, feat):
        num_channels = feat.shape[1]
        return reshape(nn.functional.avg_pool2d(feat, (feat.shape[2], feat.shape[3])), (-1, num_channels))


class NetworkInNetwork(nn.Layer):
    def __init__(self, opt):
        super(NetworkInNetwork, self).__init__()

        num_classes = opt['num_classes']
        num_inchannels = opt['num_inchannels'] if ('num_inchannels' in opt) else 3
        num_stages = opt['num_stages'] if ('num_stages' in opt) else 3
        use_avg_on_conv3 = opt['use_avg_on_conv3'] if ('use_avg_on_conv3' in opt) else True

        assert (num_stages >= 3)
        nChannels = 192
        nChannels2 = 160
        nChannels3 = 96

        blocks = [nn.Sequential() for i in range(num_stages)]
        # 1st block
        # print('num_channels',num_inchannels)

        blocks[0].add_sublayer('Block1_ConvB1', BasicBlock(num_inchannels, nChannels, 5))
        blocks[0].add_sublayer('Block1_ConvB2', BasicBlock(nChannels, nChannels2, 1))
        blocks[0].add_sublayer('Block1_ConvB3', BasicBlock(nChannels2, nChannels3, 1))
        blocks[0].add_sublayer('Block1_MaxPool', nn.MaxPool2D(kernel_size=3, stride=2, padding=1))

        # 2nd block
        blocks[1].add_sublayer('Block2_ConvB1', BasicBlock(nChannels3, nChannels, 5))
        blocks[1].add_sublayer('Block2_ConvB2', BasicBlock(nChannels, nChannels, 1))
        blocks[1].add_sublayer('Block2_ConvB3', BasicBlock(nChannels, nChannels, 1))
        blocks[1].add_sublayer('Block2_AvgPool', nn.AvgPool2D(exclusive=False,kernel_size=3, stride=2, padding=1))

        # 3rd block
        blocks[2].add_sublayer('Block3_ConvB1', BasicBlock(nChannels, nChannels, 3))
        blocks[2].add_sublayer('Block3_ConvB2', BasicBlock(nChannels, nChannels, 1))
        blocks[2].add_sublayer('Block3_ConvB3', BasicBlock(nChannels, nChannels, 1))

        if num_stages > 3 and use_avg_on_conv3:
            blocks[2].add_sublayer('Block3_AvgPool', nn.AvgPool2D(exclusive=False,kernel_size=3, stride=2, padding=1))
        for s in range(3, num_stages):
            blocks[s].add_sublayer('Block' + str(s + 1) + '_ConvB1', BasicBlock(nChannels, nChannels, 3))
            blocks[s].add_sublayer('Block' + str(s + 1) + '_ConvB2', BasicBlock(nChannels, nChannels, 1))
            blocks[s].add_sublayer('Block' + str(s + 1) + '_ConvB3', BasicBlock(nChannels, nChannels, 1))

        # global average pooling and classifier
        blocks.append(nn.Sequential())
        blocks[-1].add_sublayer('GlobalAveragePooling', GlobalAveragePooling())
        blocks[-1].add_sublayer('Classifier', nn.Linear(nChannels, num_classes))

        self._feature_blocks = nn.LayerList(blocks)
        self.all_feat_names = ['conv' + str(s + 1) for s in range(num_stages)] + ['classifier', ]
        assert (len(self.all_feat_names) == len(self._feature_blocks))
        # self.weight_initialization()

    def _parse_out_keys_arg(self, out_feat_keys):

        out_feat_keys = [self.all_feat_names[-1], ] if out_feat_keys is None else out_feat_keys
        # print(out_feat_keys)
        if len(out_feat_keys) == 0:
            raise ValueError('Empty list of output feature keys.')

        # By default return the features of the last layer / module.
        # out_feat_keys = [self.all_feat_names[-1],] if out_feat_keys is None else out_feat_keys

        # if len(out_feat_keys)==0:
        #     raise ValueError('Empty list of output feature keys.')
        for f, key in enumerate(out_feat_keys):
            if key not in self.all_feat_names:
                raise ValueError(
                    'Feature with name {0} does not exist. Existing features: {1}.'.format(key, self.all_feat_names))
            elif key in out_feat_keys[:f]:
                raise ValueError('Duplicate output feature key: {0}.'.format(key))

        # Find the highest output feature in `out_feat_keys
        max_out_feat = max([self.all_feat_names.index(key) for key in out_feat_keys])

        return out_feat_keys, max_out_feat

    def forward(self, x, out_feat_keys=None):
        """Forward an image `x` through the network and return the asked output features.

        Args:
          x: input image.
          out_feat_keys: a list/tuple with the feature names of the features
                that the function should return. By default the last feature of
                the network is returned.

        Return:
            out_feats: If multiple output features were asked then `out_feats`
                is a list with the asked output features placed in the same
                order as in `out_feat_keys`. If a single output feature was
                asked then `out_feats` is that output feature (and not a list).
        """

        # print(out_feat_keys)
        out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
        out_feats = [None] * len(out_feat_keys)

        # print('out_feat_keys', out_feat_keys)
        feat = x
        for f in range(max_out_feat + 1):
            # print(f)
            feat = self._feature_blocks[f](feat)
            key = self.all_feat_names[f]
            # print('paddle', key)
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat

        # out_feats = out_feats[0] if len(out_feats)==1 else out_feats
        # return out_feats

        out_feats = out_feats[0] if len(out_feats) == 1 else out_feats

        return out_feats

    def weight_initialization(self):
        # print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                m.weight.set_value(paddle.normal(shape=m.weight.shape, mean=0, std=math.sqrt(2. / n)))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(paddle.full_like(m.weight, fill_value=1))
                m.bias.set_value(paddle.zeros_like(m.bias))
            elif isinstance(m, nn.Linear):
                m.bias.set_value(paddle.zeros_like(m.bias))


def create_model(opt):
    return NetworkInNetwork(opt)

