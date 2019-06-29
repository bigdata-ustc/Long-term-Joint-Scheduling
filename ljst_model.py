
import paddle.fluid as fluid
import parl.layers as layers
from parl.framework.model_base import Model


class LJST_Model(Model):
    def __init__(self, act_dim):
        self.act_dim = act_dim

        self.conv1 = layers.conv2d(
            num_filters=8, filter_size=2, stride=1, padding=2, act='relu')
        self.conv2 = layers.conv2d(
            num_filters=8, filter_size=2, stride=1, padding=2, act='relu')
        self.conv3 = layers.conv2d(
            num_filters=8, filter_size=2, stride=1, padding=2, act='relu')
        self.conv4 = layers.conv2d(
            num_filters=8, filter_size=2, stride=1, padding=2, act='relu')
        self.fc1 = layers.fc(size=act_dim)

    def value(self, obs):
#         obs = obs / 255.0
        out = self.conv1(obs)
        out = layers.pool2d(
            input=out, pool_size=2, pool_stride=2, pool_type='max')
        out = self.conv2(out)
        out = layers.pool2d(
            input=out, pool_size=2, pool_stride=2, pool_type='max')
        out = self.conv3(out)
        out = layers.pool2d(
            input=out, pool_size=2, pool_stride=2, pool_type='max')
        out = self.conv4(out)
        out = layers.flatten(out, axis=1)
        out = self.fc1(out)
        return out

    