import paddle
import paddle.fluid as fluid
import parl.layers as layers
from parl.framework.model_base import Model


# class LJST_Model(Model):
#     def __init__(self, act_dim):
#         self.act_dim = act_dim

#         self.conv1 = layers.conv2d(
#             num_filters=8, filter_size=4, stride=1, padding=2, act='relu')
#         self.conv2 = layers.conv2d(
#             num_filters=8, filter_size=4, stride=1, padding=2, act='relu')
#         self.conv3 = layers.conv2d(
#             num_filters=8, filter_size=4, stride=1, padding=2, act='relu')
#         self.conv4 = layers.conv2d(
#             num_filters=8, filter_size=4, stride=1, padding=2, act='relu')
#         self.fc1 = layers.fc(size=act_dim)

#     def value(self, obs):
# #         obs = obs / 255.0
#         out = self.conv1(obs)
#         out = layers.pool2d(
#             input=out, pool_size=2, pool_stride=2, pool_type='max')
#         out = self.conv2(out)
#         out = layers.pool2d(
#             input=out, pool_size=2, pool_stride=2, pool_type='max')
#         out = self.conv3(out)
#         out = layers.pool2d(
#             input=out, pool_size=2, pool_stride=2, pool_type='max')
#         out = self.conv4(out)
#         out = layers.flatten(out, axis=1)
#         out = self.fc1(out)
#         return out
class LJST_Model(Model):
    def __init__(self, act_dim):
        self.act_dim = act_dim

        self.conv1 = layers.conv2d(
            num_filters=8, filter_size=4, stride=1, padding=2, act='relu')
        self.conv2 = layers.conv2d(
            num_filters=8, filter_size=4, stride=1, padding=2, act='relu')
        self.conv3 = layers.conv2d(
            num_filters=8, filter_size=4, stride=1, padding=2, act='relu')
        self.conv4 = layers.conv2d(
            num_filters=8, filter_size=4, stride=1, padding=2, act='relu')
        self.fc1 = layers.fc(size=act_dim, act=None)
        self.fc2 = layers.fc(size=act_dim, act=None)
        self.fc3 = layers.fc(size=1, act='sigmoid')

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
        start = self.fc1(out)
        start = paddle.fluid.layers.softmax(start)
        end = self.fc2(out)
        end = paddle.fluid.layers.softmax(end)
        num = self.fc3(out)
#         print('start',start.shape)
#         print('end',end.shape)
#         print('num',num.shape)
        vec = layers.concat([start,end,num],axis=1)
#         print('model_shape: ',vec.shape)
        return vec
#         return start
    
