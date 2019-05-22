import chainer
import os

out = os.path.join(os.getcwd(), "out")
print(os.getcwd())
print(os.path.abspath(__file__))
print(os.path.exists(out))
chainer.functions.roi_pooling_2d()


# chainer.functions.roi_pooling_2d()