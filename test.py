import torch
import torch.nn.functional as functional
import chainer
import os
import numpy as np
import getpass
from visdom import Visdom
from sys import platform as _platform
# import urllib
from six.moves import urllib

# out = os.path.join(os.getcwd(), "out")
# print(os.getcwd())
# print(os.path.abspath(__file__))
# print(os.path.exists(out))
# chainer.functions.roi_pooling_2d()
# functional.adaptive_max_pool2d()

# chainer.functions.roi_pooling_2d()


vis = Visdom()
# video_url = "http://media.w3.org/2010/05/sintel/trailer.ogv"
# print(_platform)  # Linux
# print(getpass.getuser())   # wendi
# video_file = "/home/{}/data/videos/trailer.ogv".format(getpass.getuser())
# urllib.request.urlretrieve(video_url, video_file)   # download video

# if os.path.isfile(video_file):
#     vis.video(videofile=video_file)


# vis = visdom.Visdom()
vis.text('Hello, world!')
vis.image(np.ones((3, 10, 10)))

