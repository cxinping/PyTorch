from visdom import Visdom

viz = Visdom()

# image demo
viz.image(
    np.random.rand(3, 512, 256),
    opts=dict(title='单图片', caption='图片标题'),
)

# grid of images
viz.images(
    np.random.randn(20, 3, 64, 64),
    opts=dict(title='网格图像', caption='图片标题')
)