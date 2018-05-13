import tensorflow as tf
slim = tf.contrib.slim
from resnet import vgg

# ImageNet mean statistics
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def deeplab_v3(inputs, args, is_training,reuse):

    # mean subtraction normalization
    inputs = inputs - [_R_MEAN, _G_MEAN, _B_MEAN]

    # inputs has shape [batch, 513, 513, 3]
    resnet = getattr(vgg, args.resnet_model)
    if is_training:
        new_dropout = 0.5
    else:
        new_dropout = 1.0

    _, end_points = resnet(inputs,args.number_of_classes,is_training=is_training,dropout_keep_prob=new_dropout, spatial_squeeze=False, scope=args.resnet_model,fc_conv_padding = 'VALID',global_pool=False,reuse=reuse)
    with tf.variable_scope("DeepLab_v3"):
        # get block 4 feature outputs
        if args.resnet_model != "vgg_16":
            net = end_points[args.resnet_model + '/block4']
            net = atrous_spatial_pyramid_pooling(net, "ASPP_layer", depth=256, reuse=reuse)
            net = slim.conv2d(net, args.number_of_classes, [1, 1], activation_fn=None,
                          normalizer_fn=None, scope='logits')
        elif args.resnet_model == "vgg_16":
            net = end_points[args.resnet_model + '/fc8']
            #net = slim.conv2d(net, args.number_of_classes, [1, 1], activation_fn=None,normalizer_fn=None, scope='logits')



        size = tf.shape(inputs)[1:3]
        # resize the output logits to match the labels dimensions
        #net = tf.image.resize_nearest_neighbor(net, size)
        net = tf.image.resize_bilinear(net, size)
        return net
