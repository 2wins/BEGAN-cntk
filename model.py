import cntk as C
from cntk.layers import BatchNormalization, Dense, Convolution2D, ConvolutionTranspose2D


# Architectural parameters
kernel_h, kernel_w = 5, 5
stride_h, stride_w = 2, 2

gkernel = dkernel = kernel_h
gstride = dstride = stride_h


def l1_loss(x, y):
    return C.reduce_mean(C.abs(x - y)) 


# Helper functions
def bn_with_leaky_relu(x, leak=0.2):
    h = BatchNormalization(map_rank=1)(x)
    return C.leaky_relu(h, leak)


# Architecutres similar to DCGAN in some part
def generator(z, opt):
    with C.layers.default_options(init=C.normal(scale=0.02)):
        img_h = img_w = opt.imageSize
        s_h2, s_w2 = img_h//2, img_w//2 #Input shape (14,14)
        s_h4, s_w4 = img_h//4, img_w//4 # Input shape (7,7)
        gfc_dim = 64 
        gf_dim = 64

        h = Dense(gfc_dim, activation=None)(z)
        h = bn_with_leaky_relu(h)

        h = Dense([gf_dim * 2, s_h4,  s_w4], activation=None)(h)
        h = bn_with_leaky_relu(h)

        h = ConvolutionTranspose2D(gkernel,
                                  num_filters=gf_dim*2,
                                  strides=gstride,
                                  bias=False,
                                  pad=True,
                                  output_shape=(s_h2, s_w2),
                                  activation=None)(h)
        h = bn_with_leaky_relu(h)

        h = ConvolutionTranspose2D(gkernel,
                                  num_filters=1,
                                  strides=gstride,
                                  pad=True,
                                  output_shape=(img_h, img_w),
                                  activation=C.tanh)(h)

        return C.reshape(h, img_h * img_w)


def discriminator(x, opt):
    with C.layers.default_options(init=C.normal(scale=0.02)):
        img_h = img_w = opt.imageSize
        s_h2, s_w2 = img_h//2, img_w//2 #Input shape (14,14)
        s_h4, s_w4 = img_h//4, img_w//4 # Input shape (7,7)
        dfc_dim = 64 
        df_dim = 64 

        x = C.reshape(x, (1, img_h, img_w))

        h = Convolution2D(dkernel, 1, strides=dstride)(x)
        h = bn_with_leaky_relu(h, leak=0.2)

        h = Convolution2D(dkernel, df_dim, strides=dstride)(h)
        h = bn_with_leaky_relu(h, leak=0.2)

        h = Dense(opt.z_dim, activation=C.tanh)(h)

        h = Dense(dfc_dim, activation=None)(h)
        h = bn_with_leaky_relu(h)

        h = Dense([df_dim * 2, s_h4,  s_w4], activation=None)(h)
        h = bn_with_leaky_relu(h)

        h = ConvolutionTranspose2D(dkernel,
                                  num_filters=df_dim*2,
                                  strides=dstride,
                                  bias=False,
                                  pad=True,
                                  output_shape=(s_h2, s_w2),
                                  activation=None)(h)
        h = bn_with_leaky_relu(h)

        h = ConvolutionTranspose2D(dkernel,
                                  num_filters=1,
                                  strides=dstride,
                                  pad=True,
                                  output_shape=(img_h, img_w),
                                  activation=C.tanh)(h)

        return C.reshape(h, img_h * img_w)
