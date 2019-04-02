from ops import *
from bilinear_sampler import *


def stem_block(input_batch):
    print(' - Stem Block for Multi-scale Shared Features Extraction')

    with tf.variable_scope("conv1"):
        conv1 = conv2d(input_batch, [7, 7, 3, 64], 2, True)
        print ('conv1:')
        print (conv1.get_shape().as_list())

    with tf.variable_scope("up_conv1"):
        up_conv1 = conv2d_transpose(conv1, [4, 4, 32, 64], 2, True)
        print ('up_conv1:')
        print (up_conv1.get_shape().as_list())

    with tf.variable_scope("conv2"):
        conv2 = conv2d(conv1, [5, 5, 64, 128], 2, True)
        print ('conv2:')
        print (conv2.get_shape().as_list())

    with tf.variable_scope("up_conv2"):
        up_conv2 = conv2d_transpose(conv2, [8, 8, 32, 128], 4, True)
        print ('up_conv2:')
        print (up_conv2.get_shape().as_list())

    with tf.variable_scope("up_conv12"):
        up_conv12 = conv2d(tf.concat([up_conv1, up_conv2], axis=3), [1, 1, 64, 32], 1, True)
        print ('up_conv12:')
        print (up_conv12.get_shape().as_list())

    return conv1, up_conv1, conv2, up_conv2, up_conv12


def disparity_estimation(features):
    print(' - Initial Disparity Estimation Sub-network')

    conv1a = features[0]
    conv2a = features[2]
    up_conv1a2a = features[4]

    with tf.variable_scope("conv_redir"):
        conv_redir = conv2d(conv2a, [1, 1, 128, 64], 1, True)
        print ('conv_redir:')
        print (conv_redir.get_shape().as_list())

    with tf.variable_scope("conv3"):
        conv3 = conv2d(tf.concat(conv_redir, axis=3), [3, 3, 64, 256], 2, True)
        print ('conv3:')
        print (conv3.get_shape().as_list())

    with tf.variable_scope("conv3_1"):
        conv3_1 = conv2d(conv3, [3, 3, 256, 256], 1, True)
        print ('conv3_1:')
        print (conv3_1.get_shape().as_list())

    with tf.variable_scope("conv4"):
        conv4 = conv2d(conv3_1, [3, 3, 256, 512], 2, True)
        print ('conv4:')
        print (conv4.get_shape().as_list())

    with tf.variable_scope("conv4_1"):
        conv4_1 = conv2d(conv4, [3, 3, 512, 512], 1, True)
        print ('conv4_1:')
        print (conv4_1.get_shape().as_list())

    with tf.variable_scope("conv5"):
        conv5 = conv2d(conv4_1, [3, 3, 512, 512], 2, True)
        print ('conv5:')
        print (conv5.get_shape().as_list())

    with tf.variable_scope("conv5_1"):
        conv5_1 = conv2d(conv5, [3, 3, 512, 512], 1, True)
        print ('conv5_1:')
        print (conv5_1.get_shape().as_list())

    with tf.variable_scope("conv6"):
        conv6 = conv2d(conv5_1, [3, 3, 512, 1024], 2, True)
        print ('conv6:')
        print (conv6.get_shape().as_list())

    with tf.variable_scope("conv6_1"):
        conv6_1 = conv2d(conv6, [3, 3, 1024, 1024], 1, True)
        print ('conv6_1:')
        print (conv6_1.get_shape().as_list())

    with tf.variable_scope("disp6_loss6"):
        disp6 = conv2d(conv6_1, [3, 3, 1024, 2], 1, True)
        print ('disp6:')
        print (disp6.get_shape().as_list())

    with tf.variable_scope("upconv5"):
        upconv5 = conv2d_transpose(conv6_1, [4, 4, 512, 1024], 2, True)
        print ('upconv5:')
        print (upconv5.get_shape().as_list())

    with tf.variable_scope("iconv5"):
        iconv5 = conv2d(tf.concat([upconv5, conv2d_transpose(disp6, [4, 4, 1, 2], 2, True), conv5_1], axis=3),
                        [3, 3, 1025, 512], 1, True)
        print ('iconv5:')
        print (iconv5.get_shape().as_list())

    with tf.variable_scope("disp5_loss5"):
        disp5 = conv2d(iconv5, [3, 3, 512, 2], 1, True)
        print ('disp5:')
        print (disp5.get_shape().as_list())

    with tf.variable_scope("upconv4"):
        upconv4 = conv2d_transpose(iconv5, [4, 4, 256, 512], 2, True)
        print ('upconv4:')
        print (upconv4.get_shape().as_list())

    with tf.variable_scope("iconv4"):
        iconv4 = conv2d(tf.concat([upconv4, conv2d_transpose(disp5, [4, 4, 1, 2], 2, True), conv4_1], axis=3),
                        [3, 3, 769, 256], 1, True)
        print ('iconv4:')
        print (iconv4.get_shape().as_list())

    with tf.variable_scope("disp4_loss4"):
        disp4 = conv2d(iconv4, [3, 3, 256, 2], 1, True)
        print ('disp4:')
        print (disp4.get_shape().as_list())

    with tf.variable_scope("upconv3"):
        upconv3 = conv2d_transpose(iconv4, [4, 4, 128, 256], 2, True)
        print ('upconv3:')
        print (upconv3.get_shape().as_list())

    with tf.variable_scope("iconv3"):
        iconv3 = conv2d(tf.concat([upconv3, conv2d_transpose(disp4, [4, 4, 1, 2], 2, True), conv3_1], axis=3),
                        [3, 3, 385, 128], 1, True)
        print ('iconv3:')
        print (iconv3.get_shape().as_list())

    with tf.variable_scope("disp3_loss3"):
        disp3 = conv2d(iconv3, [3, 3, 128, 2], 1, True)
        print ('disp3:')
        print (disp3.get_shape().as_list())

    with tf.variable_scope("upconv2"):
        upconv2 = conv2d_transpose(iconv3, [4, 4, 64, 128], 2, True)
        print ('upconv2:')
        print (upconv2.get_shape().as_list())

    with tf.variable_scope("iconv2"):
        iconv2 = conv2d(tf.concat([upconv2, conv2d_transpose(disp3, [4, 4, 1, 2], 2, True),
                                   conv2a], axis=3), [3, 3, 193, 64], 1, True)
        print ('iconv2:')
        print (iconv2.get_shape().as_list())

    with tf.variable_scope("disp2_loss2"):
        disp2 = conv2d(iconv2, [3, 3, 64, 2], 1, True)
        print ('disp2:')
        print (disp2.get_shape().as_list())

    with tf.variable_scope("upconv1"):
        upconv1 = conv2d_transpose(iconv2, [4, 4, 32, 64], 2, True)
        print ('upconv1:')
        print (upconv1.get_shape().as_list())

    with tf.variable_scope("iconv1"):
        iconv1 = conv2d(tf.concat([upconv1, conv2d_transpose(disp2, [4, 4, 1, 2], 2, True), conv1a], axis=3),
                        [3, 3, 97, 32], 1, True)
        print ('iconv1:')
        print (iconv1.get_shape().as_list())

    with tf.variable_scope("disp1_loss1"):
        disp1 = conv2d(iconv1, [3, 3, 32, 2], 1, True)
        print ('disp1:')
        print (disp1.get_shape().as_list())

    with tf.variable_scope("upconv0"):
        upconv0 = conv2d_transpose(iconv1, [4, 4, 16, 32], 2, True)
        print ('upconv0:')
        print (upconv0.get_shape().as_list())

    with tf.variable_scope("iconv0"):
        iconv0 = conv2d(tf.concat([upconv0, conv2d_transpose(disp1, [4, 4, 1, 2], 2, True), up_conv1a2a], axis=3),
                        [3, 3, 49, 32], 1, True)
        print ('iconv0:')
        print (iconv0.get_shape().as_list())

    with tf.variable_scope("disp0_loss0"):
        disp0 = conv2d(iconv0, [3, 3, 32, 2], 1, True)
        print ('disp0:')
        print (disp0.get_shape().as_list())

    return disp0, disp1, disp2, disp3, disp4, disp5, disp6


def disparity_refinement(features, disp):
    conv1a = features[0]
    up_conv1a2a = features[4]
    disp0 = disp[0]
    disp1 = disp[1]
    disp2 = disp[2]

    disp0_a = tf.expand_dims(disp0[:, :, :, 0], 3)
    conv1b = generate_image_right(conv1a, tf.expand_dims(disp1[:, :, :, 1], 3))
    up_conv1b2b = generate_image_right(up_conv1a2a, tf.expand_dims(disp0[:, :, :, 1], 3))

    print(' - Disparity Refinement Sub-network')

    with tf.variable_scope("w_up_conv1b2b"):
        w_up_conv1b2b = generate_image_left(up_conv1b2b, disp0_a)
        print ('w_up_conv1b2b:')
        print (w_up_conv1b2b.get_shape().as_list())

    with tf.variable_scope("r_conv0"):
        r_conv0 = conv2d(tf.concat([tf.abs(up_conv1a2a - w_up_conv1b2b), disp0, up_conv1a2a], axis=3),
                         [3, 3, 66, 32], 1, True)
        print ('r_conv0:')
        print (r_conv0.get_shape().as_list())

    with tf.variable_scope("r_conv1"):
        r_conv1 = conv2d(r_conv0, [3, 3, 32, 64], 2, True)
        print ('r_conv1:')
        print (r_conv1.get_shape().as_list())

    with tf.variable_scope("c_conv1a"):
        c_conv1a = conv2d(conv1a, [3, 3, 64, 16], 1, True)
        print ('c_conv1a:')
        print (c_conv1a.get_shape().as_list())

    with tf.variable_scope("c_conv1b"):
        c_conv1b = conv2d(conv1b, [3, 3, 64, 16], 1, True)
        print ('c_conv1b:')
        print (c_conv1b.get_shape().as_list())

    with tf.variable_scope("r_corr"):
        r_corr = correlation_map(c_conv1a, c_conv1b, 20)
        print ('r_corr:')
        print (r_corr.get_shape().as_list())

    with tf.variable_scope("r_conv1_1"):
        r_conv1_1 = conv2d(tf.concat([r_corr, r_conv1], axis=3), [3, 3, 105, 64], 1, True)
        print ('r_conv1_1:')
        print (r_conv1_1.get_shape().as_list())

    with tf.variable_scope("r_conv2"):
        r_conv2 = conv2d(r_conv1_1, [3, 3, 64, 128], 2, True)
        print ('r_conv2:')
        print (r_conv2.get_shape().as_list())

    with tf.variable_scope("r_conv2_1"):
        r_conv2_1 = conv2d(r_conv2, [3, 3, 128, 128], 1, True)
        print ('r_conv2_1:')
        print (r_conv2_1.get_shape().as_list())

    with tf.variable_scope("r_res2"):
        r_res2 = conv2d(tf.concat([r_conv2_1, disp2], axis=3), [3, 3, 130, 1], 1, True)
        print ('r_res2:')
        print (r_res2.get_shape().as_list())

    with tf.variable_scope("r_upconv1"):
        r_upconv1 = conv2d_transpose(r_conv2_1, [4, 4, 64, 128], 2, True)
        print ('r_upconv1:')
        print (r_upconv1.get_shape().as_list())

    with tf.variable_scope("r_iconv1"):
        r_iconv1 = conv2d(tf.concat([r_upconv1, conv2d_transpose(r_res2, [4, 4, 2, 1], 2, True), r_conv1_1],
                                    axis=3), [3, 3, 130, 64], 1, True)
        print ('r_iconv1:')
        print (r_iconv1.get_shape().as_list())

    with tf.variable_scope("r_res1"):
        r_res1 = conv2d(tf.concat([r_iconv1, disp1], axis=3), [3, 3, 66, 1], 1, True)
        print ('r_res1:')
        print (r_res1.get_shape().as_list())

    with tf.variable_scope("r_upconv0"):
        r_upconv0 = conv2d_transpose(r_iconv1, [4, 4, 32, 64], 2, True)
        print ('r_upconv0:')
        print (r_upconv0.get_shape().as_list())

    with tf.variable_scope("r_iconv0"):
        r_iconv0 = conv2d(tf.concat([r_upconv0, conv2d_transpose(r_res1, [4, 4, 2, 1], 2, True), r_conv0],
                                    axis=3), [3, 3, 66, 32], 1, True)
        print ('r_iconv0:')
        print (r_iconv0.get_shape().as_list())

    with tf.variable_scope("r_res0"):
        r_res0 = conv2d(tf.concat([r_iconv0, disp0], axis=3), [3, 3, 34, 1], 1, True)
        print ('r_res0:')
        print (r_res0.get_shape().as_list())

    return r_res0, r_res1, r_res2
