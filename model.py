import tensorflow as tf
from bilinear_sampler import *
from monoResMatch import *
from collections import namedtuple
from utils import *
from losses import *

network_parameters = namedtuple('network_parameters',
                        'patch_height, patch_width, '
                        'alpha_SSIM_L1, '
                        'alpha_image_loss, '
                        'alpha_proxy_loss, '
                        'alpha_smoothness_loss, '
                        'scales_initial, scales_refined, '
                        'display_factor')


class Network(object):

    def __init__(self, left, right, proxy_left, proxy_right, is_training, model_collection, params):
        self.model_collection = model_collection
        self.is_training = is_training
        self.params = params

        self.left = left
        self.right = right
        self.proxy_left = proxy_left
        self.proxy_right = proxy_right

        self.build_network()
        self.build_output()

        if self.is_training:
            self.build_losses()
            self.build_summaries()

    def build_network(self):
        with tf.variable_scope('monoResMatch'):
            with tf.variable_scope("Stem_Block"):
                self.features = stem_block(self.left)

            with tf.variable_scope("Disparity_Estimation"):
                self.disp = disparity_estimation(self.features)

            with tf.variable_scope("Disparity_Refinement"):
                self.disp_refined = disparity_refinement(self.features, self.disp)

    def build_output(self):
        s = tf.shape(self.left)
        h = tf.to_int32(s[1])
        w = tf.to_int32(s[2])

        with tf.variable_scope('output'):
            print(" [*] Building outputs...")
            self.disp_est = [tf.image.resize_images(self.disp[i] * (2 ** i), [h, w]) for i in range(self.params.scales_initial)]
            self.disp_est_refined = [tf.image.resize_images(self.disp_refined[i] * (2 ** i), [h, w]) for i in range(self.params.scales_refined)]

            self.disp_left_est = [tf.expand_dims(d[:, :, :, 0], 3) for d in self.disp_est]
            self.disp_right_est = [tf.expand_dims(d[:, :, :, 1], 3) for d in self.disp_est]
            self.disp_left_est_refined = [tf.expand_dims(d[:, :, :, 0], 3) for d in self.disp_est_refined]

        if self.is_training:

            [self.disp_left_est[i].set_shape([None, self.params.patch_height, self.params.patch_width, 1]) for i in range(self.params.scales_initial)]
            [self.disp_right_est[i].set_shape([None, self.params.patch_height, self.params.patch_width, 1]) for i in range(self.params.scales_initial)]
            [self.disp_left_est_refined[i].set_shape([None, self.params.patch_height, self.params.patch_width, 1]) for i in range(self.params.scales_refined)]

            with tf.variable_scope('warped_images'):
                self.left_est = [generate_image_left(self.right, self.disp_left_est[i]) for i in range(self.params.scales_initial)]
                self.right_est = [generate_image_right(self.left, self.disp_right_est[i]) for i in range(self.params.scales_initial)]

            with tf.variable_scope('warped_images_refined'):
                self.left_est_refined = [generate_image_left(self.right, self.disp_left_est_refined[i]) for i in range(self.params.scales_refined)]

            with tf.variable_scope('smoothness'):
                self.disp_left_smoothness = [get_disparity_smoothness(self.disp_left_est[i], self.left) for i in range(self.params.scales_initial)]
                self.disp_right_smoothness = [get_disparity_smoothness(self.disp_right_est[i], self.right) for i in range(self.params.scales_initial)]
                self.disp_left_smoothness_refined = [get_disparity_smoothness(self.disp_left_est_refined[i], self.left) for i in range(self.params.scales_refined)]

    def build_losses(self):
        with tf.variable_scope('loss'):
            print(" [*] Building losses...")

            # IMAGE LOSS
            self.l1_loss_left = [tf.reduce_mean(tf.abs(self.left_est[i] - self.left)) for i in range(self.params.scales_initial)]
            self.l1_loss_right = [tf.reduce_mean(tf.abs(self.right_est[i] - self.right)) for i in range(self.params.scales_initial)]
            self.l1_loss_left_refined = [tf.reduce_mean(tf.abs(self.left_est_refined[i] - self.left)) for i in range(self.params.scales_refined)]

            self.ssim_loss_left = [tf.reduce_mean(SSIM(self.left_est[i], self.left)) for i in range(self.params.scales_initial)]
            self.ssim_loss_right = [tf.reduce_mean(SSIM(self.right_est[i], self.right)) for i in range(self.params.scales_initial)]
            self.ssim_loss_left_refined = [tf.reduce_mean(SSIM(self.left_est_refined[i], self.left)) for i in range(self.params.scales_refined)]

            self.image_loss_left = [self.params.alpha_SSIM_L1 * self.ssim_loss_left[i] + (1 - self.params.alpha_SSIM_L1) * self.l1_loss_left[i] for i in range(self.params.scales_initial)]
            self.image_loss_right = [self.params.alpha_SSIM_L1 * self.ssim_loss_right[i] + (1 - self.params.alpha_SSIM_L1) * self.l1_loss_right[i] for i in range(self.params.scales_initial)]
            self.image_loss_left_refined = [self.params.alpha_SSIM_L1 * self.ssim_loss_left_refined[i] + (1 - self.params.alpha_SSIM_L1) * self.l1_loss_left_refined[i] for i in range(self.params.scales_refined)]

            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)
            self.image_loss_refined = tf.add_n(self.image_loss_left_refined)

            # DISPARITY SMOOTHNESS
            self.disp_loss_left = [tf.reduce_mean(tf.abs(self.disp_left_smoothness[i]))  for i in range(self.params.scales_initial)]
            self.disp_loss_right = [tf.reduce_mean(tf.abs(self.disp_right_smoothness[i])) for i in range(self.params.scales_initial)]
            self.disp_loss_left_refined = [tf.reduce_mean(tf.abs(self.disp_left_smoothness_refined[i])) for i in range(self.params.scales_refined)]
            self.disp_gradient_loss = tf.add_n(self.disp_loss_left + self.disp_loss_right)
            self.disp_gradient_loss_refined = tf.add_n(self.disp_loss_left_refined)

            # PROXY LOSS
            mask_left = tf.cast(self.proxy_left > 0.0, tf.float32)
            mask_right = tf.cast(self.proxy_right > 0.0, tf.float32)
            valid_points_left = tf.maximum(tf.reduce_sum(mask_left), tf.reduce_sum(tf.ones([1])))
            valid_points_right = tf.maximum(tf.reduce_sum(mask_right), tf.reduce_sum(tf.ones([1])))

            self.proxy_loss_left = [berhu_loss(self.proxy_left, self.disp_left_est[i]) for i in range(self.params.scales_initial)]
            self.proxy_loss_right= [berhu_loss(self.proxy_right, self.disp_right_est[i]) for i in range(self.params.scales_initial)]
            self.proxy_loss_left_refined = [berhu_loss(self.proxy_left, self.disp_left_est_refined[i]) for i in range(self.params.scales_refined)]

            self.reconstruction_proxy_loss_left = [tf.reduce_sum(l * mask_left)/valid_points_left for l in self.proxy_loss_left]
            self.reconstruction_proxy_loss_right = [tf.reduce_sum(l * mask_right)/valid_points_right for l in self.proxy_loss_right]
            self.reconstruction_proxy_loss_left_refined = [tf.reduce_sum(l * mask_left)/valid_points_left for l in self.proxy_loss_left_refined]

            self.proxy_loss = tf.add_n(self.reconstruction_proxy_loss_left + self.reconstruction_proxy_loss_right)
            self.proxy_loss_refined = tf.add_n(self.reconstruction_proxy_loss_left_refined)

            # TOTAL LOSS
            self.loss = self.params.alpha_proxy_loss * (self.proxy_loss + self.proxy_loss_refined) \
                        + self.params.alpha_image_loss * (self.image_loss + self.image_loss_refined)\
                        + self.params.alpha_smoothness_loss * (self.disp_gradient_loss + self.disp_gradient_loss_refined)

    def build_summaries(self):
        tf.summary.image('right', self.right, collections=self.model_collection)
        tf.summary.image('left', self.left, collections=self.model_collection)
        tf.summary.image('proxy_left', color_disparity(self.proxy_left * self.params.display_factor), collections=self.model_collection)
        tf.summary.image('proxy_right', color_disparity(self.proxy_right * self.params.display_factor), collections=self.model_collection)
        tf.summary.scalar('image_loss', self.image_loss, collections=self.model_collection)
        tf.summary.scalar('image_loss_refined', self.image_loss_refined, collections=self.model_collection)
        tf.summary.scalar('proxy_loss', self.proxy_loss, collections=self.model_collection)
        tf.summary.scalar('proxy_loss_refined', self.proxy_loss_refined, collections=self.model_collection)
        tf.summary.scalar('total_loss', self.loss, collections=self.model_collection)

        for i in range(self.params.scales_initial):
            tf.summary.image('disparity_left' + str(i), color_disparity(self.disp_left_est[i] * self.params.display_factor), collections=self.model_collection)
            tf.summary.image('disparity_right' + str(i), color_disparity(self.disp_right_est[i] * self.params.display_factor), collections=self.model_collection)
            tf.summary.image('left_est_' + str(i), self.left_est[i], collections=self.model_collection)
            tf.summary.image('right_est_' + str(i), self.right_est[i], collections=self.model_collection)
            tf.summary.scalar('proxy_loss_left' + str(i), self.reconstruction_proxy_loss_left[i], collections=self.model_collection)
            tf.summary.scalar('proxy_loss_right' + str(i), self.reconstruction_proxy_loss_right[i], collections=self.model_collection)
            tf.summary.scalar('image_loss_left_' + str(i), self.image_loss_left[i], collections=self.model_collection)
            tf.summary.scalar('image_loss_right_' + str(i), self.image_loss_right[i], collections=self.model_collection)

        for i in range(self.params.scales_refined):
            tf.summary.image('disparity_left_refined_' + str(i), color_disparity(self.disp_left_est_refined[i] * self.params.display_factor), collections=self.model_collection)
            tf.summary.image('left_est_refined_' + str(i), self.left_est_refined[i], collections=self.model_collection)
            tf.summary.scalar('proxy_loss_left_refined' + str(i), self.reconstruction_proxy_loss_left_refined[i], collections=self.model_collection)
            tf.summary.scalar('image_loss_left_refined_' + str(i), self.image_loss_left_refined[i], collections=self.model_collection)


