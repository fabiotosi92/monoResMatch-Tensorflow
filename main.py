import argparse
import tensorflow as tf
import time
import numpy as np
import os
import cv2
from model import *
from dataloader import *
from utils import *
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Argument parser')

"""Arguments related to run mode"""
parser.add_argument('--is_training',  help='train, test', action='store_true')
parser.add_argument('--test_single',  help='try on a single image', action='store_true')
parser.add_argument('--post_process',  help='if able the post process is during testing applied', action='store_true')
parser.add_argument('--cpu',  help='the network runs on CPU if enabled', action='store_true')

"""Arguments related to training"""
parser.add_argument('--iterations',  dest='iterations', type=int, default=300000, help='# of iterations')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='# images in batch')
parser.add_argument('--patch_width', dest='patch_width', type=int, default=512, help='# images in patches')
parser.add_argument('--patch_height', dest='patch_height', type=int, default=256, help='# images in patches')
parser.add_argument('--width', dest='width', type=int, default=1280, help='# image height')
parser.add_argument('--height', dest='height', type=int, default=384, help='# image width')
parser.add_argument('--initial_learning_rate', dest='initial_learning_rate', type=float, default=0.0001, help='initial learning rate for gradient descent')
parser.add_argument('--learning_rate_scale_factor', dest='learning_rate_scale_factor', type=float, default=2.0, help='lr will be reduced to lr/learning_rate_scale_factor every N steps')
parser.add_argument('--learning_rate_schedule', type=str, help='Enter the list of steps in which the learning rate will be reduced')
parser.add_argument('--num_threads', dest='num_threads', type=int, default=4, help='num_threads')
parser.add_argument('--scales_initial', dest='scales_initial', type=int, default=4, help='number of considered scales during the initial disparity loss computation')
parser.add_argument('--scales_refined', dest='scales_refined', type=int, default=3, help='number of considered scales during the disparity refinement loss computation')
parser.add_argument('--max_to_keep', dest='max_to_keep', type=int, default=5, help='indicates the maximum number of recent checkpoint files to keep')

"""Arguments related to dataset"""
parser.add_argument('--dataset', dest='dataset', type=str, default='kitti', help='name dataset')
parser.add_argument('--data_path_image',  dest='data_path_image', type=str, default='', help='dataset path image')
parser.add_argument('--data_path_proxy',  dest='data_path_proxy', type=str, default='', help='dataset path proxy')
parser.add_argument('--filenames_file',  dest='filenames_file', type=str, default='./utils/filenames/kitti_train_files.txt', help='filenames_file path')
parser.add_argument('--image_path',  dest='image_path', type=str, default='./test/example0.png', help='single image path')

"""Arguments related to monitoring and outputs"""
parser.add_argument('--log_directory', dest='log_directory', type=str, default='./log', help='directory to save checkpoints and summaries')
parser.add_argument('--checkpoint_path', dest='checkpoint_path', type=str, default='', help='path to a specific checkpoint to load' )
parser.add_argument('--save_iter_freq', dest='save_iter_freq', type=int, default=5000, help='save a model every save_iter_freq steps (does not overwrite previously saved models)')
parser.add_argument('--model_name', dest='model_name', type=str, default='model', help='model name')
parser.add_argument('--output_path', dest='output_path', type=str, default='./output', help='model name')
parser.add_argument('--save_colored',  help='output test images will be saved in output_path/images_colored', action='store_true')
parser.add_argument('--save_images',  help='output test images will be saved in output_path/images', action='store_true')
parser.add_argument('--display_factor', dest='display_factor', type=float, default=2.0, help='display_factor scales the output disparity map for visualization only')
parser.add_argument('--display_step', dest='display_step', type=int, default=100, help='every display_step the training current state will be printed and the summary updated')
parser.add_argument('--retrain', help='if used with checkpoint_path, will restart training from step zero', action='store_true')

"""Arguments related to losses"""
parser.add_argument('--alpha_SSIM_L1', type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--alpha_image_loss', type=float, help='image loss weigth', default=1.0)
parser.add_argument('--alpha_proxy_loss', type=float, help='proxy loss weigth', default=1.0)
parser.add_argument('--alpha_smoothness_loss', type=float, help='disparity smoothness weigth', default=0.1)
args = parser.parse_args()

if args.cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def configure_parameters():

    network_params = network_parameters(
        patch_height=args.patch_height,
        patch_width=args.patch_width,
        alpha_SSIM_L1=args.alpha_SSIM_L1,
        alpha_image_loss=args.alpha_image_loss,
        alpha_proxy_loss=args.alpha_proxy_loss,
        alpha_smoothness_loss=args.alpha_smoothness_loss,
        scales_initial=args.scales_initial,
        scales_refined=args.scales_refined,
        display_factor=args.display_factor)

    dataloader_params = dataloader_parameters(
        patch_height=args.patch_height,
        patch_width=args.patch_width,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        num_threads=args.num_threads)

    return network_params, dataloader_params


def configure_network(network_params, dataloader_params):
    dataloader = Dataloader(args.data_path_image,
                            args.data_path_proxy,
                            args.filenames_file,
                            args.dataset,
                            args.is_training,
                            args.test_single,
                            args.image_path,
                            args.post_process,
                            dataloader_params)

    network = Network(dataloader.left_image_batch,
                      dataloader.right_image_batch,
                      dataloader.proxy_left_batch,
                      dataloader.proxy_right_batch,
                      args.is_training,
                      ['monoResMatch'],
                      network_params)
    return network, dataloader


def makedirs(cond, pp, output_path, name):
    if cond:
        if not os.path.exists(os.path.join(output_path, name)):
            os.makedirs(os.path.join(output_path, name, "raw"))
        if pp and not os.path.exists(os.path.join(output_path, name, "pp")):
            os.makedirs(os.path.join(output_path, name, "pp"))


def train(network):
    print(" [*] Training....")

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    if not os.path.exists(args.log_directory):
        os.makedirs(args.log_directory)

    learning_rate = tf.placeholder(tf.float32, shape=[])
    training_flag = tf.placeholder(tf.bool)
    learning_rate_schedule = [int(i) for i in args.learning_rate_schedule.split(',')]
    tf.summary.scalar('learning_rate', learning_rate, collections=network.model_collection)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(network.loss)
    saver = tf.train.Saver(max_to_keep=args.max_to_keep)
    summary_op = tf.summary.merge_all(network.model_collection[0])
    writer = tf.summary.FileWriter(args.log_directory + "/summary/", graph=sess.graph)

    global_step = tf.Variable(0, trainable=False)
    total_num_parameters = 0
    vars = [k for k in tf.trainable_variables()]
    for variable in vars:
        total_num_parameters += np.array(variable.get_shape().as_list()).prod()

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord=coordinator)
    print(" [*] Number of trainable parameters: {}".format(total_num_parameters))

    print(' [*] Training data loaded successfully')
    lr = args.initial_learning_rate

    if args.checkpoint_path != '':
        saver.restore(sess, args.checkpoint_path)
        print(" [*] Load model: SUCCESS")
        if args.retrain:
            sess.run(global_step.assign(0))
        else:
            sess.run(global_step.assign(int(os.path.basename(args.checkpoint_path).split("-")[1])))

    start_step = global_step.eval(session=sess)

    print(" [*] Start Training...")
    for step in range(start_step, args.iterations):
        before_op_time = time.time()
        _, loss = sess.run([optimizer, network.loss], feed_dict={learning_rate: lr, training_flag: True})
        duration = time.time() - before_op_time

        if step and step % args.display_step == 0:
            examples_per_sec = args.batch_size / duration
            training_time_left = ((args.iterations - step)/examples_per_sec) * args.batch_size / 3600.0

            print("Step: [%2d]" % step + "/[%2d]" % args.iterations + ", Loss: [%2f]" % loss +
                  ", Examples/s: [%2f]"% examples_per_sec + ", Time left: [%2f]" % training_time_left)

            summary_str = sess.run(summary_op, feed_dict={learning_rate: lr, training_flag: True})
            writer.add_summary(summary_str, global_step=step)

        if step % args.save_iter_freq == 0:
            saver.save(sess, args.log_directory + '/' + args.model_name, global_step=step)

        if step in learning_rate_schedule:
            lr = lr/args.learning_rate_scale_factor

    saver.save(sess, args.log_directory + '/' + args.model_name, global_step=args.iterations)

    print('[*] done')

    coordinator.request_stop()
    coordinator.join(threads)


def test(network, dataloader):
    print("\n [*] Testing....")

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    makedirs(args.save_colored, args.post_process, args.output_path, "disp_colored")
    makedirs(args.save_images, args.post_process, args.output_path, "disp")
    makedirs(True, args.post_process, args.output_path, "npy")

    training_flag = tf.placeholder(tf.bool)
    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
              tf.local_variables_initializer())
    sess.run(init_op)

    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    if args.checkpoint_path != '':
        saver.restore(sess, args.checkpoint_path)
        print(" [*] Load model: SUCCESS")

    num_test_samples = 1 if args.test_single else count_text_lines(args.filenames_file)

    for step in range(num_test_samples):
        before_op_time = time.time()
        disp, height, width = sess.run([network.disp_left_est_refined[0], dataloader.image_h,
                                        dataloader.image_w], feed_dict={training_flag: False})
        if args.post_process:
            disparity_pp=post_process_disparity(disp.squeeze())
        duration = time.time() - before_op_time
        examples_per_sec = args.batch_size / duration
        filename = os.path.basename(args.image_path).split('.')[0] if args.test_single else str(step)

        # Resize and scale images
        disp = cv2.resize(disp[0], (width, height), interpolation=cv2.INTER_LINEAR) * (width/args.width)
        if args.post_process:
            disp_pp = cv2.resize(disparity_pp, (width, height), interpolation=cv2.INTER_LINEAR) * (width / args.width)

        # Save npy
        np.save(os.path.join(args.output_path, "npy", "raw", filename + '.npy'), np.array(disp))
        if args.post_process:
            np.save(os.path.join(args.output_path, "npy", "pp", filename + '.npy'), np.array(disp_pp))

        # Save png
        if args.save_colored:
            plt.imsave(os.path.join(args.output_path, "disp_colored", "raw", filename + '.png'),
                       disp * args.display_factor, cmap='plasma')
        if args.save_images:
            cv2.imwrite(os.path.join(args.output_path, "disp", "raw", filename + '.png'), disp)

        if args.post_process:
            if args.save_colored:
                plt.imsave(os.path.join(args.output_path, "disp_colored", "pp", filename + '.png'),
                           disp_pp * args.display_factor, cmap='plasma')
            if args.save_images:
                cv2.imwrite(os.path.join(args.output_path, "disp", "pp", filename + '.png'), disp_pp)

        print("Test image [%2d]" % step + "/[%2d]" % num_test_samples + ", Examples/s: [%2f]" % examples_per_sec)

    print('[*] done')
    print('[*] writing disparities.')
    print('done.')

    coordinator.request_stop()
    coordinator.join(threads)


def main(_):

        network_params, dataloader_params = configure_parameters()
        monoResMatch, dataloader = configure_network(network_params, dataloader_params)

        if args.is_training:
            train(monoResMatch)
        else:
            test(monoResMatch, dataloader)


if __name__ == '__main__':
    tf.app.run()
