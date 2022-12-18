import os, warnings, argparse, time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
warnings.filterwarnings('ignore')

import tensorflow as tf

import datamanager as dman
import neuralnet as nn
import tf_process as tfp


def main():
    dataset = dman.Dataset(normalize=FLAGS.datnorm)
    neuralnet = nn.Context_Encoder(height=dataset.height, width=dataset.width, channel=dataset.channel, \
        z_dim=FLAGS.z_dim, learning_rate=FLAGS.lr)

    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=sess_config)
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    
    time_1 = time.time()
    tfp.training(sess=sess, neuralnet=neuralnet, saver=saver, dataset=dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch, normalize=True)
    time_2 = time.time()
    tfp.test(sess=sess, neuralnet=neuralnet, saver=saver, dataset=dataset, batch_size=FLAGS.batch)
    time_3 = time.time()
    
    print("TR: ", time_2 - time_1, dataset.num_tr)
    print("TE: ", (time_3 - time_2)/dataset.num_te)

if __name__ == '__main__':
    tf.compat.v1.reset_default_graph()
    parser = argparse.ArgumentParser()
    parser.add_argument('--datnorm', type=bool, default=True, help='Data normalization')
    parser.add_argument('--z_dim', type=int, default=128, help='Dimension of latent vector')
    parser.add_argument('--lr', type=int, default=1e-4, help='Learning rate for training')
    parser.add_argument('--epoch', type=int, default=20, help='Training epoch')
    parser.add_argument('--batch', type=int, default=64, help='Mini batch size')

    FLAGS, unparsed = parser.parse_known_args()

    main()
