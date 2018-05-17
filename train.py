import argparse
import os
import numpy as np

import cntk as C
from cntk import Trainer
from cntk.learners import (adam, UnitType, learning_rate_schedule,
                          momentum_as_time_constant_schedule, momentum_schedule)
from cntk.logging import *

import matplotlib.pyplot as plt

from utils import *
from model import *


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--mm', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--mm_var', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--z_dim', type=int, default=64, help='dimension of seed vector. default=64')
parser.add_argument('--kt', type=float, default=0.0, help='parameter kt. default=0.0')
parser.add_argument('--gamma', type=float, default=1.0, help='parameter gamma. default=1.0')
parser.add_argument('--lamda', type=float, default=0.001, help='parameter lambda. default=0.001')
parser.add_argument('--batchSize', type=int, default=64, help='mini-batch size. default=64')
parser.add_argument('--imageSize', type=int, default=28, help='height / width of the input image to network. default=28')
parser.add_argument('--dataPath', default='Train-28x28_cntk_text.txt', help='Data path. default=Train-28x28_cntk_text.txt')

opt = parser.parse_args()

num_minibatches = 300000  # iterations
lr_update_step = 60000
print_frequency_mbsize = 1000

input_dim = opt.imageSize ** 2  # assumption: height==width
noise_shape = opt.z_dim
kt = opt.kt


if __name__=='__main__':
    check_path(opt.dataPath)
    reader_train = create_reader(opt.dataPath, True, input_dim)

    ##
    input_dynamic_axes = [C.Axis.default_batch_axis()]
    Z = C.input_variable(noise_shape, dynamic_axes=input_dynamic_axes)
    X_real = C.input_variable(input_dim, dynamic_axes=input_dynamic_axes)
    X_real_scaled = X_real / 127.5 -1

    kt_in = C.constant(kt)
    
    # Create the model function for the generator and discriminator models
    X_fake = generator(Z, opt)
    D_real = discriminator(X_real_scaled, opt)
    D_fake = D_real.clone(
        method = 'share',
        substitutions = {X_real_scaled.output: X_fake.output})
    
    # Create loss functions and configure optimazation algorithms
    D_real_loss = l1_loss(X_real_scaled, D_real)
    D_fake_loss = l1_loss(X_fake, D_fake)
    G_loss = D_fake_loss
    D_loss = D_real_loss - D_fake_loss * kt_in
    
    lr_schedule = list( opt.lr * np.asarray([0.95**t for t in range(0, num_minibatches//lr_update_step+1)]) )
    G_learner = adam(
        parameters = X_fake.parameters,
        lr = learning_rate_schedule(lr_schedule, UnitType.sample, lr_update_step*opt.batchSize),
        momentum = momentum_schedule(opt.mm),
        variance_momentum = momentum_schedule(opt.mm_var)
    )
    D_learner = adam(
        parameters = D_real.parameters,
        lr = learning_rate_schedule(lr_schedule, UnitType.sample, lr_update_step*opt.batchSize),
        momentum = momentum_schedule(opt.mm),
        variance_momentum = momentum_schedule(opt.mm_var)
    )

    pp_G = ProgressPrinter(print_frequency_mbsize, metric_is_pct=False)
    pp_D = ProgressPrinter(print_frequency_mbsize, metric_is_pct=False)
    tensorboard_logdir = 'log/'
    tb = TensorBoardProgressWriter(freq = print_frequency_mbsize, log_dir=tensorboard_logdir)
    
    # Instantiate the trainers
    G_trainer = Trainer(
        X_fake,
        (G_loss, None),
        G_learner
    )
    D_trainer = Trainer(
        D_real,
        (D_loss, None),
        D_learner
    )

    ##
    input_map = {X_real: reader_train.streams.features}
    m_global_pre = 10

    X_fake_node  = C.combine([X_fake.owner])
    sample_seed = noise_sample(25, opt.z_dim)
    exists_or_mkdir('./samples')

    for train_step in range(num_minibatches):       
        Z_data = noise_sample(opt.batchSize, opt.z_dim)
        batch_inputs = {Z: Z_data}
        G_trainer.train_minibatch(batch_inputs)
        
        Z_data = noise_sample(opt.batchSize, opt.z_dim)
        X_data = reader_train.next_minibatch(opt.batchSize, input_map)

        batch_inputs = {X_real: X_data[X_real].data, Z: Z_data}
        D_trainer.train_minibatch(batch_inputs)
        
        pp_G.update_with_trainer(G_trainer)
        pp_D.update_with_trainer(D_trainer)

        temp = C.combine([C.reduce_mean(D_real_loss, axis=C.Axis.all_axes()),
                       C.reduce_mean(D_fake_loss, axis=C.Axis.all_axes())]
                     ).eval({X_real: X_data[X_real].data, Z: Z_data})
        val = list(temp.values())

        kt = np.clip(kt + opt.lamda*(opt.gamma * val[0] - val[1]), 0, 1).astype(np.float32)
        C.assign(kt_in, kt).eval()
        m_global = val[0] + np.abs(opt.gamma * val[0] - val[1])

        tb.write_value("m_global", m_global, train_step)
        tb.write_value("kt", kt, train_step)

        if train_step % 1000 == 0:
            output = np.rint((X_fake_node.eval({Z: sample_seed}) + 1) * 127.5)
            output = np.clip(output, 0, 255)
            output = output.astype(np.uint8)
            output = np.reshape(output, [-1, opt.imageSize, opt.imageSize])
            save_images(output, [5, 5], './samples/{:05d}.png'.format(train_step))

        if train_step % 100 == 0 and m_global < m_global_pre:
            G_trainer.save_checkpoint('models/BEGAN_G_{}.dnn'.format(train_step))
            D_trainer.save_checkpoint('models/BEGAN_D_{}.dnn'.format(train_step))
            m_global_pre = m_global           


    X_fake_output = X_fake_node.eval({Z: noise_sample(36, opt.z_dim)})
    plot_images(X_fake_output, subplot_shape=[6, 6])
