import tensorflow as tf
import tensorflow.contrib as tc
from baselines.ddpg.models import Actor, Critic
from math import sqrt

class MyActor(Actor):
    def __init__(self, nb_actions, name='actor', layer_norm=True, architecture = 'Divyam'):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.architecture = architecture

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            if self.architecture == 'Divyam':
                layers_shape = [400, 300]
                obs_len = obs.get_shape()[1].value
                x = obs
                ki=tf.random_uniform_initializer(
                        minval=-1/sqrt(obs_len), 
                        maxval= 1/sqrt(obs_len))
                x = tf.layers.dense(x, layers_shape[0], kernel_initializer=ki)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
                
                ki=tf.random_uniform_initializer(
                        minval=-1/sqrt(layers_shape[0]), 
                        maxval= 1/sqrt(layers_shape[0]))
                x = tf.layers.dense(x, layers_shape[1], kernel_initializer=ki)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
                
                x = tf.layers.dense(x, self.nb_actions, kernel_initializer=
                                    tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = tf.nn.tanh(x)
            elif self.architecture == '64x64':
                x = self.create(obs, [64, 64])
            elif self.architecture == '400x300':
                x = self.create(obs, [400, 300])
            else:
                raise ValueError('Unknown architecture specified: %s' % self.architecture)
        return x
    
    def create(self, obs, layers_shape=[64, 64]):
        x = obs
        x = tf.layers.dense(x, self.layers_shape[0])
        if self.layer_norm:
            x = tc.layers.layer_norm(x, center=True, scale=True)
        x = tf.nn.relu(x)
        
        x = tf.layers.dense(x, self.layers_shape[1])
        if self.layer_norm:
            x = tc.layers.layer_norm(x, center=True, scale=True)
        x = tf.nn.relu(x)
        
        x = tf.layers.dense(x, self.nb_actions, kernel_initializer=
                            tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        x = tf.nn.tanh(x)
        return x


class MyCritic(Critic):
    def __init__(self, name='critic', layer_norm=True, architecture = 'Divyam'):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.architecture = architecture


    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            if self.architecture == 'Divyam':
                layers_shape = [400, 300]
                
                obs_len = obs.get_shape()[1].value
                action_len = action.get_shape()[1].value
                
                x = obs
                ki=tf.random_uniform_initializer(
                        minval=-1/sqrt(obs_len), 
                        maxval= 1/sqrt(obs_len))
                x = tf.layers.dense(x, layers_shape[0], kernel_initializer=ki)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
                
                ki=tf.random_uniform_initializer(
                        minval=-1/sqrt(layers_shape[0]+action_len), 
                        maxval= 1/sqrt(layers_shape[0]+action_len))
                x = tf.layers.dense(x, layers_shape[1], kernel_initializer=ki, use_bias=False)
    
                y = action
                ki=tf.random_uniform_initializer(
                        minval=-1/sqrt(layers_shape[0]+action_len), 
                        maxval= 1/sqrt(layers_shape[0]+action_len))
                y = tf.layers.dense(y, layers_shape[1], kernel_initializer=ki)
                
                z = x + y
                if self.layer_norm:
                    z = tc.layers.layer_norm(z, center=True, scale=True)
                z = tf.nn.relu(z)
                
                z = tf.layers.dense(z, 1, kernel_initializer=
                                    tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            elif self.architecture == '64x64':
                z = self.create(obs, action, [64, 64])
            elif self.architecture == '400x300':
                z = self.create(obs, action, [400, 300])
            else:
                raise ValueError('Unknown architecture specified: %s' % self.architecture)
        return z
    
    def create(self, obs, action, layers_shape=[64, 64]):
        x = obs
        x = tf.layers.dense(x, self.layers_shape[0])
        if self.layer_norm:
            x = tc.layers.layer_norm(x, center=True, scale=True)
        x = tf.nn.relu(x)
        
        x = tf.concat([x, action], axis=-1)
        x = tf.layers.dense(x, self.layers_shape[1])
        if self.layer_norm:
            x = tc.layers.layer_norm(x, center=True, scale=True)
        x = tf.nn.relu(x)
        
        x = tf.layers.dense(x, 1, kernel_initializer=
                            tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x
