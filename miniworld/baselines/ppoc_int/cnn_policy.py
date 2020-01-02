import baselines.common.tf_util as U
import numpy as np
import tensorflow as tf
import gym, gym_miniworld
from baselines.common.distributions import make_pdtype
import  pdb


def dense3D2(x, size, name, option, num_options=1, weight_init=None, bias=True):
    w = tf.get_variable(name + "/w", [num_options, x.get_shape()[1], size], initializer=weight_init)
    ret = tf.matmul(x, w[option[0]])
    if bias:
        b = tf.get_variable(name + "/b", [num_options,size], initializer=tf.zeros_initializer())
        return ret + b[option[0]]

    else:
        return ret


class CnnPolicy(object):
    recurrent = False
    def __init__(self, name, ob_space, ac_space, kind='large', num_options=2, dc=0, w_intfc=True):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, kind, num_options,dc,w_intfc)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, kind, num_options=2, dc=0, w_intfc=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.w_intfc = w_intfc
        self.state_in = []
        self.state_out = []
        self.dc = dc
        self.num_options = num_options
        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        option = U.get_placeholder(name="option", dtype=tf.int32, shape=[None])

        x = ob / 255.0
        if kind == 'small':  # from A3C paper
            x = tf.nn.relu(U.conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
            x = U.flattenallbut0(x)
            hidden = tf.nn.relu(tf.layers.dense(x, 256, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        elif kind == 'large':  # Nature DQN
            x = tf.nn.relu(U.conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
            x = U.flattenallbut0(x)
            hidden = tf.nn.relu(tf.layers.dense(x, 512, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        else:
            raise NotImplementedError


        logits = dense3D2(hidden, pdtype.param_shape()[0], "polfinal", option, num_options=num_options, weight_init=U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(logits)

        self.vpred = dense3D2(hidden, 1, "vffinal", option, num_options=num_options, weight_init=U.normc_initializer(1.0))[:, 0]

        self.tpred = tf.nn.sigmoid(dense3D2(tf.stop_gradient(hidden), 1, "termhead", option, num_options=num_options, weight_init=U.normc_initializer(1.0)))[:, 0]
        termination_sample = tf.greater(self.tpred, tf.random_uniform(shape=tf.shape(self.tpred), maxval=1.))

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample()  # XXX

        #self.op_pi = tf.nn.softmax(U.dense(tf.stop_gradient(hidden), num_options, "OP", weight_init=U.normc_initializer(1.0)))

        self.op_pi = tf.nn.softmax(U.dense(hidden, num_options, "OPfinal", weight_init=U.normc_initializer(1.0)))

        self.intfc = tf.sigmoid(U.dense(hidden, num_options, "intfcfinal", weight_init=U.normc_initializer(1.0)))

        self._act = U.function([stochastic, ob, option], [ac])
        self.get_term = U.function([ob, option], [termination_sample])
        self.get_tpred = U.function([ob, option], [self.tpred])
        self.get_vpred = U.function([ob, option], [self.vpred])
        self._get_op_int = U.function([ob], [self.op_pi, self.intfc])
        self._get_intfc = U.function([ob], [self.intfc])
        self._get_op = U.function([ob], [self.op_pi])


    def act(self, stochastic, ob, option):
        ac1 = self._act(stochastic, ob[None], [option])
        return ac1[0]


    def get_int_func(self, obs):
        return self._get_intfc(obs)[0]


    def get_alltpreds(self, obs, ob):
        obs = np.vstack((obs, ob[None]))

        betas = []
        for opt in range(self.num_options):
            betas.append(self.get_tpred(obs, [opt])[0])
        betas = np.array(betas).T

        return betas[:-1], betas[-1]


    def get_allvpreds(self, obs, ob):
        obs = np.vstack((obs, ob[None]))

        # Get Q(s,w)
        vals = []
        for opt in range(self.num_options):
            vals.append(self.get_vpred(obs, [opt])[0])
        vals = np.array(vals).T

        op_prob, int_func = self._get_op_int(obs)
        #op_prob = np.ones_like(op_prob) * (1./self.num_options)
        if self.w_intfc:
            pi_I = op_prob * int_func / np.sum(op_prob * int_func, axis=1)[:, None]
        else:
            pi_I = op_prob
        op_vpred = np.sum((pi_I * vals), axis=1) # Get V(s)

        return vals[:-1], op_vpred[:-1], vals[-1], op_vpred[-1], op_prob[:-1], int_func[:-1]

    def get_vpreds(self,obs):


        # Get Q(s,w)
        vals = []
        for opt in range(self.num_options):
            vals.append(self.get_vpred(obs,[opt])[0])
        vals=np.array(vals).T
        return vals

    def get_option(self, ob):
        op_prob, int_func = self._get_op_int([ob])
        #op_prob = np.ones_like(op_prob) * (1. / self.num_options)
        if self.w_intfc:
            pi_I = op_prob * int_func / np.sum(op_prob * int_func, axis=1)[:, None]
        else:
            pi_I = op_prob

        return np.random.choice(range(len(op_prob[0])), p=pi_I[0])


    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []




