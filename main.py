import tensorflow as tf
import vkge
from utils import pp


flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of adam optimizer [0.001]")
flags.DEFINE_float("beta1", 0.9, "Beta 1 of learning rate [0.9]")
flags.DEFINE_float("beta2", 0.99, "Beta 2 of learning rate [0.99]")
flags.DEFINE_float("init_sig", 6.0, "Initalised variance of variables [6.0]")
flags.DEFINE_float("init_sig2", 6.0, "Initalised variance of variables [6.0]")
flags.DEFINE_float("mog_split", 0.5, "Split between spike and slab [6.0]")
flags.DEFINE_float("epsilon", 1e-08, "Epsilon of learning rate [1e-08]")
flags.DEFINE_integer("embedding_size", 50, "The dimension of graph embeddings [50]")
flags.DEFINE_integer("batch_size", 14145, "Batch Size [14145]")
flags.DEFINE_boolean("alt_cost", True, "Switch for compression cost to be used in training [True]")
flags.DEFINE_boolean("train_mean", False, "Switch as to if mean is fixed at 0 or can train with random normal init [False]")
flags.DEFINE_boolean("alternating_updates", False, "Alternate updates around each distribution[False]")
FLAGS = flags.FLAGS



def main(_):
    pp.pprint(flags.FLAGS.__flags)

    vkge.VKGE(embedding_size=FLAGS.embedding_size,lr=FLAGS.learning_rate,b1=FLAGS.beta1,b2=FLAGS.beta2,eps=FLAGS.epsilon,ent_sig=FLAGS.init_sig,alt_cost=FLAGS.alt_cost,batch_s=FLAGS.batch_size,
              train_mean=FLAGS.train_mean,alt_updates=FLAGS.alternating_updates)


if __name__ == '__main__':
    tf.app.run()