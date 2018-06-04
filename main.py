import tensorflow as tf
import vkge

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of adam optimizer [0.001]")
flags.DEFINE_float("beta1", 0.9, "Beta 1 of learning rate [0.9]")
flags.DEFINE_float("beta2", 0.99, "Beta 2 of learning rate [0.99]")
flags.DEFINE_float("int_sig", 6.0, "Initalised variance of variables [6.0]")
flags.DEFINE_integer("embedding_size", 50, "The dimension of graph embeddings [50]")
flags.DEFINE_integer("epsilon", 1e-08, "Epsilon of learning rate [1e-08]")
flags.DEFINE_boolean("alt_cost", True, "Switch for compression cost to be used in training [True]")
FLAGS = flags.FLAGS



def main(_):
    vkge.VKGE(embedding_size=FLAGS.embedding_size,lr=FLAGS.learning_rate,b1=FLAGS.beta1,b2=FLAGS.beta2,eps=FLAGS.epsilon,ent_sig=FLAGS.int_sig,alt_cost=FLAGS.alt_cost)


if __name__ == '__main__':
    tf.app.run()