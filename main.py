import tensorflow as tf
import keras as k
import math
from keras import optimizers
from

learning_rate = 1e-3
optimizer = optimizers.SGD(learning_rate=1e-3)

class SimpleDense:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation

        # Create a weight matrix of shape:
        w_shape = (input_size, output_size)
        # Initialize W matrix with random values
        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
        self.W = tf.Variable(w_initial_value)

        # Vector b initialized with zeros
        b_shape = (output_size)
        b_initial_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_initial_value)

    def __call__(self, inputs):
        # Apply the forward pass - output = activation(dot(W, input) + b)
        return self.activation(tf.matmul(inputs, self.W) + self.b)

    @property
    def weights(self):
        # Retrieving the layer's weights
        return [self.W, self.b]


class SimpleSequential:
    # It wraps a list of layers and exposes a __call__() method that simply calls the underlying layers on the
    # inputs, in order. It also features a weights property to easily keep track of the layers’ parameters.
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights


class BatchGenerator:
    # To iterate over MINST data
    def __init__(self, images, labels, batch_size = 128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        images = self.images[self.index: self.index + self.batch_size]
        labels = self.labels[self.index: self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels


def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        per_sample_losses = k.losses.sparse_categorical_crossentropy(labels_batch, predictions)
        average_loss = tf.reduce_mean(per_sample_losses)

    # Compute the gradient of the loss with regard to the weights. The output gradients
    # is a list where each entry corresponds to a weight from the model.weights list.
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)

    return average_loss

def update_weights(gradients, weights):
    for g, w in zip(gradients, weights):
        w.assign_sub(g * learning_rate) # assign_sub is same as -=

def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels)

        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)

            if batch_counter % 100 == 0:
                print(f"loss at batch {batch_counter}: {loss:.2f}")

if __name__ == '__main__':




