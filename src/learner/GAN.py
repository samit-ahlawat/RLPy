import tensorflow as tf
import time
import logging


class GAN(object):
    def __init__(self, generator, discriminator, latent_space_len, learning_rate=1E-4, loss=None):
        assert isinstance(generator, tf.keras.Model)
        assert isinstance(discriminator, tf.keras.Model)
        self.generator = generator
        self.discriminator = discriminator
        self.latentSpaceLen = latent_space_len
        self.generatorOptimizer = tf.keras.optimizers.Adam(learning_rate)
        self.discriminatorOptimizer = tf.keras.optimizers.Adam(learning_rate)
        self.lossFn = loss
        if loss is None:
            self.lossFn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.logger = logging.getLogger(self.__class__.__name__)

    def runDiscriminator(self):
        noise = tf.random.normal([1, self.latentSpaceLen])
        generated_image = self.generator(noise, training=False)
        decision = self.discriminator(generated_image)
        print(decision)
        return decision.numpy()

    def discriminatorLoss(self, real_output, fake_output):
        real_loss = self.lossFn(tf.ones_like(real_output), real_output)
        fake_loss = self.lossFn(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generatorLoss(self, fake_output):
        return self.lossFn(tf.ones_like(fake_output), fake_output)

    @tf.function
    def trainStep(self, images):
        noise = tf.random.normal([images.shape[0], self.latentSpaceLen])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generatorLoss(fake_output)
            disc_loss = self.discriminatorLoss(real_output, fake_output)

        grad_gen = gen_tape.gradient(gen_loss, self.generator.trainable_weights)
        grad_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)

        self.generatorOptimizer.apply_gradients(zip(grad_gen, self.generator.trainable_weights))
        self.discriminatorOptimizer.apply_gradients((zip(grad_disc, self.discriminator.trainable_weights)))

    def fit(self, dataset, epochs):
        start = time.time()
        for epoch in range(epochs):
            for image_batch in dataset:
                self.trainStep(image_batch)
        end = time.time()
        self.logger.info("Took %f seconds", end - start)
