from metaflow import FlowSpec, Parameter, step, retry, conda_base
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@conda_base(libraries={'tensorflow': '2.3.0', 'numpy': '1.19.5'}, python='3.8.4')
class Visualize(FlowSpec):
    img_width = Parameter(
        'img_width',
        help="image width in model",
        type=int,
        default=180,
    )

    img_height = Parameter(
        'img_height',
        help="image height in model",
        type=int,
        default=180,
    )

    iterations = Parameter(
        'iterations',
        help="number of gradient step iters",
        type=int,
        default=30,
    )

    learning_rate = Parameter(
        'learning_rate',
        help="learning rate",
        type=float,
        default=10.0,
    )


    layer_name = Parameter(
        'layer_name',
        help="Layer we visualize. More info at model.summary()",
        type=str,
        default='conv3_block4_out',
    )

    
    @step
    def start(self):
        # Set up a model that returns the activation values for our target layer
        self.model = keras.applications.ResNet50V2(weights="imagenet", include_top=False)
        self.layer = self.model.get_layer(name=self.layer_name)
        self.feature_extractor = keras.Model(inputs=self.model.inputs, outputs=self.layer.output)
        self.next(self.preprocess)

    @retry(times=1)
    @step
    def preprocess(self):
        # Compute image inpu    ts that maximize per-filter activations
        # for the first 64 filters of our target layer
        self.filter_index_list = range(1)
        self.next(self.initialize_image, foreach = 'filter_index_list')
    
    @step
    def initialize_image(self):
        # We start from a gray image with some random noise
        img = tf.random.uniform((1, self.img_width, self.img_height, 3), dtype=tf.dtypes.float32)
        # ResNet50V2 expects inputs in the range [-1, +1].
        # Here we scale our random inputs to [-0.125, +0.125]
        img = (img - 0.5) * 0.25
        for iteration in range(self.iterations):
            with tf.GradientTape() as tape:
                tape.watch(img)
                activation = self.feature_extractor(img)
                # We avoid border artifacts by only involving non-border pixels in the loss.
                filter_activation = activation[:, 2:-2, 2:-2, self.input]
                self.loss = tf.reduce_mean(filter_activation)
            # Compute gradients.
            grads = tape.gradient(self.loss, img)
            # Normalize gradients.
            grads = tf.math.l2_normalize(grads)
            img += self.learning_rate * grads
        
        # Normalize array: center on 0., ensure variance is 0.15
        img -= img.mean()
        img /= img.std() + 1e-5
        img *= 0.15

        # Center crop
        img = img[25:-25, 25:-25, :]

        # Clip to [0, 1]
        img += 0.5
        img = np.clip(img, 0, 1)

        # Convert to RGB array
        img *= 255
        self.img = np.clip(img, 0, 255).astype("uint8")

        self.next(self.join)

       
    @step
    def join(self, inputs):
        self.all_imgs = []
        for inp in inputs:
            self.all_imgs.append(inp.img)
        self.next(self.vis)

    @step
    def vis(self):
        from IPython.display import Image, display
        # Build a black picture with enough space for
        # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
        margin = 5
        n = 8
        cropped_width = self.img_width - 25 * 2
        cropped_height = self.img_height - 25 * 2
        width = n * cropped_width + (n - 1) * margin
        height = n * cropped_height + (n - 1) * margin
        stitched_filters = np.zeros((width, height, 3))

        # Fill the picture with our saved filters
        for i in range(n):
            for j in range(n):
                img = self.all_imgs[i * n + j]
                stitched_filters[
                    (cropped_width + margin) * i : (cropped_width + margin) * i + cropped_width,
                    (cropped_height + margin) * j : (cropped_height + margin) * j
                    + cropped_height,
                    :,
                ] = img
        keras.preprocessing.image.save_img("stiched_filters.png", stitched_filters)
        display(Image("stiched_filters.png"))
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':

    Visualize()
