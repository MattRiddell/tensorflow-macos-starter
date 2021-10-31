import tensorflow as tf 
# To log device placement
# tf.debugging.set_log_device_placement(True)

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Device List: ",tf.config.experimental.list_physical_devices());

mnist = tf.keras.datasets.mnist
# x is the data, y is the category (train an test are a split)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("== MAX: ",tf.math.reduce_max(x_train))
print("== Mean: ",tf.math.reduce_mean(x_train))

# Convert the x data from 0-225 to 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0

print("== MAX: ",tf.math.reduce_max(x_train))
print("== Mean: ",tf.math.reduce_mean(x_train))

# Create a model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# Make predictions based on the model
predictions = model(x_train[:1]).numpy()
print(predictions)

# Convert them into probabilities
# Note: It is possible to bake the tf.nn.softmax function into the activation function 
# for the last layer of the network. While this can make the model output more directly 
# interpretable, this approach is discouraged as it's impossible to provide an exact 
# and numerically stable loss calculation for all models when using a softmax output.
sm = tf.nn.softmax(predictions).numpy()
print("Probabilities: ",sm)

# Define a loss function for training using losses.SparseCategoricalCrossentropy, which 
# takes a vector of logits and a True index and returns a scalar loss for each example.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# This loss is equal to the negative log probability of the true class: The loss is zero 
# if the model is sure of the correct class.
# This untrained model gives probabilities close to random (1/10 for each class), so the 
# initial loss should be close to -tf.math.log(1/10) ~= 2.3.
print("Loss: ",loss_fn(y_train[:1], predictions).numpy())

# Before you start training, configure and compile the model using Keras Model.compile. 
# Set the optimizer class to adam, set the loss to the loss_fn function you defined earlier, 
# and specify a metric to be evaluated for the model by setting the metrics parameter to accuracy.
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Use the Model.fit method to adjust your model parameters and minimize the loss:
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
print(model.evaluate(x_test,  y_test, verbose=20))