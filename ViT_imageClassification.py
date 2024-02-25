import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


num_classes = 10
inputa_shape = (32,32,3)
(x_train,y_train),(x_test,y_test) = keras.datasets.cifar10.load_data()

print(f"x_train : {x_train.shape} - y_train : {y_train.shape}")
print(f"x_test : {x_test.shape} - y_test : {y_test.shape}")


x_train = x_train[:500]
y_train = y_train[:500]
x_test = x_test[:500]
y_test =y_test[:500]


# Hyper Parameter definition
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 30
image_size = 72
patch_size = 6
num_patches = (image_size // patch_size) ** 2
projection_dim  = 64
num_heads= 4
transform_units = [projection_dim*2,projection_dim]

transformer_layer = 8
mlp_head_units = [2048,1024]


# Built Vit Classifier Model
 # data augmentation
data_augmentation = keras.Sequential(
    [
      layers.Normalization(),
    layers.Resizing(image_size,image_size),
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(factor=0.02),
    layers.RandomZoom(height_factor=0.2, width_factor=0.2)
    ],
    name="data_augmentation"

)
data_augmentation.layers[0].adapt(x_train)


#MLP architecture
def mlp(x, hidden_units, dropout_rate):
  for units in hidden_units:
    x = layers.Dense(units,activation=tf.nn.gelu)(x)
    x = layers.Dropout(dropout_rate)(x)
  return x


#Patches
class Patches(layers.Layer):
  def __init__(self, patch_size):
    super(Patches,self).__init__()
    self.patch_size=patch_size

    def call(self,image):
      batch_size = tf.shape(images)[0]
      patches = tf.image.extract_patches(
          images=images,
          size=[1,self.patch_size,self.patch_size,1],
          strides=[1,self.patch_size,self.patch_size,1],
          rates = [1,1,1,1],
          padding = "VALID"
      )
      patch_dim = patches.shape[-1]
      patches = tf.reshape(patches,[batch_size,-1,patch_dim])
      return patches



import matplotlib.pyplot as plt

plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")

resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")



class PatchEncoder(layers.Layer):
  def __init__(self,num_patches,projection_dim):
    super(PatchEncoder,self).__init__()
    self.num_patches = num_patches
    self.projection = layers.Dense(units = projection_dim)
    self.position_embedding = layers.Embedding(
          input_dim = num_patches, output_dim = projection_dim
       )
  def call(self,patch):
    positions = tf.range(start=0, limit = self.num_patches, delta=1)
    encoded = self.projection(patch) + self.position_embedding(positions)
    return encoded



def create_vit_classifier():
  inputs = layers.Input(shape=input_shape)
  #Augment data
  augmented = data_augmentation(inputs)
  #create patches
  patches = Patches(patch_size)(augmented)
  #Encode patches
  encoded_patches = PatchEncoder(num_patches,projection_dim)(patches)

  #crate multiple layers of the transformer black
  for _ in range(transformer_layers):
    #Layer normalization
    x1 = layers.LayerNormalization(espilon=1e-6)(encoded_patches)
    #create a multi-head attention layer
    attension_output = layers.MultiHeadAttention(
        num_heads = num_heads,key_dim=projection_dim,dropout=0.1
    )(x1,x1)
    #skip connection
    x2 = layers.Add()([attension_output,encoded_patches])
    # layer normalization
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    #MLP
    x4 = mlp(x3,hidden_units=transformer_units,dropout_rate=0.1)
    #skip connection2
    encoder_patches = layers.Add()([x4,x2])



  #create a [batch_size,projection_dim] tensor
  representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
  representation = layers.Flatten()(representation)
  representation = layers.Dropout(0.5)(representation)



  # Add MLP
  features = mlp(representation,hidden_units = mlp_head_units,dropout_rate = 0.5)
  #classify outputs
  logits = layers.Dense(num_classes)(features)
  #create the keras model
  model = keras.Model(inputs= inputs, outputs = logits)
  return model



def run_experiment(model):
  optimizer = tfa.optimizers.AdamW(
      learning_rate = learning_rate, weight_decay = weight_decay
  )

  model.compile( optimizer = optimizer,loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy"),keras.metrics.SparseCategoricalAccuracy(5, name="top-5-accuray"),],)
  checkpoint_filepath = "./tmp/checkpoint"
  checkpoint_callback = keras.callbacks.ModelCheckpoint(
      checkpoint_filepath,
      mointer = "val_accuracy",
      save_best_only = True,
      save_weights_only = True
  )

  history = model.fit(
      x=x_train,
      y=y_train,
      batch_size = batch_size,
      epochs=num_epochs,
      validation_split = 0.1,
      callbacks = [checkpoint_callback],
  )


  model.load_weights(checkpoint_filepath)
  _, accuracy, top_5_accuracy = model.evaluate(x_test,y_test)
  print(f"Test accuracy : {round(accuracy * 100,2)}%")
  print(f"Top 5 accuracy : {round(top_5_accuracy * 100,2)}%")


vit_classifier = create_vit_classifier
history = run_experiment(vit_classifier)


class_names=[
    'airoplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]


def img_predict(images, model):

  if len(images.shape) == 3:
    out = model.predict(images.reshape(-1,*images.shape))
  else:
    out = model.predict(images)

  prediction = np.argmax(out,axis=1)
  img_prediction = [class_names[i] for i in range(prediction)]

  return img_prediction


index = 16
plt.imshow(x_test[index])
prediction = img_predict(x_text[index],vit_classifier)
print(prediction)
