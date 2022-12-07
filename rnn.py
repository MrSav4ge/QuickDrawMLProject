import numpy as np
import tensorflow as tf
import Dataprep as d

def getData(infile):
  input = np.reshape(np.array(d.loadfile(infile)), [-1,])
  return input

def createRNN(size):
    model = tf.keras.Sequential([
      tf.keras.layers.Embedding(
          input_dim=size,
          output_dim=64,
          # Use masking to handle the variable sequence lengths
          mask_zero=True),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
      tf.keras.layers.Dense(64, activation='softmax'),
      tf.keras.layers.Dense(1)
    ])

    in_dims = 1
    out_dims = 1

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
    model.summary()
    return model

def trainRNN():
    train = tf.random.normal(getData("/trainValues.json"))  #or whatever type of file
    test = tf.random.normal(getData("/testValues.json"))  #or whatever type of file
    model = createRNN(len(train))

    history = model.fit(train, epochs=10,
                    validation_data=test,
                    validation_steps=30)

    loss, acc = model.evaluate(test)

    print(loss)
    print(acc)

    return model

def getPredictions(model):
    data = getData("/drawData.json")
    result = model.predict(np.array(data))