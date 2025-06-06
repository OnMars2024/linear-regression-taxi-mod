#Deals with Dataset exploration
import dataset_exploration as de

im = de.ld

def build_model(my_learning_rate, num_features):
  """Create and compile a simple linear regression model."""
  # Describe the topography of the model.
  # The topography of a simple linear regression model
  # is a single node in a single layer.
  inputs = im.keras.Input(shape=(num_features,))
  outputs = im.keras.layers.Dense(units=1)(inputs)
  model = im.keras.Model(inputs=inputs, outputs=outputs)

  # Compile the model topography into code that Keras can efficiently
  # execute. Configure training to minimize the model's mean squared error.
  model.compile(optimizer=im.keras.optimizers.RMSprop(learning_rate=my_learning_rate),
                loss="mean_squared_error",
                metrics=[im.keras.metrics.RootMeanSquaredError()])

  return model


def train_model(model, features, label, epochs, batch_size):
  """Train the model by feeding it data."""

  # Feed the model the feature and the label.
  # The model will train for the specified number of epochs.
  history = model.fit(x=features,
                      y=label,
                      batch_size=batch_size,
                      epochs=epochs)

  # Gather the trained model's weight and bias.
  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]

  # The list of epochs is stored separately from the rest of history.
  epochs = history.epoch

  # Isolate the error for each epoch.
  hist = im.pd.DataFrame(history.history)

  # To track the progression of training, we're going to take a snapshot
  # of the model's root mean squared error at each epoch.
  rmse = hist["root_mean_squared_error"]

  return trained_weight, trained_bias, epochs, rmse


def run_experiment(df, feature_names, label_name, learning_rate, epochs, batch_size):

  print('INFO: starting training experiment with features={} and label={}\n'.format(feature_names, label_name))

  num_features = len(feature_names)

  features = df.loc[:, feature_names].values
  label = df[label_name].values

  model = build_model(learning_rate, num_features)
  model_output = train_model(model, features, label, epochs, batch_size)

  print('\nSUCCESS: training experiment complete\n')

  #Cleans feature and label names and rmse value
  safe_feature_str = "_".join(map(str, feature_names))
  safe_label_str = str(label_name).replace(" ", "_")
  rmse_value = round(float(model_output[3].iloc[-1]), 4)

  #Create directory for specific experiment
  dir_path = f'experiment_results/features={safe_feature_str}_label={safe_label_str}'
  de.os.makedirs(dir_path, exist_ok = True)

  #Prepare file name
  file_path = de.os.path.join(dir_path, f"rmse={rmse_value}.txt")
  
  #Write results to .txt file
  model_info_str = str(de.model_info(feature_names, label_name, model_output))
  #final_rmse_value = str(float(model_output[3]))

  results = (f"{model_info_str}"
             f"\n\nFinal RMSE Value: {rmse_value}"
             f"\nNumber of Epochs: {epochs}"
             f"\nLearning Rate: {learning_rate}"
             f"\nBatch Size: {batch_size}"

  )

  with open(file_path, "w") as f:
    f.write(results)
  
  f = open(file_path, "r")
  print(f.read())

  #Creates figure of model output data
  de.make_plots(df, feature_names, label_name, model_output)

  return model

print("SUCCESS: defining linear regression functions complete.")