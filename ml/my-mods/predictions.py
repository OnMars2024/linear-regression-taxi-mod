import lib_dep as ld
import dataset_exploration as de

def format_currency(x):
  return "${:.2f}".format(x)

def build_batch(df, batch_size):
  batch = df.sample(n=batch_size).copy()
  batch.set_index(ld.np.arange(batch_size), inplace=True)
  return batch

def predict_fare(model, df, features, label, batch_size=50):
  batch = build_batch(df, batch_size)
  predicted_values = model.predict_on_batch(x=batch.loc[:, features].values)

  data = {"PREDICTED_FARE": [], "OBSERVED_FARE": [], "L1_LOSS": [],
          features[0]: [], features[1]: []}
  for i in range(batch_size):
    predicted = predicted_values[i][0]
    observed = batch.at[i, label]
    data["PREDICTED_FARE"].append(format_currency(predicted))
    data["OBSERVED_FARE"].append(format_currency(observed))
    data["L1_LOSS"].append(format_currency(abs(observed - predicted)))
    data[features[0]].append(batch.at[i, features[0]])
    data[features[1]].append("{:.2f}".format(batch.at[i, features[1]]))

  output_df = ld.pd.DataFrame(data)

  #Cleans feature and label names
  safe_feature_str = "_".join(map(str, features))
  safe_label_str = str(label).replace(" ", "_")
  safe_l1_loss = f"L1_Mean_Loss: {ld.np.mean([float(val.replace("$","")) for val in data["L1_LOSS"]]):.4f}"

  #Create directory for specific prediction
  dir_path = f'prediction_results/features={safe_feature_str}_label={safe_label_str}'
  de.os.makedirs(dir_path, exist_ok = True)

  #Creates file path
  file_path = dir_path + safe_l1_loss
  output_df.to_csv(file_path)
  return output_df

def show_predictions(output):
  header = "-" * 80
  banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
  print(banner)
  print(output)
  return
