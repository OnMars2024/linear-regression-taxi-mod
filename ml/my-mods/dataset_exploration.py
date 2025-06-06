#Loads required libraries and dependencies
import lib_dep as ld

#Use os to create directories for specific experimental results
import os

#Loads dataset
chicago_taxi_ds = ld.pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

#Updates dataframe to use specific columns
training_df = chicago_taxi_ds[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]

#Clears terminal text from importing libraries and dependencies
ld.subprocess.run("clear", shell=True)

print("Read dataset completed successfully.")
print("Total number of rows: {0}\n\n".format(len(training_df.index)))

#print(training_df.head(25))

#Dataset Statistics
stat = training_df.describe(include="all")

#Correlation Matrix
corr_mat = training_df.corr(numeric_only = True)

#Visualize relationships with pairplot
pair_plot = ld.sns.pairplot(training_df, x_vars = ['FARE', 'TRIP_MILES', 'TRIP_SECONDS'], y_vars = ['FARE', 
'TRIP_MILES', 'TRIP_SECONDS'])

training_df.to_csv("figures/training_data.csv", index = False)
stat.to_csv("figures/dataset_statistics.csv", index = True)
corr_mat.to_csv("figures/correlation_matrix.csv", index = True)
pair_plot.savefig("figures/pair_plot.png")

#----------------------------------------------------------------------------------

#Defining plotting functions
def make_plots(df, feature_names, label_name, model_output, batch_size=200):

    random_sample = df.sample(n = batch_size).copy()
    random_sample.reset_index()
    weights, bias, epochs, rmse = model_output

    if(len(feature_names) == 1):
        is_2d_plot = True

    if(is_2d_plot):
        model_plot_type = 'scatter'
    else:
        model_plot_type = 'surface'

    fig = ld.make_subplots(rows = 1, cols = 2,
                        subplot_titles = ("Loss Curve", "Model Plot"),
                        specs = [[{"type" : "scatter"}, {"type" : model_plot_type}]])

    plot_data(random_sample, feature_names,label_name, fig)
    plot_model(random_sample, feature_names, weights, bias, fig)
    plot_loss_curve(epochs, rmse, fig)

    #Cleans feature and label names and rmse value
    safe_feature_str = "_".join(map(str, feature_names))
    safe_label_str = str(label_name).replace(" ", "_")
    rmse_value = round(float(model_output[3].iloc[-1]), 4)

    #Creates file path
    dir_path = f'experiment_results/features={safe_feature_str}_label={safe_label_str}'

    #Prepare file name
    file_path = os.path.join(dir_path, f"rmse={rmse_value}.png")

    fig.write_image(file_path)
    
    return

def plot_loss_curve(epochs, rmse, fig):
    curve = ld.px.line(x = epochs, y = rmse)
    curve.update_traces(line_color = "#ff0000", line_width = 3)

    fig.append_trace(curve.data[0],row = 1, col = 1)
    fig.update_xaxes(title_text = "Epoch", row = 1, col = 1)
    fig.update_yaxes(title_text="Root Mean Squared Error", row = 1, col = 1, range = [rmse.min()*0.8, rmse.max()])
    
    return

def plot_data(df, features, label, fig):
    
    if(len(features) == 1):
        scatter = ld.px.scatter(df, x = features[0], y = label)
    else:
        scatter = ld.px.scatter_3d(df, x = features[0], y = features[1], z = label)
    
    fig.append_trace(scatter.data[0], row = 1, col = 2)

    if(len(features) == 1):
        
        fig.update_xaxes(title_text = features[0], row = 1, col = 2)
        fig.update_yaxes(title_text = label, row = 1, col = 2)

    else:

        fig.update_layout(scene1 = dict(xaxis_title = feature[0], yaxis_title = feature[1], zaxis_title = label))
   
    return

def plot_model(df, features, weights, bias, fig):
    df['FARE_PREDICTED'] = bias[0]

    for index, feature in enumerate(features):
        df['FARE_PREDICTED'] = df['FARE_PREDICTED'] + weights[index][0] * df[feature]

    if(len(features) == 1):

        model = ld.px.line(df, x = features[0], y = 'FARE_PREDICTED')
        model.update_traces(line_color = '#ff0000', line_width = 3)

    else:

        z_name, y_name = 'FARE_PREDICTED', features[1]

        z = [df[z_name].min(), (df[z_name].max() - df[z_name].min()) / 2, df[z_name].max()]
        y = [df[y_name].min(), (df[y_name].max() - df[y_name].min()) / 2, df[y_name].max()]
        x = []

        for i in range(len(y)):
            x.append((z[i] - weights[1][0] * y[i] - bias[0]) / weights[0][0])

        plane = ld.pd.DataFrame({'x':x, 'y':y, 'z':z * 3})

        light_yellow = [[0, '#89CFF0'], [1, '#FFDB58']]

        model = ld.go.Figure(data = ld.go.Surface(x = plane['x'], y = plane['y'], 
                                                  z = plane['z'], 
                                                  colorscale = light_yellow))
        
    fig.add_trace(model.data[0], row = 1, col = 2)

    return

def model_info(feature_names, label_name, model_output):

    weights = model_output[0]
    bias = model_output[1]

    n1 = "\n"
    header = "-" * 80
    banner = header + n1 + '|' + 'MODEL INFO'.center(78) + '|' + n1 + header

    info = ''
    equation = label_name + ' = '

    for index, feature in enumerate(feature_names):
        info = info + "Weight for feature[{}]: {:.3f}\n".format(feature, weights[index][0])
        equation = equation + '{:.3f} * {} + '.format(weights[index][0], feature)

    info = info + 'Bias: {:.3f}\n'.format(bias[0])
    equation = equation + '{:.3f}\n'.format(bias[0])

    return banner + n1 + info + n1 + equation

print("SUCCESS: defining plotting functions complete.")


    
