#This experiment analyses the relationship between TRIP_MILES and FARE
import training as t

im = t.de

#The following variables are hyperparameters
learning_rate = 0.0005
epochs = 20
batch_size = 50

features = ['TRIP_SECONDS']
label = 'FARE'

model_1 = t.run_experiment(im.training_df, features, label, learning_rate, epochs, batch_size)