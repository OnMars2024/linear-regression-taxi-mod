#This experiment analyses the relationship between TRIP_MILES and FARE
import training as t
import predictions as p

im = t.de

#The following variables are hyperparameters
learning_rate = 0.001
epochs = 20
batch_size = 50

features = ['TRIP_MILES']
label = 'FARE'

#Trains the model(s) based on learing_rate, epochs, batch_size, features, and label
#model_1 = t.run_experiment(im.training_df, features, label, learning_rate, epochs, batch_size)
model_2 = t.run_experiment(im.training_df, ['TRIP_MILES', 'TRIP_SECONDS'], label, learning_rate, epochs, batch_size)


#Tests the model (The test set is the same as training set, which isn't ideal)
output = p.predict_fare(model_2, im.training_df, features, label)
p.show_predictions(output)
