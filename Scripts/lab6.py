import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression,\
                    Ridge, Lasso

from sklearn.metrics import mean_squared_error,\
                    mean_absolute_error


#exercise 1
def normalize_data (train_data, test_data):

    scaler = preprocessing.StandardScaler()

    scaler.fit(train_data)
    scaled_train    = scaler.transform(train_data)
    scaled_test     = scaler.transform(test_data)

    return (scaled_train, scaled_test)


def mse_and_mae (model,
                 train_data, train_labels,
                 test_data, test_labels):

    norm_train, norm_test = normalize_data(train_data, test_data)

    fitter      = model.fit(norm_train, train_labels)
    predictions = fitter.predict(norm_test)

    mse         = mean_squared_error (test_labels, predictions)
    mae         = mean_absolute_error(test_labels, predictions)

    return mse, mae



#load data
data   = np.load("../data/training_data.npy")
prices = np.load("../data/prices.npy")

# shuffle
data, prices = shuffle(data, prices, random_state=0)

# number of samples per fold
num_samples_fold = len(data) // 3

# split data in 3 folds
data_1       = data   [:num_samples_fold]
prices_1     = prices [:num_samples_fold]
data_2       = data   [ num_samples_fold : 2 * num_samples_fold]
prices_2     = prices [ num_samples_fold : 2 * num_samples_fold]
data_3       = data   [2 * num_samples_fold:]
prices_3     = prices [2 * num_samples_fold:]



#exercise 2
model = LinearRegression()

mse_12, mae_12 = mse_and_mae(model,
                             np.concatenate((data_1, data_2)),
                             np.concatenate((prices_1, prices_2)),
                             data_3, prices_3)

mse_13, mae_13 = mse_and_mae(model,
                             np.concatenate((data_1, data_3)),
                             np.concatenate((prices_1, prices_3)),
                             data_2, prices_2)

mse_23, mae_23 = mse_and_mae(model,
                             np.concatenate((data_2, data_3)),
                             np.concatenate((prices_2, prices_3)),
                             data_1, prices_1)

mean_mse = (mse_12 + mse_13 + mse_23) / 3
mean_mae = (mae_12 + mae_13 + mae_23) / 3
print(mean_mse, mean_mae)



#exercise 3
best_mae    = 0
best_mse    = -1    #squared error cant pe negative, initialisation use
best_alpha  = 0

for alpha in [1, 10, 100, 1000]:
    model = Ridge(alpha = alpha)

    mse_12, mae_12 = mse_and_mae(model,
                                 np.concatenate((data_1, data_2)),
                                 np.concatenate((prices_1, prices_2)),
                                 data_3, prices_3)

    mse_13, mae_13 = mse_and_mae(model,
                                 np.concatenate((data_1, data_3)),
                                 np.concatenate((prices_1, prices_3)),
                                 data_2, prices_2)

    mse_23, mae_23 = mse_and_mae(model,
                                 np.concatenate((data_2, data_3)),
                                 np.concatenate((prices_2, prices_3)),
                                 data_1, prices_1)

    mean_mse = (mse_12 + mse_13 + mse_23) / 3
    mean_mae = (mae_12 + mae_13 + mae_23) / 3

    #best_mse will be -1 just if it has never been used, squared is always > 0
    if  best_mse == -1 \
        or (mean_mae + mean_mse) / 2 < (best_mae + best_mse) / 2:
        best_mae = mean_mae
        best_mse = mean_mse
        best_alpha = alpha

print("Cel mai bun alpha si mse/mae aferente:", best_alpha, best_mse,
                                                best_mae)


#exercise 4
model       = Ridge(best_alpha)
scaler      = preprocessing.StandardScaler()
scaler.fit(data)
norm_data   = scaler.transform(data)
model.fit(norm_data, prices)

attributes = ["Year", "Kilometers Driven", "Mileage",
            "Engine", "Power", "Seats", "Owner Type",
            "Fuel Type", "Transmission"]

print("Coeficientii sunt: ",     model.coef_)
print("Biasul regresiei este: ", model.intercept_)

print("Cel mai semnificativ atribut este: ",
      attributes[np.argmax(np.abs(model.coef_))])

print("Al doilea cel mai semnificativ atribut este: ",
        attributes[np.argmax(np.abs(model.coef_)) + 1])

print("Cel mai nesemnificativ atribut este:",
      attributes[np.argmin(np.abs(model.coef_))])
