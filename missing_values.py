

# fills missing values with the value of another sample with the closest label
def fill_nan(x_train_, y_train_):
    x_train_['label'] = y_train_
    x_train_ = x_train_.sort_values(by=['label'])
    x_train_ = x_train_.fillna(method='ffill')
    x_train_ = x_train_.fillna(method='bfill')
    y_train_ = x_train_.iloc[:, -1]
    x_train_ = x_train_.drop('label', axis=1)
    x_train_ = x_train_.to_numpy()
    y_train_ = y_train_.to_numpy()

    y_train_ = y_train_.flatten()
    return x_train_, y_train_
