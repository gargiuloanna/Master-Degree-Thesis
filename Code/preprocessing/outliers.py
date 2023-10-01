from sklearn.neighbors import LocalOutlierFactor

def remove_outliers(data, labels):
    local = LocalOutlierFactor(n_neighbors=15, n_jobs=-1)
    outliers = local.fit_predict(data)
    print("[INFO] Removed ", data[outliers==-1].shape[0], " outliers")
    return data[outliers == 1], labels[outliers == 1]