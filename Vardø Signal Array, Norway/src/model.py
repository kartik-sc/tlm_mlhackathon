import lightgbm as lgb

def train_lgb(X_tr, y_tr, w_tr, X_val, y_val, w_val, params):
  train_data = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
  val_data = lgb.Dataset(X_val, label=y_val, weight=w_val)

  model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    num_boost_round=2000,
    callbacks=[lgb.early_stopping(100)]
  )
  return model