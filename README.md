# Modularized Functions for Data Science Code
## _Reusable, modularized functions for data science code_

**collect.py**
- gbq_to_df(query, project)

**prep.py**
- feature_target_split(df, target)
- cut_outliers(data, col_name)
- discretize(data, bins, list_to_discrete)
- get_dummies(X)
- split(features, target, test_size=0.2, random_state=0)
- scale(X_train, X_test, scale_type = "minmax")

**model.py**
- xgb_classifier(X_train, y_train, params)
- make_predictions(model, X_test)
- plot_feat_importance(model, feature_names, size=10)
- feat_imp_to_df(model, feature_names)

