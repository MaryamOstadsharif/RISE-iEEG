import os

import pandas as pd
from matplotlib import pyplot as plt
from pytorch_tabular.models.common.heads import LinearHeadConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig, TabNetModelConfig, FTTransformerConfig, GANDALFConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)

from pytorch_tabular.models import tabnet

from src.visualization import plot_and_save_results, plot_logistic_regression_weights, plot_xgboost_feature_importance, \
    plot_pca_classification


def train_logistic_regression(X_train, y_train, X_test, y_test, save_path, sampling_method, model_name, feature_names,
                              fold=None,
                              search_best=False):
    if search_best is True:
        hyper_param = {
            'penalty': ['l1', 'l2', 'none'],
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga'],
            'max_iter': [100, 200]
        }

        log_reg = LogisticRegression()

        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=log_reg, param_grid=hyper_param, cv=5, scoring='accuracy', n_jobs=-1)

        # Fit the grid search to the training data
        grid_search.fit(X_train, y_train)

        # Best estimator from grid search
        log_reg = grid_search.best_estimator_
        # C=10, max_iter=300, penalty='l1', solver='liblinear'
    else:
        log_reg = LogisticRegression(C=10, max_iter=200, penalty='l1', solver='liblinear')
        log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    plot_pca_classification(X_test, y_test, y_pred, save_path)

    print(f'Logistic Regression Accuracy: {accuracy}')
    plot_and_save_results(y_test, y_pred, 'Logistic_Regression', save_path)

    results = {
        'model': 'logistic_regression',
        'sampling_method': sampling_method,
        'fold': fold + 1,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_true=y_test, y_pred=y_pred, zero_division=1, average='weighted'),
        'recall': recall_score(y_true=y_test, y_pred=y_pred, zero_division=1, average='weighted'),
        'f1': f1_score(y_true=y_test, y_pred=y_pred, zero_division=1, average='weighted')
    }

    plot_logistic_regression_weights(log_reg, feature_names, save_path)

    return log_reg, results, y_pred


def train_xgboost(X_train, y_train, X_test, y_test, save_path, sampling_method, model_name, feature_names, fold=None,
                  search_best=False):
    X_train = pd.DataFrame(X_train, columns=feature_names)
    if search_best is True:
        hyper_param = {

        }

        xgb_clf = XGBClassifier()

        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=xgb_clf, param_grid=hyper_param, cv=5, scoring='accuracy', n_jobs=-1)

        # Fit the grid search to the training data
        grid_search.fit(X_train, y_train)

        # Best estimator from grid search
        xgb_clf = grid_search.best_estimator_
        # C=10, max_iter=300, penalty='l1', solver='liblinear'
    else:
        xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb_clf.fit(X_train, y_train)

    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_clf.fit(X_train, y_train)
    y_pred = xgb_clf.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, y_pred)
    print(f'XGBoost Classifier Accuracy: {accuracy_xgb}')
    plot_and_save_results(y_test, y_pred, 'XGBoost', save_path)
    plot_xgboost_feature_importance(xgb_clf, feature_names, save_path)

    results = {
        'model': model_name,
        'sampling_method': sampling_method,
        'fold': fold + 1,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_true=y_test, y_pred=y_pred, zero_division=1, average='weighted'),
        'recall': recall_score(y_true=y_test, y_pred=y_pred, zero_division=1, average='weighted'),
        'f1': f1_score(y_true=y_test, y_pred=y_pred, zero_division=1, average='weighted')
    }
    return xgb_clf, results, y_pred


def train_pytorch_tabular_based(data, target, save_path, num_col_names, cat_col_names,
                                sampling_method, model_name, feature_names, fold=None,
                                search_best=False, sampler=None):
    train, val, test = split_data(data, target, train_size=0.7, validation_size=0.1, test_size=0.2, random_state=42, sampler=sampler)

    data_config = DataConfig(
        target=[
            "target"
        ],
        # target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
        continuous_cols=num_col_names,
        categorical_cols=cat_col_names,
    )
    trainer_config = TrainerConfig(
        auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
        batch_size=1024,
        max_epochs=100,
    )
    optimizer_config = OptimizerConfig()
    head_config = LinearHeadConfig(
        layers="",  # No additional layer in head, just a mapping layer to output_dim
        dropout=0.1,
        initialization="kaiming",
    ).__dict__  # Convert to dict to pass to the model config (OmegaConf doesn't accept objects)


    if model_name == 'tabnet':
        model_config = TabNetModelConfig(
            task="classification",
            learning_rate=1e-5,
            n_d=16,
            n_a=16,
            n_steps=4,
            head="LinearHead",
            head_config=head_config)
    elif model_name == 'category_embedding':
        model_config = CategoryEmbeddingModelConfig(
            task="classification",
            layers="1024-512-512",  # Number of nodes in each layer
            activation="LeakyReLU",  # Activation between each layers
            learning_rate=1e-3)
    elif model_name == 'ft_transformer':
        model_config = FTTransformerConfig(
            task="classification",
            num_attn_blocks=3,
            num_heads=4,
            learning_rate=1e-3,
            head="LinearHead",
            head_config=head_config,)
    elif model_name == 'gandalf':
        model_config = GANDALFConfig(
            task="classification",
            gflu_stages=3,  # Number of stages in the GFLU block
            gflu_dropout=0.0,  # Dropout in each of the GFLU block
            gflu_feature_init_sparsity=0.1,  # Sparsity of the initial feature selection
            head="LinearHead",  # Linear Head
            head_config=head_config,  # Linear Head Config
            learning_rate=1e-3,
        )
    else:
        raise ValueError(" Model not defined")

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    history = tabular_model.fit(train=train, validation=val)
    result = tabular_model.evaluate(test)
    pred_df = tabular_model.predict(test)

    # Extract predictions and true labels
    y_pred = pred_df["prediction"].values
    y_test = test["target"].values

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

    # Print results
    print(f'PyTorch Tabular Model Accuracy: {accuracy}')
    plot_pca_classification(test[num_col_names + cat_col_names], y_test, y_pred, save_path)
    plot_and_save_results(y_test, y_pred, model_name, save_path)

    # Store results
    results = {
        'sampling_method': sampling_method,
        'model_name': model_name,
        'fold': 1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    # Identify misclassified samples
    misclassified_mask = y_pred != y_test
    misclassified_samples = test[misclassified_mask]
    misclassified_samples = pd.concat([misclassified_samples, pred_df[misclassified_mask]], axis=1)
    #misclassified_samples['predicted_target'] = y_pred[misclassified_mask]

    # Save misclassified samples to a CSV file
    misclassified_samples.to_csv(os.path.join(save_path, 'misclassified_samples.csv'), index=False)

    correct_samples_mask = y_pred == y_test
    correct_samples = test[correct_samples_mask]
    correct_samples = pd.concat([correct_samples, pred_df[correct_samples_mask]], axis=1)


    misclassified_samples_summary = misclassified_samples.describe().T
    correct_samples_summary = correct_samples.describe().T

    # Concatenate the summaries with suffixes to distinguish between train and test
    combined_summary = pd.concat([misclassified_samples_summary, correct_samples_summary], axis=1,
                                 keys=['Miss_classified', 'Correctly_classified'])
    # Flatten the MultiIndex columns and add proper suffixes
    combined_summary.columns = ['{}_{}'.format(col, src) for src, col in combined_summary.columns]

    # Save the combined summary to a CSV file
    combined_summary.to_csv(save_path + 'summary_statistics.csv')

    try:
        df = tabular_model.feature_importance().sort_values("importance", ascending=False)
        df.to_csv(save_path +'/feature_importance.csv')
    except:
        pass
    return tabular_model, results, y_pred


def split_data(data, target, sampler, train_size=0.7, validation_size=0.1, test_size=0.2, random_state=42):
    # First, split into training+validation and test sets
    data_train_val, data_test, target_train_val, target_test = train_test_split(
        data, target, test_size=test_size, random_state=random_state, stratify=target)

    # Calculate the proportion of validation data within the training+validation set
    validation_proportion = validation_size / (train_size + validation_size)

    # Split the training+validation set into training and validation sets
    data_train, data_val, target_train, target_val = train_test_split(
        data_train_val, target_train_val, test_size=validation_proportion, random_state=random_state,
        stratify=target_train_val)

    if sampler is None:
        data_train_resampled, target_train_resampled = data_train, target_train
        data_val_resampled, target_val_resampled = data_val, target_val
    else:
        data_train_resampled, target_train_resampled = sampler.fit_resample(data_train, target_train)
        data_val_resampled, target_val_resampled = sampler.fit_resample(data_val, target_val)

    data_train_resampled['target'], data_val_resampled['target'], data_test['target'] = (target_train_resampled.values,
                                                                                         target_val_resampled.values,
                                                                                         target_test.values)

    return data_train_resampled, data_val_resampled, data_test
