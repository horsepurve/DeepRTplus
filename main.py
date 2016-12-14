from DeepRT import *

#%% ==================== split data ====================
split_dataset(dataset = 'mod.txt', 
           train_ratio = 8, 
           validate_ratio = 1, 
           test_ratio = 1, 
           seed = 42)

#%% ==================== CNN training ====================
cnn_train(learning_rate = 0.01,
          k_size = 3,
          drop_value = 0.2,
          Layers = "20,20,20,20",
          batch_size = 128,
          Flag = 1,
          log = 'log/cnn_3_20_20_20_20_0.2.txt')
          
cnn_train(learning_rate = 0.01,
          k_size = 4,
          drop_value = 0.2,
          Layers = "20,20,20,20",
          batch_size = 128,
          Flag = 0,
          log = 'log/cnn_4_20_20_20_20_0.2.txt')

cnn_train(learning_rate = 0.001,
          k_size = 5,
          drop_value = 0.5,
          Layers = "20,20,20,20",
          batch_size = 128,
          Flag = 0,
          log = 'log/cnn_5_20_20_20_20_0.5.txt')

cnn_train(learning_rate = 0.001,
          k_size = 6,
          drop_value = 0.2,
          Layers = "20,20,20,20",
          batch_size = 128,
          Flag = 0,
          log = 'log/cnn_6_20_20_20_20_0.2.txt')

cnn_train(learning_rate = 0.001,
          k_size = 7,
          drop_value = 0.2,
          Layers = "20,20,20,20",
          batch_size = 128,
          Flag = 0,
          log = 'log/cnn_7_20_20_20_20_0.2.txt')

cnn_train(learning_rate = 0.001,
          k_size = 8,
          drop_value = 0.5,
          Layers = "20,20,20,20",
          batch_size = 128,
          Flag = 0,
          log = 'log/cnn_8_20_20_20_20_0.5.txt')

cnn_train(learning_rate = 0.01,
          k_size = 9,
          drop_value = 0.2,
          Layers = "20,20,20,20",
          batch_size = 128,
          Flag = 0,
          log = 'log/cnn_9_20_20_20_20_0.2.txt')

#%% ==================== CNN feature extraction ====================
CNN_feature_extractor()

#%% ==================== LSTM training ====================
LSTM_training(batch_size = 128,
              layer_num = 1,
              active = 'linear')

LSTM_training(batch_size = 128,
              layer_num = 2,
              active = 'linear')

LSTM_training(batch_size = 128,
              layer_num = 1,
              active = 'sigmoid')

LSTM_training(batch_size = 128,
              layer_num = 2,
              active = 'sigmoid')

#%% ==================== LSTM feature extraction ====================
LSTM_feature_extractor()

#%% ==================== PCA ====================
PCA(0.95)

#%% ==================== Support vector machine regression ====================
SVR_regression(gamma = 1e-9, C = 1e6)

#%% ==================== Random forest regression ====================
RF_regression(n_estimators = 400)

#%% ==================== Gradient boosting regression ====================
GB_regression(n_estimators = 1000)

#%% ==================== Bagging ====================
bagging()

#%% ==================== Evaluation ====================
Evaluation()



