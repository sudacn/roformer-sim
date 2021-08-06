class Config:
    # 基本信息
    is_train = True

    maxlen = 20
    batch_size = 50

    steps_per_epoch = 150
    epochs = 10
    batch_size = 50
    # bert配置
    config_path = '/root/ft_local/bert/chinese_roformer-sim-char_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = '/root/ft_local/bert/chinese_roformer-sim-char_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = '/root/ft_local/bert/chinese_roformer-sim-char_L-12_H-768_A-12/vocab.txt'
    csv_path = '/root/ft_local/0712-all-hospital.csv'
    train_path = '/root/ft_local/model_data/hospital_train.csv'
    dev_path = '/root/ft_local/model_data/hospital_dev.csv'
    test_path = '/root/ft_local/model_data/hospital_test.csv'
    last_model_path = '/root/gitWOA/roformer-sim/data/beijing_latest_model.weights'
    best_model_path = '/root/gitWOA/roformer-sim/data/hospital_best_model.weights'
