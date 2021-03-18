def build_config(dataset):
    cfg = type('', (), {})()
    if dataset == 'multithumos':
        cfg.annotations_file = '../datasets/Multi-THUMOS/multithumos.json'
        cfg.train_list = '../datasets/Multi-THUMOS/train_list.txt'
        cfg.test_list = '../datasets/Multi-THUMOS/test_list.txt'
        cfg.rgb_train_file = '../datasets/Multi-THUMOS/features/rgb_val.h5'
        cfg.rgb_test_file = '../datasets/Multi-THUMOS/features/rgb_test.h5'
        cfg.flow_train_file = '../datasets/Multi-THUMOS/features/flow_val.h5'
        cfg.flow_test_file = '../datasets/Multi-THUMOS/features/flow_test.h5'
        cfg.combined_train_file = '../datasets/Multi-THUMOS/features/combined_val.h5'
        cfg.combined_test_file = '../datasets/Multi-THUMOS/features/combined_test.h5'
        cfg.num_classes = 65
    elif dataset == 'charades':
        cfg.annotations_file = '../datasets/Charades/charades.json'
        cfg.train_list = '../datasets/Charades/train_list.txt'
        cfg.test_list = '../datasets/Charades/test_list.txt'
        cfg.rgb_train_file = '../datasets/Charades/features/I3D/rgb_train.h5'
        cfg.rgb_test_file = '../datasets/Charades/features/I3D/rgb_test.h5'
        cfg.flow_train_file = '../datasets/Charades/features/I3D/flow_train.h5'
        cfg.flow_test_file = '../datasets/Charades/features/I3D/flow_test.h5'
        cfg.combined_train_file = '../datasets/Charades/features/I3D/combined_train.h5'
        cfg.combined_test_file = '../datasets/Charades/features/I3D/combined_test.h5'
        cfg.num_classes = 157
    cfg.saved_models_dir = './results/saved_models'
    cfg.tf_logs_dir = './results/logs'
    return cfg
