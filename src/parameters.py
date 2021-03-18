num_classes = 65

num_epochs = 50
batch_size = 16
steps = 4

num_clips = 128

feature_dim = 2048
hidden_dim = 128
num_heads = 0
num_layers = 5

learning_rate = 1e-3
weight_decay = 1e-6
optimizer = 'ADAM'

f1_threshold = 0.5

swa_start = -1
swa_update_interval = 0

num_workers = 0

weights = {'init_output': 1.0, 'final_output': 1.0}
