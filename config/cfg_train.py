import os.path as osp

phase = 'train'

# data path
data_path = './data'

train_name = 'part0_train'

# save model path
save_path = './saves'

# global config
knn_method = 'faiss'
seed = 1

# density threshold
threshold_rho = 0.7
# Percentage of low / high density points

batch_size = 512
workers = 24
k = 80

epochs = 100
# print set up
print_iter = 50
print_epoch = 5

# cal faiss
cal_faiss = False
# view intermediate results
print_results = True

# model
model = dict(type='pairwise',
             kwargs=dict(feature_dim=1024, nclass=2, k=k))

# training args
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4)

STAGES = [int(epochs*0.4), int(epochs*0.6), int(epochs*0.8)]
lr_config = dict(
    policy='step',
    step=[int(r * epochs) for r in [0.4, 0.6, 0.8]]
)

data = dict(feat_path=osp.join(data_path, 'features', '{}.bin'.format(train_name)),
            label_path=osp.join(data_path, 'labels', '{}.meta'.format(train_name)),
            knn_path=osp.join(data_path, 'knns', train_name,
                            '{}_k_{}.npz'.format(knn_method, 80)),
            feature_dim=256,
            threshold_rho=threshold_rho,
            phase=phase,
            cf=cal_faiss,
            k=k)

# misc
metrics = ['pairwise', 'bcubed', 'nmi']
