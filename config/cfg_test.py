import os.path as osp

phase = 'test'

# data path
data_path = './data'

test_name = 'part1_test'

# save model path
save_path = './saves/checkpoint.pth'

# global config
knn_method = 'faiss'
seed = 1

# density threshold
threshold_rho = 0.7
# Percentage of low / high density points

batch_size = 512
workers = 20
k = 80

# cal faiss
cal_faiss = True

# model
model = dict(type='pairwise',
             kwargs=dict(feature_dim=1024, nclass=2, k=k))

model_c = dict(type='center',
             kwargs=dict(feature_dim=1024, nclass=2, k=k))

data = dict(feat_path=osp.join(data_path, 'features', '{}.bin'.format(test_name)),
            label_path=osp.join(data_path, 'labels', '{}.meta'.format(test_name)),
            knn_path=osp.join(data_path, 'knns', test_name,
                            '{}_k_{}.npz'.format(knn_method, 80)),
            feature_dim=256,
            threshold_rho=threshold_rho,
            phase=phase,
            cf=cal_faiss,
            k=k)

# misc
metrics = ['pairwise', 'bcubed', 'nmi']
