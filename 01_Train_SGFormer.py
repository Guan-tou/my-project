from tqdm import tqdm
from fc_funcs_GNN1 import *
from SGFormer_model import *
from scipy.io import loadmat
import time
import random
import matplotlib.pyplot as plt
import pandas as pd

random.seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(42)
np.random.seed(42)

start_time = time.time()


folder_path = r''
data_list = []

for file_name in sorted(os.listdir(folder_path)):
    if file_name.endswith('.mat') and file_name != '1.mat':

        file_path = os.path.join(folder_path, file_name)
        mat_contents = loadmat(file_path)
        matrix_var_name = 'ROICorrelation'

        if matrix_var_name in mat_contents:
            data_matrix = mat_contents[matrix_var_name]
            data_list.append(data_matrix)

data = np.stack(data_list, axis=0)

df = pd.read_csv('')

train_indices = df['ABIDE'].values == 2
test_indices = df['ABIDE'].values == 1

data_test = data[test_indices]
data_train = data[train_indices]

df_test = df.iloc[test_indices].copy()
df_train = df.iloc[train_indices].copy()
patients = df['DX_GROUP'].values == 1
controls = df['DX_GROUP'].values == 2
TD_subs = data[controls, :]
DX_subs = data[patients, :]

with tf.device('/GPU:0'):
    batch_size = 64
    in_channels = data.shape[1]
    hidden_channels = 64
    out_channels = 16

    cvae, z_encoder, s_encoder, cvae_decoder = get_FC_CAE(batch_size=batch_size,
                                                          in_channels=in_channels,
                                                          hidden_channels=hidden_channels,
                                                          out_channels=out_channels)

    z_encoder.summary(line_length=200)
    s_encoder.summary(line_length=200)
    cvae_decoder.summary(line_length=100)

    loss_history = {
        'total_loss': [],
        'dx_mse_loss': [],
        'td_mse_loss': [],
        'kl_tg_z_loss': [],
        'kl_tg_s_loss': [],
        'kl_bg_z_loss': [],
        'kl_total_loss': [],
        'tc_loss': [],
        'discriminator_loss': []
    }

    n = 2001
    for i in tqdm(range(1, int(n))):

        DX_batch_ind = np.random.choice(DX_subs.shape[0], size=batch_size, replace=False)
        DX_batch = DX_subs[DX_batch_ind, :]

        TD_batch_ind = np.random.choice(TD_subs.shape[0], size=batch_size, replace=False)
        TD_batch = TD_subs[TD_batch_ind, :]

        metrics = cvae.train_on_batch(
            [DX_batch, DX_batch, TD_batch, TD_batch]
        )

        total_loss = metrics[0]
        dx_mse = metrics[1]
        td_mse = metrics[2]
        kl_tg_z = metrics[3]
        kl_tg_s = metrics[4]
        kl_bg_z = metrics[5]
        kl_total = metrics[6]
        tc = metrics[7]
        disc = metrics[8]

        loss_history['total_loss'].append(total_loss)
        loss_history['dx_mse_loss'].append(dx_mse)
        loss_history['td_mse_loss'].append(td_mse)
        loss_history['kl_tg_z_loss'].append(kl_tg_z)
        loss_history['kl_tg_s_loss'].append(kl_tg_s)
        loss_history['kl_bg_z_loss'].append(kl_bg_z)
        loss_history['kl_total_loss'].append(kl_total)
        loss_history['tc_loss'].append(tc)
        loss_history['discriminator_loss'].append(disc)

        assert not np.isnan(total_loss), f'总损失包含 NaN - 迭代: {i}'
        assert not np.isnan(dx_mse), f'DX MSE损失包含 NaN - 迭代: {i}'
        assert not np.isnan(td_mse), f'TD MSE损失包含 NaN - 迭代: {i}'

        weight_name = '../ALL/weight_outputs/cvae/' + 'cvae_weights_' + str(i) + '.h5'
        cvae.save_weights(weight_name)

    plot_all_losses(loss_history, n - 1)

end_time = time.time()
elapsed_time = end_time - start_time
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Runtime：{int(hours)}:{int(minutes)}:{int(seconds)}.")