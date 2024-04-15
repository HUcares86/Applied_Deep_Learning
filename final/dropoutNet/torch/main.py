import numpy as np
import pandas as pd
import torch
import datetime
from tqdm import tqdm
import argparse
import os
from metric import mapk
import utils
import model
import random

mode = 'val' 
task = 'seen'
# mode = 'test'

def main():
    data_path           = args.data_dir
    checkpoint_path     = args.checkpoint_path
    model_select        = args.model_select

    rank_out            = args.rank
    user_batch_size     = 5000
    n_scores_user       = 50
    data_batch_size     = 5000
    dropout             = args.dropout
    eval_batch_size     = 5000
    max_data_per_step   = 2500000
    eval_every          = args.eval_every
    num_epoch           = 1

    _lr = args.lr
    _decay_lr_every = 50
    _lr_decay = 0.1

    experiment = '%s_%s' % (
        datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),
        '-'.join(str(x / 100) for x in model_select) if model_select else 'simple'
    )

    print('running: ' + experiment)
    dat = load_data(data_path)
    u_pref_scaled = dat['u_pref_scaled']
    v_pref_scaled = dat['v_pref_scaled']
    eval_cold_user = dat['eval_cold_user']
    eval_cold_item = dat['eval_cold_item']
    user_content = dat['user_content'].astype(np.float32)
    item_content = dat['item_content'].astype(np.float32)
    u_pref = dat['u_pref']
    v_pref = dat['v_pref']
    user_indices = dat['user_indices']
    # append pref factors for faster dropout
    v_pref_expanded = np.vstack([v_pref_scaled, np.zeros_like(v_pref_scaled[0, :])])
    v_pref_last = v_pref_scaled.shape[0]
    u_pref_expanded = np.vstack([u_pref_scaled, np.zeros_like(u_pref_scaled[0, :])])
    u_pref_last = u_pref_scaled.shape[0]
    u_mean =  np.mean(u_pref_expanded, 0)
    v_mean =  np.mean(v_pref_expanded, 0)
    new_Vin = torch.tensor(v_mean).repeat( len(eval_cold_item),1)
    new_user = eval_cold_user.drop(['user_id'], axis=1).values
    train_data = dat['train_data']
    # for i in range(len(eval_cold_item)):
    #     _v = np.array([0]*v_pref_expanded.shape[1])
    #     cnt = 0
    #     for j in item_content:
            
    #         # temp = difflib.SequenceMatcher(None,j ,eval_cold_item.iloc[i].values[1:])
    #         # if temp.ratio() > 0.8:
    #         if j[0] == 1 and eval_cold_item['price'][i] == 1:
    #             _v = np.add(_v, v_pref_expanded[i])
    #             cnt += 1
    #     if cnt != 0:
    #         _v /= cnt
    #         new_Vin[i] = torch.from_numpy(_v)
       
    eval_batch_size = eval_batch_size
    dropout_net = model.get_model(latent_rank_in=u_pref.shape[1],
                               user_content_rank=user_content.shape[1],
                               item_content_rank=item_content.shape[1],
                               model_select=model_select,
                               rank_out=rank_out)

    row_index = np.copy(user_indices)
    n_step = 0
    n_batch_trained = 0
    accuracy_best = 0
    optimizer = torch.optim.SGD(dropout_net.parameters(), _lr, momentum=0.9)
    crit = torch.nn.MSELoss()
    d_train = torch.device(args.model_device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=_decay_lr_every, gamma=_lr_decay)
    dropout_net.to(d_train)
    dropout_net.train()

    for epoch in range(num_epoch):
        print(epoch)
        np.random.shuffle(row_index)
        for iter, b in enumerate(utils.batch(row_index, user_batch_size)):
            n_step += 1
            # prep targets
            target_users = np.repeat(b, n_scores_user)
            target_users_rand = np.repeat(np.arange(len(b)), n_scores_user)
            target_items_rand = [np.random.choice(v_pref.shape[0], n_scores_user) for _ in b]
            target_items_rand = np.array(target_items_rand).flatten()
            target_ui_rand = np.transpose(np.vstack([target_users_rand, target_items_rand]))
            preds_pref = np.matmul(u_pref[b, :], v_pref.T)
            # print((preds_pref))
            for user in range(len(b)):
                tmp = train_data.loc[[b[user]]].values[0][1:]
                # preds_pref[user] = preds_pref[user] / max(preds_pref[user])
                for ID in range(len(tmp)):
                    if tmp[ID] == 1:
                        preds_pref[user][ID] = 10
            preds_pref = torch.tensor(preds_pref)
            target_scores, target_items = torch.topk(preds_pref, k=n_scores_user, sorted=True)
            random_scores = preds_pref.detach().cpu().numpy()[target_ui_rand[:,0],target_ui_rand[:,1]]

            # merge topN and randomN items per user
            target_scores = np.append(target_scores, random_scores)
            target_items = np.append(target_items, target_items_rand)
            target_users = np.append(target_users, target_users)

            n_targets = len(target_scores)
            perm = np.random.permutation(n_targets)
            n_targets = min(n_targets, max_data_per_step)
            data_batch = [(n, min(n + data_batch_size, n_targets)) for n in range(0, n_targets, data_batch_size)]
            f_batch = 0
            pbar = tqdm(data_batch, desc='ubatch')
            
            for (start, stop) in pbar:
                batch_perm = perm[start:stop]
                batch_users = target_users[batch_perm]
                batch_items = target_items[batch_perm]
                if dropout != 0:
                    n_to_drop = int(np.floor(dropout * len(batch_perm)))
                    perm_user = np.random.permutation(len(batch_perm))[:n_to_drop]
                    perm_item = np.random.permutation(len(batch_perm))[:n_to_drop]
                    batch_v_pref = np.copy(batch_items)
                    batch_u_pref = np.copy(batch_users)
                    batch_v_pref[perm_user] = v_pref_last
                    batch_u_pref[perm_item] = u_pref_last
                else:
                    batch_v_pref = batch_items
                    batch_u_pref = batch_users

                Uin = u_pref_expanded[batch_u_pref, :]#.astype(np.float32)
                Vin = v_pref_expanded[batch_v_pref, :]#.astype(np.float32)
                Ucontent = user_content[batch_users, :]#.astype(np.float32)
                Vcontent = item_content[batch_items, :]#.astype(np.float32)
                targets = target_scores[batch_perm]#.astype(np.float32)
                
                Uin = torch.tensor(Uin).to(d_train)
                Vin = torch.tensor(Vin).to(d_train)
                Ucontent = torch.tensor(Ucontent).to(d_train)
                Vcontent = torch.tensor(Vcontent).to(d_train)
                targets = torch.tensor(targets).to(d_train)
                preds, U_embedding, V_embedding = dropout_net.forward(Uin, Vin, Ucontent, Vcontent)
                loss = crit(preds, targets)
                loss_out = loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                f_batch += loss_out
                if np.isnan(f_batch):
                    raise Exception('f is nan')
                n_batch_trained += 1
                pbar.set_description(f'updates={n_batch_trained/1000:.0f}k f={loss_out:.4f} f_tot={f_batch:.2f}')
                
            # step after every ubatch, decay is based on # of ubatch
            scheduler.step()

            if n_step % eval_every == 0:

                dropout_net.eval()
                if task == "seen":
                    out = dat['eval_warm'][['user_id']].copy()
                    eval_warm = dat['eval_warm']['uid'].tolist()
                else:
                    out = dat['eval_cold_user'][['user_id']].copy()
                out['course_id'] = ""
                course = pd.read_csv(os.path.join( data_path, 'course', 'course_feature.csv'))
                course = pd.concat([course, eval_cold_item], ignore_index=True)
                Vcontent = torch.tensor(item_content).to(d_train) # 1, 1
                new_item = torch.tensor( eval_cold_item.drop(['course_id'], axis=1).values.astype(np.float32)).to(d_train)
                Vcontent = torch.cat((Vcontent, new_item),0)
                Vin = torch.tensor(v_pref_expanded[:-1,:]).to(d_train) # 1, 5


                # new_Vin = torch.tensor(v_mean).repeat( len(eval_cold_item),1).to(d_train)
                new_Vin = new_Vin.to(d_train)
                Vin = torch.cat((Vin, new_Vin),0)
                for i in tqdm(range(len(out))):
                    if task == "seen":
                        uid = eval_warm[i]
                        Uin = torch.tensor(u_pref_expanded[uid]).to(d_train) # 1, 5
                        Ucontent = torch.tensor(user_content[uid]).to(d_train)
                    else:
                        Uin = torch.tensor(u_mean).to(d_train) # 1, 5
                        Ucontent = torch.tensor(new_user[i]).to(d_train)
                    Uin = Uin.repeat((Vcontent.shape[0]),1)
                    Ucontent = Ucontent.repeat((Vcontent.shape[0]),1)
                    pred = dropout_net.inference(Uin, Vin, Ucontent, Vcontent)
                    _, eval_preds = torch.topk(pred, k=50, sorted=True)
                    pred_course_id = []
                    for j in eval_preds:
                        pred_course_id.append(course['course_id'][j.item()])
                    out['course_id'][i] = ' '.join(pred_course_id)
                if mode == "val":
                    val = out['course_id'].tolist()
                    val = [ value.split(' ') for value in val]

                    gt = pd.read_csv(f'data/val_{task}.csv')
                    gt = gt.fillna("")
                    gt = gt['course_id'].tolist()
                    gt = [ value.split(' ') for value in gt]

                    accuracy = mapk(gt, val, k=50)
                    print(accuracy)
                # out.to_csv("test.csv", index = False)
                dropout_net.train()

                # checkpoint
                if accuracy > accuracy_best:
                    accuracy_best = accuracy
                    torch.save(dropout_net.state_dict(), os.path.join( checkpoint_path, f"{epoch}_{iter}_model.ckpt"))
                    print(f"save {epoch}_{iter}_model.ckpt")



def load_data(data_path):
    uv_file                 = np.load("asl_model.npz")#, dtype=np.float32)
    # data                    = pd.read_csv( os.path.join(data_path, 'train.csv'))
    data                    = pd.read_csv( os.path.join(data_path, 'course', 'train_preprocess.csv'))
    user_content_file       = os.path.join( data_path, 'course', 'user_feature.csv')
    item_content_file       = os.path.join( data_path, 'course', 'course_feature.csv')
    test_warm_file          = os.path.join( data_path, 'course', f'{mode}_uid.csv')
    test_cold_item_file          = os.path.join( data_path, 'course', 'new_course_feature.csv')
    test_cold_user_file     = os.path.join( data_path, 'course', f'{mode}_unseen_user_feature.csv')

    dat = {}
    # load preference data
    dat['train_data'] = data
    u_pref = np.array(uv_file['user_factors'], dtype=np.float32)
    v_pref = np.array(uv_file['item_factors'], dtype=np.float32)
    dat['u_pref'] = u_pref
    dat['v_pref'] = v_pref

    # pre-process
    _, dat['u_pref_scaled'] = utils.prep_standardize(u_pref)
    _, dat['v_pref_scaled'] = utils.prep_standardize(v_pref)

    # load content data
    user_content = pd.read_csv(user_content_file)
    dat['user_indices'] = np.arange(len(user_content['user_id']))
    user_content = user_content.drop(['user_id'], axis=1)
    item_content = pd.read_csv(item_content_file).drop(['course_id'],axis=1)
    dat['user_content'] = user_content.values#user_content.tolil(copy=False)
    dat['item_content'] = item_content.values#.tolil(copy=False)

    dat['eval_warm'] = pd.read_csv(test_warm_file)
    dat['eval_cold_item'] = pd.read_csv(test_cold_item_file)
    dat['eval_cold_user'] = pd.read_csv(test_cold_user_file)

    return dat


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    parser = argparse.ArgumentParser(description="Demo script to run DropoutNet on RecSys data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', type=str, required=True, help='path to eval in the downloaded folder')

    parser.add_argument('--model-device', type=str, default='cuda:0', help='device to use for training')
    parser.add_argument('--inf-device', type=str, default='cpu', help='device to use for inference')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoint',
                        help='path to dump checkpoint data from TensorFlow')
    parser.add_argument('--tb-log-path', type=str, default=None,
                        help='path to dump TensorBoard logs')
    parser.add_argument('--model-select', nargs='+', type=int,
                        default=[100, 50],
                        help='specify the fully-connected architecture, starting from input,'
                             ' numbers indicate numbers of hidden units',
                        )
    parser.add_argument('--rank', type=int, default=200, help='output rank of latent model')
    parser.add_argument('--dropout', type=float, default=0.3, help='DropoutNet dropout')
    parser.add_argument('--eval-every', type=int, default=1, help='evaluate every X user-batch')
    parser.add_argument('--lr', type=float, default=0.0005, help='starting learning rate')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    main()
