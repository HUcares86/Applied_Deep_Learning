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
from pathlib import Path
import postprocess as post

mode = 'test' 
# mode = 'test'

def main():
    data_path           = args.data_dir
    model_path     = args.model_path
    model_select        = args.model_select

    rank_out            = args.rank
    eval_batch_size     = 5000
    task                = args.task

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
    u_pref_expanded = np.vstack([u_pref_scaled, np.zeros_like(u_pref_scaled[0, :])])
    u_mean =  np.mean(u_pref_expanded, 0)
    v_mean =  np.mean(v_pref_expanded, 0)
    new_Vin = torch.tensor(v_mean).repeat( len(eval_cold_item),1)
    new_user = eval_cold_user.drop(['user_id'], axis=1).values
       
    eval_batch_size = eval_batch_size
    dropout_net = model.get_model(latent_rank_in=u_pref.shape[1],
                               user_content_rank=user_content.shape[1],
                               item_content_rank=item_content.shape[1],
                               model_select=model_select,
                               rank_out=rank_out)

    dropout_net.load_state_dict(torch.load( model_path))
    d_train = torch.device(args.model_device)

    dropout_net.to(d_train)
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

    out = post.zero(out, course)
    if task == "seen":
        out = post.remove_seen( os.path.join(data_path,'train.csv'),out)
    if mode == "val":
        val = out['course_id'].tolist()
        val = [ value.split(' ') for value in val]

        gt = pd.read_csv(f'data/val_{task}.csv')
        gt = gt.fillna("")
        gt = gt['course_id'].tolist()
        gt = [ value.split(' ') for value in gt]

        accuracy = mapk(gt, val, k=50)
        print(accuracy)
    out.to_csv(args.output, index = False)



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
    parser.add_argument('--model_path', type=str, required=True ,help='model_path')
    parser.add_argument('--dropout', type=float, default=0.3, help='DropoutNet dropout')
    parser.add_argument('--eval-every', type=int, default=1, help='evaluate every X user-batch')
    parser.add_argument('--lr', type=float, default=0.0005, help='starting learning rate')
    parser.add_argument('--task', type=str, required=True, help='seen/unseen')
    parser.add_argument('--output', type=str, required=True, help='output file')

    args = parser.parse_args()

    output_file = Path(args.output)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    main()
