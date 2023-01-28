import os
import yaml
import itertools

# Configures for slurm scripts
gpu_devices = [
    'tesla_v100-sxm2-16gb',
    'tesla_p100-pcie-16gb',
]
partion = 'compute'
shell = ''
conda = 'nrcrs-env'
config_root_path = './crslab/config'

# Path to save the experiment yaml files
ROOT_PATH = './'


def generate_yaml(
    base_yaml,
    _batch_size,
    slurms_tag,
    dataset,
    save_experiment_yamls,
    sample_mask_score=0.95,
    rec_mask_score=0.6,
    sample_num=60,
    centroids_num=1,
    _yaml_idx=0,
    k_popular=1024,
    impatience=5,
    focal_gamma=0,
    abation_type=None,
    XXXX_13=True,
    similar_convitem_in_dec=True,
    popular_stat=True,
    random_seed=3407,
    vknn_type='default',
    ongoing_bias=True,
    ongoing_sim_en=True,
    neighbor_bias=True,
    neighbor_sim_en=True,
    train_conv=False,
    random_bias=False,
    use_pretrained_item_embs=True,
    neighboring_conv_key_vec='user_pref',
    item_bias_type='old',
    pseudo_item_topk=2,
    item_cal_func='cos_sim',
    conv_item_dict_dict_key='full_up',
    entity_item_threshold=None,
    norm_and_weight=False,
    XXXX_17='pre_rec_sum',
    entity_item_op='sum',
    entity_item_topk=1700,
    freeze_modules=None,
    static_neighbor=False,
    ongoing_rep_bias=True,
    neighbor_rep_bias=True,
    dropout=0.3,
    attention_dropout=0.3,
    relu_dropout=0.1,
    label_smoothing=0.5,
    tgt_neighbor_num=0,
    pseudo_k_times=1,
    pseudo_item_threshold=0.6,
    cont_loss_threshold=0.4,
    pseudo_item_freeze=False,
    pseudo_epoch=1,
    contrastive_loss=None,
    classifier_pretrain_epoch=0,
    pred_k_neighbor=False,
    cont_ex_num=0,
    XXXX_28=0.95,
    knn_score_type='gate',
    pseudo_smoothing=0.4,
    rec_optimizer_lr=0.0005,
    sim_crs_norm_type='linear',
    neighboring_items=False,
    neighboring_convs=False,
    cont_loss_weight=0.5,
    fill_empty_topk=1,
    freeze_parameters=False,
    freeze_parameters_name='k',
    pretrain_pseudo_epochs=0,
    wandb_project='nr-crs',
    repeat_ep_idx=0,
    ni_loss=False,
    XXXX_33=False,
    XXXX_32=False,
):
    _slurms = []
    _ep_model_tags = []

    gen_yaml = base_yaml
    gen_yaml['label_smoothing'] = label_smoothing
    gen_yaml['sample_mask_score'] = sample_mask_score
    gen_yaml['rec_mask_score'] = rec_mask_score
    gen_yaml['centroids_num'] = centroids_num
    gen_yaml['XXXX_13'] = XXXX_13
    gen_yaml['similar_convitem_in_dec'] = similar_convitem_in_dec
    gen_yaml['dropout'] = dropout
    gen_yaml['sample_num'] = sample_num
    gen_yaml['k_popular'] = k_popular
    gen_yaml['focal_gamma'] = focal_gamma
    gen_yaml['rec']['impatience'] = impatience
    gen_yaml['ablation'] = abation_type
    gen_yaml['load_pretrain'] = False
    gen_yaml['save_pretrain'] = False
    gen_yaml['popular_stat'] = popular_stat
    gen_yaml['random_seed'] = random_seed
    gen_yaml['rec']['batch_size'] = _batch_size
    gen_yaml['conv']['batch_size'] = _batch_size
    gen_yaml['vknn_type'] = vknn_type
    gen_yaml['ongoing_bias'] = ongoing_bias
    gen_yaml['ongoing_sim_en'] = ongoing_sim_en
    gen_yaml['neighbor_bias'] = neighbor_bias
    gen_yaml['neighbor_sim_en'] = neighbor_sim_en
    gen_yaml['random_bias'] = random_bias
    gen_yaml['use_pretrained_item_embs'] = use_pretrained_item_embs
    gen_yaml['neighboring_conv_key_vec'] = neighboring_conv_key_vec
    gen_yaml['item_bias_type'] = item_bias_type
    gen_yaml['item_cal_func'] = item_cal_func
    gen_yaml['entity_item_threshold'] = entity_item_threshold
    gen_yaml['norm_and_weight'] = norm_and_weight
    gen_yaml['conv_item_dict_dict_key'] = conv_item_dict_dict_key
    gen_yaml['XXXX_17'] = XXXX_17
    gen_yaml['entity_item_op'] = entity_item_op
    gen_yaml['entity_item_topk'] = entity_item_topk
    gen_yaml['freeze_modules'] = freeze_modules
    gen_yaml['static_neighbor'] = static_neighbor
    gen_yaml['ongoing_rep_bias'] = ongoing_rep_bias
    gen_yaml['neighbor_rep_bias'] = neighbor_rep_bias
    gen_yaml['tgt_neighbor_num'] = tgt_neighbor_num
    gen_yaml['pseudo_k_times'] = pseudo_k_times
    gen_yaml['pseudo_item_threshold'] = pseudo_item_threshold
    gen_yaml['cont_loss_threshold'] = cont_loss_threshold
    gen_yaml['pseudo_item_freeze'] = pseudo_item_freeze
    gen_yaml['pseudo_epoch'] = pseudo_epoch
    gen_yaml['pretrain_pseudo_epochs'] = pretrain_pseudo_epochs
    gen_yaml['wandb_name'] = f'{slurms_tag}_{_yaml_idx}'
    gen_yaml['wandb_project'] = wandb_project
    gen_yaml['contrastive_loss'] = contrastive_loss
    gen_yaml['classifier_pretrain_epoch'] = classifier_pretrain_epoch
    gen_yaml['pred_k_neighbor'] = pred_k_neighbor
    gen_yaml['cont_ex_num'] = cont_ex_num
    gen_yaml['sim_crs_norm_type'] = sim_crs_norm_type
    gen_yaml['attention_dropout'] = attention_dropout
    gen_yaml['relu_dropout'] = relu_dropout
    gen_yaml['impatience'] = impatience
    gen_yaml['pseudo_smoothing'] = pseudo_smoothing
    gen_yaml['knn_score_type'] = knn_score_type
    gen_yaml['rec_optimizer_lr'] = rec_optimizer_lr
    gen_yaml['XXXX_28'] = XXXX_28
    gen_yaml['neighboring_items'] = neighboring_items
    gen_yaml['neighboring_convs'] = neighboring_convs
    gen_yaml['cont_loss_weight'] = cont_loss_weight
    gen_yaml['fill_empty_topk'] = fill_empty_topk
    gen_yaml['freeze_parameters'] = freeze_parameters
    gen_yaml['freeze_parameters_name'] = freeze_parameters_name
    gen_yaml['pseudo_item_topk'] = pseudo_item_topk
    gen_yaml['ni_loss'] = ni_loss
    gen_yaml['XXXX_33'] = XXXX_33
    gen_yaml['XXXX_32'] = XXXX_32

    ep_model_tag = f"ep{repeat_ep_idx}_nc{str(neighboring_convs)[0]}_ibw{int(XXXX_28*10)}_pit{pseudo_item_topk}_pe{pseudo_epoch}_pif{str(pseudo_item_freeze)[0]}_lb{int(label_smoothing*10)}_abure{str(XXXX_33)[0]}_abupe{str(XXXX_32)[0]}_ncn{sample_num}"

    ifnr = 'our_nr' if XXXX_13 else 'no_nr'
    ifnr_dec = 'nrde' if similar_convitem_in_dec else 'oride'
    if abation_type is None:
        ep_model_tag = f'{slurms_tag}_{ifnr}_{ep_model_tag}'
    else:
        ep_model_tag = f'{slurms_tag}_{ifnr}_ab_{abation_type}_{ifnr_dec}_{ep_model_tag}'
    gen_yaml['ep_model_tag'] = ep_model_tag

    save_yaml_path = os.path.join(save_experiment_yamls,
                                  f'{dataset}_{_yaml_idx}.yaml')
    with open(save_yaml_path, 'w') as file:
        documents = yaml.dump(gen_yaml, file)
    print(f'saved {save_yaml_path}')

    train_conv_sh = '' if train_conv else '# '

    shs = [
        f'sh script/{dataset}/train/{dataset}_rec_train_template.sh {slurms_tag} {_yaml_idx} {_batch_size}',
        f'{train_conv_sh}sh script/{dataset}/train/{dataset}_conv_train_template.sh {slurms_tag} {_yaml_idx} 256 {ep_model_tag} 70'
    ]
    _slurms.append('\n'.join(shs))
    _ep_model_tags.append(ep_model_tag)
    _yaml_idx += 1

    return _slurms, _yaml_idx, _ep_model_tags


def parse_paras(params, base_yaml, _batch_size, dataset, save_experiment_yamls,
                yaml_index):
    keys = list(params)
    slurms = []
    ep_model_tags = []
    for values in itertools.product(*map(params.get, keys)):
        _slurms, yaml_index, _ep_model_tags = generate_yaml(
            base_yaml,
            _batch_size,
            slurms_tag,
            dataset,
            save_experiment_yamls,
            _yaml_idx=yaml_index,
            **dict(zip(keys, values)))
        slurms += _slurms
        ep_model_tags += _ep_model_tags
    return slurms, ep_model_tags, yaml_index


def generate_slurms_on_dataset(dataset='redial',
                               slurms_tag='main_ep'):

    with open(os.path.join(config_root_path, f'{dataset}.yaml')) as file:
        base_yaml = yaml.load(file, Loader=yaml.FullLoader)

    save_experiment_yamls = os.path.join(config_root_path, 'experiment_yamls',
                                         slurms_tag)
    if not os.path.exists(save_experiment_yamls):
        os.makedirs(save_experiment_yamls)

    yaml_index = 0
    slurms = []
    ep_model_tags = []

    for _batch_size in [32]:
        # main
        params = {
            'wandb_project': ['SIGIR23-Results'],
            'neighboring_convs': [True],
            'train_conv': [True],
            'label_smoothing': [0.9],
            'pseudo_epoch': [1],
            'pseudo_item_topk': [1],
            'XXXX_33': [False],
            'XXXX_32': [False],
            'repeat_ep_idx': [0],
        }
        tmp_slurms, tmp_ep_model_tags, yaml_index = parse_paras(
            params, base_yaml, _batch_size, dataset, save_experiment_yamls,
            yaml_index)
        slurms += tmp_slurms
        ep_model_tags += tmp_ep_model_tags

        # ablation study of neighboring conversations
        params = {
            'wandb_project': ['SIGIR23-Results'],
            'neighboring_convs': [True],
            'train_conv': [False],
            'label_smoothing': [0.9],
            'pseudo_epoch': [1],
            'pseudo_item_topk': [1],
            'XXXX_33': [True],
            'XXXX_32': [False],
            'repeat_ep_idx': [0],
        }
        tmp_slurms, tmp_ep_model_tags, yaml_index = parse_paras(
            params, base_yaml, _batch_size, dataset, save_experiment_yamls,
            yaml_index)
        slurms += tmp_slurms
        ep_model_tags += tmp_ep_model_tags

        params = {
            'wandb_project': ['SIGIR23-Results'],
            'neighboring_convs': [True],
            'train_conv': [False],
            'label_smoothing': [0.9],
            'pseudo_epoch': [1],
            'pseudo_item_topk': [1],
            'XXXX_33': [False],
            'XXXX_32': [True],
            'repeat_ep_idx': [0],
        }
        tmp_slurms, tmp_ep_model_tags, yaml_index = parse_paras(
            params, base_yaml, _batch_size, dataset, save_experiment_yamls,
            yaml_index)
        slurms += tmp_slurms
        ep_model_tags += tmp_ep_model_tags

        params = {
            'wandb_project': ['SIGIR23-Results'],
            'neighboring_convs': [False],
            'train_conv': [False],
            'label_smoothing': [0.9],
            'pseudo_epoch': [1],
            'pseudo_item_topk': [1],
            'XXXX_33': [False],
            'XXXX_32': [False],
            'repeat_ep_idx': [0],
        }
        tmp_slurms, tmp_ep_model_tags, yaml_index = parse_paras(
            params, base_yaml, _batch_size, dataset, save_experiment_yamls,
            yaml_index)
        slurms += tmp_slurms
        ep_model_tags += tmp_ep_model_tags

        # ablation study of neighboring items
        params = {
            'wandb_project': ['SIGIR23-Results'],
            'neighboring_convs': [True],
            'train_conv': [False],
            'label_smoothing': [0.9],
            'pseudo_epoch': [0],
            'pseudo_item_topk': [1],
            'XXXX_33': [False],
            'XXXX_32': [False],
            'repeat_ep_idx': [0],
        }
        tmp_slurms, tmp_ep_model_tags, yaml_index = parse_paras(
            params, base_yaml, _batch_size, dataset, save_experiment_yamls,
            yaml_index)
        slurms += tmp_slurms
        ep_model_tags += tmp_ep_model_tags

        params = {
            'wandb_project': ['SIGIR23-Results'],
            'neighboring_convs': [True],
            'train_conv': [False],
            'label_smoothing': [0.9],
            'pseudo_epoch': [10],
            'pseudo_item_topk': [1],
            'XXXX_33': [False],
            'XXXX_32': [False],
            'repeat_ep_idx': [0],
        }
        tmp_slurms, tmp_ep_model_tags, yaml_index = parse_paras(
            params, base_yaml, _batch_size, dataset, save_experiment_yamls,
            yaml_index)
        slurms += tmp_slurms
        ep_model_tags += tmp_ep_model_tags

        # impact of neighboring conversations
        params = {
            'wandb_project': ['SIGIR23-Results'],
            'neighboring_convs': [True],
            'train_conv': [False],
            'label_smoothing': [0.9],
            'pseudo_epoch': [1],
            'pseudo_item_topk': [1],
            'XXXX_33': [False],
            'XXXX_32': [False],
            'sample_num': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'repeat_ep_idx': [0],
        }
        tmp_slurms, tmp_ep_model_tags, yaml_index = parse_paras(
            params, base_yaml, _batch_size, dataset, save_experiment_yamls,
            yaml_index)
        slurms += tmp_slurms
        ep_model_tags += tmp_ep_model_tags

        # impact of pseudo-labeled data
        params = {
            'wandb_project': ['SIGIR23-Results'],
            'neighboring_convs': [True],
            'train_conv': [False],
            'label_smoothing': [0.9],
            'pseudo_epoch': [10],
            'pseudo_item_topk': [1, 2, 3, 4, 5],
            'XXXX_33': [False],
            'XXXX_32': [False],
            'repeat_ep_idx': [0, 1, 2, 3, 4],
        }
        tmp_slurms, tmp_ep_model_tags, yaml_index = parse_paras(
            params, base_yaml, _batch_size, dataset, save_experiment_yamls,
            yaml_index)
        slurms += tmp_slurms
        ep_model_tags += tmp_ep_model_tags

    return output_slurm(slurms, ep_model_tags, dataset, slurms_tag)


def output_slurm(slurms, ep_model_tags, dataset, slurms_tag):
    run_sbatch = []
    slurm_root_path = f'script/{dataset}'

    slurm_output_path = os.path.join(ROOT_PATH, 'slurms_output', dataset,
                                     slurms_tag)
    if not os.path.exists(slurm_output_path):
        os.makedirs(slurm_output_path)

    save_experiment_slurms = os.path.join(slurm_root_path, 'experiment_slurms',
                                          slurms_tag)
    if not os.path.exists(save_experiment_slurms):
        os.makedirs(save_experiment_slurms)

    for idx, slurm in enumerate(slurms):
        gpu = gpu_devices[idx % len(gpu_devices)]
        save_slurm_path = os.path.join(save_experiment_slurms,
                                       f'base_{idx}.slurm')
        with open(save_slurm_path, 'w') as f:
            base = f"""#!/bin/bash
#SBATCH -J {dataset[0]}_ep_{ep_model_tags[idx]}    
#SBATCH -o {slurm_output_path}/{dataset[0]}_ep_{ep_model_tags[idx]}.out                           
#SBATCH -p {partion}                        
#SBATCH -N 1                                  
#SBATCH -t 72:00:00
#SBATCH --mem=60G
#SBATCH --gres=gpu:{gpu}:1

{shell}

source activate {conda}
"""
            f.write(base + '\n' + slurm)
        run_sbatch.append(f'sbatch {save_slurm_path}')
        print(f'saved {save_slurm_path}')

    return run_sbatch


if __name__ == '__main__':
    run_sbatch = []
    slurms_tag = 'nr-crs'
    run_sbatch += generate_slurms_on_dataset(dataset='redial',
                                             slurms_tag=slurms_tag)

    with open(f'run_sbatch_{slurms_tag}.sh', 'w') as f:
        f.write('\n'.join(run_sbatch))
        print(f'saved run_sbatch_{slurms_tag}.sh')
