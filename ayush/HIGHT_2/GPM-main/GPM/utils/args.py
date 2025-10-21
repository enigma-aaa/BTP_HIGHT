import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_params", action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--group', type=str, default='default')

    # Dataset
    parser.add_argument('--dataset', '--data', type=str, default="cora")
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', '--bs', type=int, default=256)
    parser.add_argument('--split', type=str, default='public')
    parser.add_argument('--split_repeat', type=int, default=3)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=0)

    # If you want to use the pretrain model, you need to set the pretrain_data and pretrain_epoch
    parser.add_argument('--pretrain_data', type=str, default='none')
    parser.add_argument('--pretrain_epoch', type=int, default=0)
    parser.add_argument('--linear_probe', action='store_true', help='Only train the linear classifier')
    parser.add_argument('--reset_head', action='store_true', help='Reset the head of the model')
    parser.add_argument('--inference', action='store_true', help='Only inference the model')

    # Patterns and Augmentation
    parser.add_argument('--pre_sample_pattern_num', type=int, default=128)
    parser.add_argument('--pre_sample_batch_size', type=int, default=8192)
    parser.add_argument('--num_patterns', type=int, default=16)
    parser.add_argument('--pattern_size', type=int, default=8)
    parser.add_argument('--multiscale', type=int, nargs='+', default=[2, 4, 6, 8])
    parser.add_argument('--p', type=float, default=1)
    parser.add_argument('--q', type=float, default=1)

    # Encoder
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--num_layers', '--layers', type=int, default=1)
    parser.add_argument('--norm_first', action='store_true')

    # PE
    parser.add_argument('--node_pe', type=str, default='rw', choices=['rw', 'lap', 'none'])
    parser.add_argument('--node_pe_dim', type=int, default=8)
    parser.add_argument('--pe_encoder', type=str, default='gru', choices=['mean', 'gru', 'none'])
    parser.add_argument('--pe_weight', type=float, default=1.0)

    # Pattern Encoder
    parser.add_argument('--pattern_encoder', type=str, default='transformer', choices=['mean', 'gru', 'transformer'])
    parser.add_argument('--pattern_encoder_layers', type=int, default=1)
    parser.add_argument('--pattern_encoder_heads', type=int, default=1)

    # VQ and Others
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--codebook_size', type=int, default=8196)
    parser.add_argument('--use_cls_token', action='store_true')
    parser.add_argument('--use_attn_fusion', action='store_true')

    # Optimization
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', '--decay', type=float, default=0)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    parser.add_argument('--scheduler', type=str, default='warmup', choices=['warmup', 'cosine', 'none'])
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--eta_min', type=float, default=1e-6)

    parser.add_argument('--opt_beta1', type=float, default=0.9)
    parser.add_argument('--opt_beta2', type=float, default=0.999)
    parser.add_argument('--opt_eps', type=float, default=1e-8)

    # Ablation
    # If set "use_params", the model will use the default parameters, which cannot be changed by the arguments
    # If you want to change the parameters, we can set the ablation parameters here
    parser.add_argument('--no_node_pe', action='store_true')
    parser.add_argument('--no_ap', action='store_true')

    args = parser.parse_args()
    return vars(args)


def get_da_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_params", action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--group', type=str, default='default')

    # Dataset
    # parser.add_argument('--dataset', '--data', type=str, default="cora")
    parser.add_argument('--source', type=str, default='acm')
    parser.add_argument('--target', type=str, default='dblp')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', '--bs', type=int, default=256)
    parser.add_argument('--split', type=str, default='public')
    parser.add_argument('--split_repeat', type=int, default=3)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=0)

    # If you want to use the pretrain model, you need to set the pretrain_data and pretrain_epoch
    parser.add_argument('--pretrain_data', type=str, default='none')
    parser.add_argument('--pretrain_epoch', type=int, default=0)
    parser.add_argument('--linear_probe', action='store_true', help='Only train the linear classifier')
    parser.add_argument('--reset_head', action='store_true', help='Reset the head of the model')
    parser.add_argument('--inference', action='store_true', help='Only inference the model')

    # Patterns and Augmentation
    parser.add_argument('--pre_sample_pattern_num', type=int, default=128)
    parser.add_argument('--pre_sample_batch_size', type=int, default=8192)
    parser.add_argument('--num_patterns', type=int, default=16)
    parser.add_argument('--pattern_size', type=int, default=8)
    parser.add_argument('--multiscale', type=int, nargs='+', default=[2, 4, 6, 8])
    parser.add_argument('--p', type=float, default=1)
    parser.add_argument('--q', type=float, default=1)

    # Encoder
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--num_layers', '--layers', type=int, default=1)
    parser.add_argument('--norm_first', action='store_true')

    # PE
    parser.add_argument('--node_pe', type=str, default='rw', choices=['rw', 'lap', 'none'])
    parser.add_argument('--node_pe_dim', type=int, default=8)
    parser.add_argument('--pe_encoder', type=str, default='gru', choices=['mean', 'gru', 'none'])
    parser.add_argument('--pe_weight', type=float, default=1.0)

    # Pattern Encoder
    parser.add_argument('--pattern_encoder', type=str, default='transformer', choices=['mean', 'gru', 'transformer'])
    parser.add_argument('--pattern_encoder_layers', type=int, default=1)
    parser.add_argument('--pattern_encoder_heads', type=int, default=1)

    # VQ and Others
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--codebook_size', type=int, default=8196)
    parser.add_argument('--use_cls_token', action='store_true')
    parser.add_argument('--use_attn_fusion', action='store_true')

    # Optimization
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', '--decay', type=float, default=0)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    parser.add_argument('--scheduler', type=str, default='warmup', choices=['warmup', 'cosine', 'none'])
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--eta_min', type=float, default=1e-6)

    parser.add_argument('--opt_beta1', type=float, default=0.9)
    parser.add_argument('--opt_beta2', type=float, default=0.999)
    parser.add_argument('--opt_eps', type=float, default=1e-8)

    args = parser.parse_args()
    return vars(args)
