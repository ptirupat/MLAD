import argparse
from datetime import datetime
import os
import torch
import parameters as params
from train import train_model
from configuration import build_config
from tensorboardX import SummaryWriter


def train_classifier(run_id, use_cuda, args):
    cfg = build_config(args.dataset)
    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    writer = SummaryWriter(os.path.join(cfg.tf_logs_dir, str(run_id)))
    for arg in vars(args):
        writer.add_text(arg, str(getattr(args, arg)))
    train_model(cfg, run_id, save_dir, use_cuda, args, writer)


def main(args):
    print("Run description : ", args.run_description)

    # call a function depending on the 'mode' parameter
    if args.train_classifier:
        run_id = args.run_id + '_' + datetime.today().strftime('%d-%m-%y_%H%M')
        use_cuda = torch.cuda.is_available()
        train_classifier(run_id, use_cuda, args)


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to train Multi-label Classification model')

    # 'mode' parameter (mutually exclusive group) with five modes : train/test classifier, train/test generator, test
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--train_classifier', dest='train_classifier', action='store_true',
                       help='Training the Classifier')

    parser.add_argument("--gpu", dest='gpu', type=str, required=False, help='Set CUDA_VISIBLE_DEVICES environment variable, optional')

    parser.add_argument('--run_id', dest='run_id', type=str, required=False, help='Please provide an ID for the current run')

    parser.add_argument('--run_description', dest='run_description', type=str, required=False, help='Please description of the run to write to log')

    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use.', choices=["multithumos", "charades"])

    parser.add_argument('--model_version', type=str, required=True, help='Specify the model version to use for transformer.', 
                        choices=["v1", "baseline1", "baseline2", "baseline3", "baseline4"])

    parser.add_argument('--train_mode', type=str, required=True, help='Specify the training mode.', choices=["fixed", "variable"])

    parser.add_argument('--eval_mode', type=str, required=True, help='Specify the evaluation mode.', choices=["truncate", "pad", "slide"])

    parser.add_argument('--input_type', type=str, required=True, help='Specify if the input is either RGB or Flow.', choices=["rgb", "flow", "combined"])

    parser.add_argument('--num_clips', type=int, default=params.num_clips, help='Number of clips in the input sequence.')

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--skip', type=int, help='Number of clips to skip in the input sequence.')

    group.add_argument("--random_skip", action='store_true')

    parser.add_argument('--batch_size', type=int, default=params.batch_size, help='Batch size.')

    parser.add_argument('--steps', type=int, default=params.steps, help='Number of accumulation steps.')

    parser.add_argument('--feature_dim', type=int, default=params.feature_dim, help='Size of the features for each input clip.')

    parser.add_argument('--hidden_dim', type=int, default=params.hidden_dim, help='Dimension of the internal features of the model.')

    parser.add_argument('--num_layers', type=int, default=params.num_layers, help='Number of layers for encoder/decoder in the transformer model.')

    parser.add_argument('--num_epochs', type=int, default=params.num_epochs, help='Number of epochs.')

    parser.add_argument('--num_workers', type=int, default=params.num_workers, help='Number of workers in the dataloader.')

    parser.add_argument('--learning_rate', type=float, default=params.learning_rate, help='Learning rate for the FC layers.')

    parser.add_argument('--weight_decay', type=float, default=params.weight_decay, help='Weight decay.')

    parser.add_argument('--optimizer', type=str, default=params.optimizer, help='provide optimizer preference')

    parser.add_argument('--f1_threshold', type=float, default=params.f1_threshold, help='Probability threshold for computing F1 Score')

    parser.add_argument('--swa_start', type=int, default=params.swa_start, help='start epoch for SWA')

    parser.add_argument('--swa_update_interval', type=int, default=params.swa_update_interval, help="Update interval for SWA")

    parser.add_argument("--add_background", action='store_true')

    parser.add_argument("--varied_length", action='store_true')
    
    # parse arguments
    args = parser.parse_args()

    # set environment variables to use GPU-0 by default
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # exit when the mode is 'train_classifier' and the parameter 'run_id' is missing
    if args.train_classifier:
        if args.run_id is None:
            parser.print_help()
            exit(1)

    main(args)
