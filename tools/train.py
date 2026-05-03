import argparse
import logging
import os
import os.path as osp
import sys
from datetime import datetime
from pathlib import Path

# Add src to sys.path so mowing_terrain_seg can be imported without installation
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import mowing_terrain_seg

mowing_terrain_seg.register_all()

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--work-dir',
        default=None,
        help='the dir to save logs and models')
    parser.add_argument(
        '--exp-name',
        default=None,
        help='Name of the experiment. Defaults to the config filename if not specified.')
    parser.add_argument(
        '--run-id',
        default=None,
        help='ID of the run. Defaults to a timestamp if not specified.')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='Resume from the latest checkpoint in the work_dir (reuse the same run_id directory).')
    parser.add_argument(
        '--load-from',
        help='The checkpoint file to load weights from.')
    parser.add_argument(
        '--tracking-uri',
        default=None,
        help='MLflow tracking URI. Overrides the value in the config.')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # MLflow tracking URI override
    if args.tracking_uri:
        if 'visualizer' in cfg and 'vis_backends' in cfg.visualizer:
            for backend in cfg.visualizer.vis_backends:
                if backend.get('type') == 'MLflowVisBackend':
                    backend['tracking_uri'] = args.tracking_uri
                    print_log(f'Overriding MLflow tracking URI to: {args.tracking_uri}', logger='current')

    # Determine Experiment Name (CLI > Config > Filename)
    if args.exp_name:
        cfg.exp_name = args.exp_name
    elif cfg.get('exp_name', None) is None:
        cfg.exp_name = osp.splitext(osp.basename(args.config))[0]

    # Determine Run ID (CLI > Config > Timestamp)
    if args.run_id:
        cfg.run_id = args.run_id
    elif cfg.get('run_id', None) is None:
        cfg.run_id = f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    # work_dir logic (CLI > Config > Default Path)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # Use the already-determined names to build the path
        cfg.work_dir = osp.join('./work_dirs', cfg.exp_name, cfg.run_id)

    print_log(f'Experiment: {cfg.exp_name}', logger='current')
    print_log(f'Run ID:     {cfg.run_id}', logger='current')
    print_log(f'Work Dir:   {cfg.work_dir}', logger='current')


    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    # load from a specific checkpoint
    if args.load_from is not None:
        cfg.load_from = args.load_from

    if cfg.resume and cfg.get('load_from', None) is not None:
        raise ValueError(
            'Both "resume" and "load_from" are specified in the config or CLI. '
            'Please use only one: "resume" to continue a run, or "load_from" '
            'to initialize a new run with weights.')

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
