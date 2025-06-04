import os
import time
import argparse
from model import build_model
from metric import compute_metrics, MetricLog
from utils import load_cfg_file, summary_results, make_dirs, init_logger, set_random_seed, dict_to_log

def config_params():
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--config', required=True, help='the config file path')
    
    args = parser.parse_args()
    return args

def main():

    args = config_params()
    cfg = load_cfg_file(args.config)
    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    experiment_dir = os.path.join(cfg.experiment_name, "test", timestamp)
    make_dirs(experiment_dir, "test")
    
    logger = init_logger(experiment_dir)

    dict_to_log(cfg, logger)

    set_random_seed(cfg.seed)

    model = build_model(cfg.model)

    scene_list = os.listdir(cfg.data_path)
    test_metrics = MetricLog()

    for scene in scene_list:
        scene_path = os.path.join(cfg.data_path, scene)
        logger.info("=" * 20)
        logger.info("Scene: {}".format(scene))

        out = model.run(scene_path)

        cfg.info["gt_file_path"] = os.path.join(scene_path, "PointCloud")
        scene_metrics = compute_metrics(out, None, cfg.info)

        test_metrics.add_metrics(scene_metrics)
        logger.info("Result: {}".format(scene_metrics))

    results = summary_results("test", test_metrics, None)
    
    logger.info("Overall result: {}".format(results))
    

if __name__ == "__main__":
    main()

