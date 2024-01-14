import os
import sys
import math
import pprint

import torch

from torchdrug import core
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import dataset, layer, model, task, util


def test(cfg, solver):
    #solver.model.split = "valid"
    #solver.evaluate("valid")
    # Test only on test set? What is the test set here? How can I get the full graph completion?
    solver.model.split = "test"
    solver.evaluate("test")


if __name__ == "__main__":
    if torch.cuda.is_available():
        args, vars = util.parse_args()
        cfg = util.load_config(args.config, context=vars)
        working_dir = util.create_working_directory(cfg)
    
        torch.manual_seed(args.seed + comm.get_rank())
    
        logger = util.get_root_logger()
        if comm.get_rank() == 0:
            logger.warning("Config file: %s" % args.config)
            logger.warning(pprint.pformat(cfg))
    
        dataset = core.Configurable.load_config_dict(cfg.dataset)
        solver = util.build_solver(cfg, dataset)

        #Dont train
        #train_and_validate(cfg, solver)
      
        test(cfg, solver)
    else:
        print("Cuda disponible?", torch.cuda.is_available())
