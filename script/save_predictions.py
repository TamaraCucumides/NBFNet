import os
import sys
import pprint

import torch

from torchdrug import core
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import dataset, layer, model, task, util

def save_tensor(relation, preds, folder="data"):
  """receives the relation number, alongside with the tensor with all predictions 
  and saves it into the respective .pt"""

  if not os.path.exists(folder):
    os.makedirs(folder)
    pred_filename = os.path.join(folder, f'{relation}.pt')
    torch.save(preds, pred_filename)

def create_triples(relation):
  """Create all the relevant triples"""

  triples = [] 
  for i in range(14541):
    t = [i, relation, 1]
    triples.append(t)
  return triples

def obtain_results(solver, triplet):
    h, t, r = triplet
    triplet = torch.as_tensor([[h, t, r]], device=solver.device)
    #inverse = torch.as_tensor([[t, h, r + num_relation]], device=solver.device)
    solver.model.eval()
    pred, (mask, target) = solver.model.predict_and_target(triplet)
    #pos_pred = pred.gather(-1, target.unsqueeze(-1))
    #rankings = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    #rankings = rankings.squeeze(0)

    return torch.round(pred[0][0] * 100).to(torch.int8)


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    logger.warning("Config file: %s" % args.config)
    logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)

    # get the current working directory
    current_working_directory = os.getcwd()

    # print output to the console
    print("Current directory", current_working_directory)

    for relation in range(4):
      triples = create_triples(relation)
      result_tensor = torch.empty(0, 14541, dtype=torch.int8, device=solver.device)
      print("Result tensor shape", result_tensor.shape)
      print(result_tensor)
      for t in triples:
        tensor_row = obtain_results(solver, t).unsqueeze(0)  # Unsqueezing to add a new dimension (to make it a row tensor)
        print("Tensor row shape", tensor_row.shape)
        print(tensor_row)
        result_tensor = torch.cat((result_tensor, tensor_row.cuda()), dim=0)
      save_tensor(relation, result_tensor)


    
