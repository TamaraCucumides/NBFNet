import os
import sys
import pprint

import torch

from torchdrug import core
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import dataset, layer, model, task, util

def save_tensor(relation, preds, folder="../data"):
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

def batch_tensors(tensor, batch_size):
    num_batches = (tensor.size(0) + batch_size - 1) // batch_size
    for i in range(0, tensor.size(0), batch_size):
        yield tensor[i:min(i+batch_size, tensor.size(0))]

def obtain_results(solver, triplet):
    h, t, r = triplet
    triplet = torch.as_tensor([[h, t, r]], device=solver.device)
    #inverse = torch.as_tensor([[t, h, r + num_relation]], device=solver.device)
    solver.model.eval()
    pred, (mask, target) = solver.model.predict_and_target(triplet)
    #pos_pred = pred.gather(-1, target.unsqueeze(-1))
    #rankings = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    #rankings = rankings.squeeze(0)

    return torch.round(pred[0][0]).to(torch.float16)

def batch_results(solver, batch):
    solver.model.eval()
    pred = solver.model.predict(batch, only_head=True)
    #pred, (mask, target) = solver.model.predict_and_target(batch)
    pred_cpu = pred.to("cpu")
    #return torch.round(pred[0][0]).to(torch.float16)
    return pred_cpu


if __name__ == "__main__":
    print("Torch version", torch.__version__)
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
      batch_size = cfg.engine.batch_size
      print("######################################")
      print("Relation", relation)
      index = 0
      triples = torch.tensor(create_triples(relation), device="cpu")
      result_tensor = torch.empty(14541, 14541, dtype=torch.float16, device="cpu")

      if False:
        for batch in batch_tensors(triples, batch_size):
          print("Numero batch", count)
          torch.cuda.empty_cache()
          batch_cuda = batch.to("cuda")
          batch_preds = batch_results(solver, batch_cuda)
          #print("Batch_preds", batch_preds)
          print("Dev. b atch preds", batch_preds.device)
          batch_size = batch_preds.size(0)
          result_tensor[index:index+batch_size] = batch_preds
          index += batch_size
          count +=1
        save_tensor(relation, result_tensor)
        
      for t in triples:
        print("Triplet", t)
        tensor_row = obtain_results(solver, t).unsqueeze(0).cpu()  # Unsqueezing to add a new dimension (to make it a row tensor)
        result_tensor[index] = tensor_row
        index += 1
      save_tensor(relation, result_tensor)


    
