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

  print("Saving relation", relation)
  if not os.path.exists(folder):
    os.makedirs(folder)
  pred_filename = os.path.join(folder, f'{relation}.pt')

  preds_save = torch.sigmoid(preds).half()
  torch.save(preds_save, pred_filename)

def create_triples(relation, num_triples):
  """Create all the relevant triples"""

  triples = [] 
  for i in range(num_triples):
    t = [i, 1, relation]
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
    batch_gpu = batch.to("cuda")
    pred = solver.model.predict(batch_gpu)
    print("*")
    #pred, (mask, target) = solver.model.predict_and_target(batch)
    pred_cpu = pred.to("cpu")
    del pred
    #return torch.round(pred[0][0]).to(torch.float16)
    return pred_cpu[:, 0, :]

@torch.no_grad()
def batch_evaluate(solver, batch):
  batch = batch.to("cuda")
  pred, (mask, target) = solver.model.predict_and_target(batch)

  pred_cpu = pred.cpu()
  #target_cpu = target.cpu()

  del pred
  del target
  
  return pred_cpu[:, 0, :]
  


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

    print("Cuda memory after building solver", torch.cuda.memory_allocated())

    # get the current working directory
    current_working_directory = os.getcwd()

    # print output to the console
    print("Current directory", current_working_directory)


    for relation in range(80):
      batch_size = cfg.engine.batch_size
      print("######################################")
      print("Relation", relation)
      index = 0
      count = 0
      num_triples = 14541
      triples = torch.tensor(create_triples(relation, num_triples), device="cpu")
      result_tensor = torch.empty(14541, 14541, device="cpu")

      batches_operation = True

      if batches_operation:
        solver.model.eval()
        for batch in batch_tensors(triples, batch_size):
          print("Numero batch", count)
          pred = batch_evaluate(solver, batch)
          batch_actual_size = pred.size(0)
          result_tensor[index:index+batch_actual_size] = pred
          index += batch_actual_size
          count +=1
        save_tensor(relation, result_tensor)
        


    
