import os
import sys
import pprint
import time

import torch

from torchdrug import core
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import dataset, layer, model, task, util


vocab_file = os.path.join(os.path.dirname(__file__), "../data/fb15k237_entity.txt")
vocab_file = os.path.abspath(vocab_file)


def load_vocab(dataset):
    entity_mapping = {}
    with open(vocab_file, "r") as fin:
        for line in fin:
            k, v = line.strip().split("\t")
            entity_mapping[k] = v
    entity_vocab = [entity_mapping[t] for t in dataset.entity_vocab]
    relation_vocab = ["%s (%d)" % (t[t.rfind("/") + 1:].replace("_", " "), i)
                      for i, t in enumerate(dataset.relation_vocab)]

    return entity_vocab, relation_vocab


def visualize_path(solver, triplet, entity_vocab, relation_vocab):
    num_relation = len(relation_vocab)
    h, t, r = triplet.tolist()
    triplet = torch.as_tensor([[h, t, r]], device=solver.device)
    inverse = torch.as_tensor([[t, h, r + num_relation]], device=solver.device)
    solver.model.eval()
    pred, (mask, target) = solver.model.predict_and_target(triplet)
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    rankings = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    rankings = rankings.squeeze(0)

    logger.warning("")
    samples = (triplet, inverse)
    for sample, ranking in zip(samples, rankings):
        h, t, r = sample.squeeze(0).tolist()
        h_name = entity_vocab[h]
        t_name = entity_vocab[t]
        r_name = relation_vocab[r % num_relation]
        if r >= num_relation:
            r_name += "^(-1)"
        logger.warning(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        logger.warning("rank(%s | %s, %s) = %g" % (t_name, h_name, r_name, ranking))

        paths, weights = solver.model.visualize(sample)
        for path, weight in zip(paths, weights):
            triplets = []
            for h, t, r in path:
                h_name = entity_vocab[h]
                t_name = entity_vocab[t]
                r_name = relation_vocab[r % num_relation]
                if r >= num_relation:
                    r_name += "^(-1)"
                triplets.append("<%s, %s, %s>" % (h_name, r_name, t_name))
            logger.warning("weight: %g\n\t%s" % (weight, " ->\n\t".join(triplets)))


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    logger.warning("Config file: %s" % args.config)
    logger.warning(pprint.pformat(cfg))

    #if cfg.dataset["class"] != "FB15k237":
    #    raise ValueError("Visualization is only implemented for FB15k237")

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)

    entity_vocab, relation_vocab = load_vocab(dataset)

    num_relation = len(relation_vocab)

    #Symmetry of predictions 
    if False:
        for i in range(5):
            triplet = solver.test_set[i]
            h, t, r = triplet.tolist()
            triplet = torch.as_tensor([[h, t, r]], device=solver.device)
    
            print("h", h)
            print("t", t)
    
    
            pred, (mask, target) = solver.model.predict_and_target(triplet)
    
            print("pred[0][t]", pred[0][0][t])
            print("pred[1][h]", pred[0][1][h])

    # Correspondance of entities and relations
    if False:
        for i in range(10):
            triplet = solver.test_set[i]
            h, t, r = triplet.tolist()
            print(h,t,r)
            h_name = entity_vocab[h]
            t_name = entity_vocab[t]
            r_name = relation_vocab[r % num_relation]
            print(h_name, t_name, r_name)
            h, t, r = triplet.tolist()
            
            triplet = torch.as_tensor([[h, t, r]], device=solver.device)
            inverse = torch.as_tensor([[t, h, r + num_relation]], device=solver.device)
            solver.model.eval()
            pred, (mask, target) = solver.model.predict_and_target(triplet)
    
            print("De la columna 5 a 10 [5:10]")
    
            print("pred[5:10]", pred[5:10])
            print("mask", mask)
            print("target", target)

    # Evaluate a couple of 1p queries
    if True:
        triplet = solver.test_set[1]
        # Query is 6651 637 13
        h, t, r = triplet.tolist()
        triplet = torch.as_tensor([[h, t, r]], device=solver.device)
        inverse = torch.as_tensor([[t, h, r + num_relation]], device=solver.device)

        h_name = entity_vocab[h]
        t_name = entity_vocab[t]
        r_name = relation_vocab[r % num_relation]
        
        print("h t r", h,t,r)
        print("h t r", h_name, t_name, r_name)
        
        solver.model.eval()
        pred, (mask, target) = solver.model.predict_and_target(triplet)

        print("pred[0][0]", pred[0][0])
        print("pred[0][1]", pred[0][1])

        print("Positions where mask is False")
        false_indices = torch.nonzero(~mask[0][0]).squeeze()
        print(false_indices)

        print("Positions where mask-1 is False")
        false_indices = torch.nonzero(~mask[0][1]).squeeze()
        print(false_indices)

        print("Easy answers")
        for i in [160,259,1800,2370,2736,3730,5341,7940,8644]:
            print("Preds 1")
            print(i, pred[0][0][i])
            print("Mask 1")
            print(mask[0][0][i])
            
        print("Hard answers")
        for i in [3232]:
            print("Preds")
            print(i, pred[0][0][i])
            print("Mask")
            print(mask[0][0][i])

    ## check relations numbers
    if False:
        for r in range(10):
            r_name = relation_vocab[r % num_relation]
            print("r, r_name")
            print(r, r_name)

    if False:
        print("Num relation", num_relation)
        for i in range(10):
            triplet = solver.test_set[i]
            h, t, r = triplet.tolist()
            print(h,t,r)


    

        

    
      
