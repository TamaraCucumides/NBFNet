import os
import sys
import pprint
import time

import torch

from torchdrug import core
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import dataset, layer, model, task, util

import pickle


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
    print("Triplet", triplet)
    num_relation = len(relation_vocab)
    print("num relation", num_relation)
    h, t, r = triplet.tolist()
    print("h,t,r", h,t,r)
    triplet = torch.as_tensor([[h, t, r]], device=solver.device)
    inverse = torch.as_tensor([[t, h, r + num_relation]], device=solver.device)
    start_time = time.time()
    solver.model.eval()
    pred, (mask, target) = solver.model.predict_and_target(triplet)
    end_time = time.time()
    print("Time to predict", end_time - start_time)

    print("pred", pred)
    print("mask", mask)
    print("target", target)
    
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    rankings = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    rankings = rankings.squeeze(0)

    print(rankings)
    return 

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

    print(cfg)

    if False:
        
        working_dir = util.create_working_directory(cfg)
    
        torch.manual_seed(args.seed + comm.get_rank())
    
        logger = util.get_root_logger()
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    
        #if cfg.dataset["class"] != "FB15k237":
        #    raise ValueError("Visualization is only implemented for FB15k237")
    
        dataset = core.Configurable.load_config_dict(cfg.dataset)
        #solver = util.build_solver(cfg, dataset)
    
        entity_vocab, relation_vocab = load_vocab(dataset)
    
        folder = "vocab"
    
        if not os.path.exists(folder):
          os.makedirs(folder)
    
        # File path to save the pickle file
        file_path = os.path.join(folder, 'entity_vocab.pickle')
    
        with open(file_path, 'wb') as file:
          pickle.dump(entity_vocab, file)
    
        file_path = os.path.join(folder, 'relation_vocab.pickle')
    
        with open(file_path, 'wb') as file:
          pickle.dump(relation_vocab, file)

    
