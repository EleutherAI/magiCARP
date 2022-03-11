from carp.pytorch.data import get_datapipeline_names
from carp.pytorch.model.architectures import get_architecture_names
from carp.pytorch.model.encoders import get_encoder_names
from carp.pytorch.scalability_utils import print_rank_0
from carp.pytorch.training.trainer import get_trainer_names


def print_available_configs(args):
    if args.get_architectures:
        print_rank_0("FORMAT: Architecture")
        print_rank_0("Available architectures are:")
        print_rank_0("***************")
        print_rank_0("\n".join(get_architecture_names()))
    elif args.get_encoders:
        print_rank_0("FORMAT: Encoder")
        print_rank_0("Available encoders are:")
        print_rank_0("***************")
        print_rank_0("\n".join(get_encoder_names()))
    elif args.get_datapipelines:
        print_rank_0("FORMAT: DataPipeline")
        print_rank_0("Available data pipelines are:")
        print_rank_0("***************")
        print_rank_0("\n".join(get_datapipeline_names()))
    elif args.get_trainers:
        print_rank_0("FORMAT: Trainer")
        print_rank_0("Available trainers are:")
        print_rank_0("***************")
        print_rank_0("\n".join(get_trainer_names()))
    else:
        return False
    return True
    
# Get parameter groups for model, want to split so that
# biases dont get weight decay but weights do
# via: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
def make_param_groups(model, weight_decay):
    decay = set()
    no_decay = set()

    # Find all linear layers and put their weights in decay
    for mn, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn

                if pn.endswith('weight'):
                    decay.add(fpn)

    # Add  everything to no decay
    for pn, p in model.named_parameters():
        no_decay.add(pn)

    # Remove the exceptions
    no_decay = no_decay - decay

    # assertions to make sure all params have been accounted for
    param_dict = {pn : p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay

    assert len(inter_params) == 0 # shouldn't be any in both
    assert len(param_dict.keys() - union_params) == 0 # shouldn't be any in neither

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}
    ]

    return optim_groups