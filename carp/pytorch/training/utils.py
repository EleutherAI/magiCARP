from carp.pytorch.data import get_datapipeline_names
from carp.pytorch.model.architectures import get_architecture_names
from carp.pytorch.model.encoders import get_encoder_names
from carp.pytorch.scalability_utils import print_rank_0
from carp.pytorch.training import get_orchestrator_names


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
    elif args.get_orchestrators:
        print_rank_0("FORMAT: Orchestrator")
        print_rank_0("Available orchestrators are:")
        print_rank_0("***************")
        print_rank_0("\n".join(get_orchestrator_names()))
    else:
        return False
    return True
