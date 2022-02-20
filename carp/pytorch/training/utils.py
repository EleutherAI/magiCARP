from carp.pytorch.data import get_datapipeline_names
from carp.pytorch.model.architectures import get_architecture_names
from carp.pytorch.model.encoders import get_encoder_names
from carp.pytorch.training import get_orchestrator_names


def print_available_configs(args):
    if args.get_architectures:
        print("FORMAT: Architecture")
        print("Available architectures are:")
        print("***************")
        print("\n".join(get_architecture_names()))
    elif args.get_encoders:
        print("FORMAT: Encoder")
        print("Available encoders are:")
        print("***************")
        print("\n".join(get_encoder_names()))
    elif args.get_datapipelines:
        print("FORMAT: DataPipeline")
        print("Available data pipelines are:")
        print("***************")
        print("\n".join(get_datapipeline_names()))
    elif args.get_orchestrators:
        print("FORMAT: Orchestrator")
        print("Available orchestrators are:")
        print("***************")
        print("\n".join(get_orchestrator_names()))
    else:
        return False
    return True
