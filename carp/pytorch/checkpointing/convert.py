from argparse import ArgumentParser

from carp.pytorch.checkpointing import *


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--origin_path", type=str, required=False)
    parser.add_argument("--destination_path", type=str, required=False)
    parser.add_argument("--origin_type", type=str, required=False)
    parser.add_argument("--destination_type", type=str, required=False)
    parser.add_argument("--get_converters", action="store_true")
    return parser


if __name__ == "__main__":
    parser = get_arguments()
    args, _ = parser.parse_known_args()

    # do not actually run the conversion, just return the list of converters
    if args.get_converters:
        print("FORMAT: Origin, Desination")
        print("Available converters are:")
        print("***************")
        print("\n".join(get_converter_names()))
    else:
        origin_type = args.origin_type.lower()
        destination_type = args.destination_type.lower()

        # retrieve, initialize, and execute converter
        # TODO: Handle kwargs!!!
        converter = get_converter(origin_type, destination_type)()
        converter.convert(args.origin_path, args.destination_path)
        print("Conversion successful!")
