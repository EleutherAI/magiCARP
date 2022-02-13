import csv
from datasets import load_from_disk
from typing import List, Tuple

# import util


def get_dataset(val_size: int, use_bucket: bool = False, dupe_protection: bool = True) -> Tuple[List[str], List[str]]:
    if use_bucket:
        # dataset = util.load_dataset_from_bucket()
        # TODO: Figure out where util.load_dataset_from_bucket is defined
        raise NotImplementedError(
            "Function `load_dataset_from_bucket` got lost in the shuffle. Load your data locally for now."
        )
    else:
        dataset = load_from_disk("../../../dataset")
    train = dataset["train"]
    passages = train["story_target"]
    reviews = train["target_comment"]

    if dupe_protection:
        size = len(passages)
        orig_size = size
        i = 0
        while i < size:
            if len(reviews[i]) <= 7 or len(passages[i]) <= 7:
                del passages[i]
                del reviews[i]
                size -= 1
            else:
                i += 1
        print(
            "Duplicate protection purged "
            + str(orig_size - size)
            + " samples from original "
            + str(orig_size)
            + " samples ("
            + str(100 - 100 * round(size / orig_size, 4))
            + "% purged)"
        )
    res = list(zip(passages, reviews))
    return res[:-val_size], res[-val_size:]


def get_toy_dataset(val_size: int):
    passages = ["a b c d e f g" for _ in range(2048)]
    reviews = ["h i j k l m n o" for _ in range(2048)]
    res = list(zip(passages, reviews))
    return res[:-val_size], res[-val_size:]


def write_dataset_csv(data, filepath):
    with open(filepath, mode='w') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(data)


if __name__ == "__main__":
    train_set, val_set = get_dataset(100,dupe_protection=False)
    train_set = list(map(lambda x: list(x), train_set))

    train_stories = list(map(lambda x: ["", list(x)[0]], train_set))
    train_crits = list(map(lambda x: ["", list(x)[0]], train_set))

    val_stories = list(map(lambda x: ["", list(x)[0]], val_set))
    val_crits = list(map(lambda x: ["", list(x)[1]], val_set))

    write_dataset_csv(train_stories, 'train_stories.csv')
    write_dataset_csv(train_crits, 'train_crits.csv')

    write_dataset_csv(val_stories, 'val_stories.csv')
    write_dataset_csv(val_crits, 'val_crits.csv')

