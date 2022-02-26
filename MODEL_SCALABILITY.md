# Document for model scalability
- author: [Kevin Ko](https://github.com/hyunwoongko)

## 1. Activation Checkpointing
I added activation checkpointing to save activation memory. 
You can use this feature by specifying `gradient_checkpointing` in the `train_job`.

Example:

```
gradient_checkpointing: true
```

## 2. DeepSpeed Training

Run the following instruction.

```
python -m torch.distributed.launch \
       --nproc_per_node NUM_YOUR_GPUS \
       -m carp.pytorch.training.train \
       --data_path="carp/dataset" \
       --config_path ./configs/YOUR_CONFIG_NAME.yml \
       --deepspeed_config ./configs/scalability/YOUR_DS_CONFIG_NAME.json
```

## 3. Integrated architectures
- `carp.py`
- `carp_cloob.py`
- `carp_coop.py`
- `carp_shared_encoder.py`

`carp_mlm.py` and `carp_momentum.py` are not integrated yet.


## 4. Note
### 4.1. Disabled options
In order to preserve the overall structure of the current codebase, 
I have disabled several deepspeed options. 
I used logic in our codebase for the following features, not deepspeed version.

- `gradient_accumulation_steps`
- `train_micro_batch_size_per_gpu`
- `scheduler`

### 4.2. Necessary options
You must specify `optimizer` in your deepspeed config.

### 4.3. The "auto" values
You can specify "auto" in your deepspeed config for the following options.

- `train_batch_size`
- `lr` (optimizer.params)
- `eps` (optimizer.params)
- `weight_decay` (optimizer.params)

Example:
```
{
    "train_batch_size": "auto",
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    }
}
```

