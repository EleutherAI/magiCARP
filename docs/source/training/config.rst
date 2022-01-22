Configuring CARP
===========================
Refer to the base_config.yml file location in magicarp/configs as an example config file. Config arguments are divided into two categories:
model config and train config.

**Model Config**

* latent_dim | *int* | specifies latent dimensionality for encodings
* proj_dropout | *float* | dropout probability if linear projection is disabled
* linear_projection | *boolean* | if true, language model embedding is linearly projected to encoding space, if false, a feed forward network with dropout is used for this projection
* model_path | *str* | huggingface path to model being used for language embedding
* model_arch | *str* | the type of model (determines how embedding is extracted) (currently supported: `"roberta" <https://huggingface.co/roberta-base>`_, `"neo" <https://huggingface.co/EleutherAI/gpt-neo-2.7B>`_, `"declutr" <https://huggingface.co/johngiorgi/declutr-base>`_)
* encoder_type | *str* | the type of encoder used to generate embedding from language model (see `Encoders <../notes/encoders>`_)
* momentum | *float* | 
* device | *str* | device to run model on 
* grad_accum | *int* | 
* model_eps | *float* | small value to add to logits in contrastive loss before performing cross entropy (intended to circumvent NaNs caused by rounding issues)
* grad_clip | *float* | value to clip gradient norms to during training
  
**Training Config**

* n_ctx | *int* | maximum context length for model inputs (anything longer is truncated)
* epochs | *int* | total number of passes over dataset desired during training
* batch_size | *int* | size of contrastive batches during training
* microbatch_size | *int* | contrastive batches (generally quite large) are split into smaller microbatches of this size for gradient accumulation
* lr_ramp_steps | *int* | number of steps for learning rate to reach learning_rate_init
* lr_decay_steps | *int* | number of steps for learning rate to decay from learning_rate_init to learning_rate_target
* learning_rate_init | *float* | maximum learning rate for scheduler to ramp up to 
* learning_rate_target | *float* | the value learning rate will decay to after ramp up
* do_log | *boolean* | set true to log training to wandb, false for no logging 
* log_interval | *int* | how often to log training results to wandb and terminal
* checkpoint_interval | *int* | how often to save checkpoints of the model
* validate_interval | *int* | how often to run model on validation set
* eval_selection | *string* | how the validation set is chosen
* data_pipeline | *string* | the data pipeline to use for training (see `Preparing The Dataset <dataset>`_)
* orchestrator | *string* |
* dupe_protection | *boolean* | if true, prunes any data points that have less than 8 characters
* hard_dupe_protection | *boolean* | if true, manually checks all batches for duplicate data points and skips batches containing duplicates
* validation_size | *int* | size of validation set
* use_half | *bool* | use half precision tensors in training (training uses amp by default, which is prefered since complete half precision is very unstable)
* use_bucket | *bool* | if true, uses a google cloud storage bucket for model checkpoints and dataset
* opt_eps | *float* | small value used in optimizer to prevent rounding errors