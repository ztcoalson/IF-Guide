"""
Utilities for training models with HuggingFace Transformers.
Contains the custom trainer for implementing our penalty-based training objective.
"""
import torch

from transformers import Trainer, TrainerCallback


class LogToFileCallback(TrainerCallback):
    """
    Callback to log training progress to a file.
    """
    def __init__(self, log_file=None):
        self.log_file = log_file

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.log_file, "a") as f:
                f.write(f"{str(logs)}\n")


class SuppressToxicTokensTrainer(Trainer):
    """
    Trainer class for suppressing toxic tokens during training.
    """
    def __init__(self, *args, **kwargs):
        self.toxic_lambda = kwargs.pop("toxic_lambda", 1.0)
        super().__init__(*args, **kwargs)

    # Credit: https://github.com/huggingface/transformers/blob/v4.50.0/src/transformers/trainer.py#L3768
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        try:
            toxic_token_mask = inputs.pop("toxic_token_mask")
            labels = inputs.pop("labels")
        except KeyError:
            raise ValueError("The inputs must contain 'toxic_token_mask' and 'labels' keys.")
        
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.loss_fn(outputs, labels, toxic_token_mask, num_items_in_batch)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss
    

    def loss_fn(self, outputs, labels, toxic_token_mask, num_items_in_batch=None):
        logits = outputs.logits

        # Shift to align predictions with labels
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        shifted_mask = toxic_token_mask[..., 1:].contiguous()
        
        # Calculate loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        loss = loss_fct(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1)
        )
        # Reshape loss back to token shape
        loss = loss.view(shifted_labels.size())
        
        # Apply penalty
        weights = torch.where(shifted_mask == True, -self.toxic_lambda, 1.0)
        loss = loss * weights

        # Normalize by number of items in batch if provided
        if num_items_in_batch is not None:
            loss = loss.sum() / num_items_in_batch
        else:
            loss = loss.mean()
        
        return loss