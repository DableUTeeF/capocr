from transformers import GPT2LMHeadModel, GPT2Config, AutoModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, functional as F
import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class DistillTrainGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, istrain, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.istrain = istrain
        if istrain:
            self.teacher = GPT2LMHeadModel.from_pretrained('ai-forever/mGPT')

    def forward(
        self,
        input_ids,
        past_key_values,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        inputs_embeds,
        encoder_hidden_states,
        encoder_attention_mask,
        labels,
        use_cache,
        output_attentions,
        output_hidden_states,
        return_dict,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            bce = BCEWithLogitsLoss()
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            if self.istrain:
                teacher_output = self.teacher(
                    input_ids,
                    past_key_values,
                    attention_mask,
                    token_type_ids,
                    position_ids,
                    head_mask,
                    inputs_embeds,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    labels,
                    use_cache,
                    output_attentions,
                    output_hidden_states,
                    return_dict,
                )
                shift_logits_teacher = teacher_output.logits[..., :-1, :].contiguous()
                loss_KD = bce(shift_logits.view(-1, shift_logits.size(-1)), shift_logits_teacher.view(-1, shift_logits_teacher.size(-1)))
                loss = loss + loss_KD

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

