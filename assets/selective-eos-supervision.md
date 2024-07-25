# Selective EOS Supervision

## Code Implementation

We make the following modification to the original LLaVA code for Selective EOS Supervision:

In `./LLaVA/llava/model/language_model/llava_llama.py`, the original corresponding code in the `forward` function is:

```python
outputs = super().forward(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_values=past_key_values,
    inputs_embeds=inputs_embeds,
    labels=labels,
    use_cache=use_cache,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    return_dict=return_dict
)

return outputs
```

We calculate a new loss before return `outputs`, and replace `outputs.loss` with the new loss:

```
# Selective EOS Supervision
logits = outputs.logits
shift_logits = logits[..., :-1, :].contiguous() # [batch_size, seq_len, vocab_size]
shift_labels = labels[..., 1:].contiguous() # [batch_size, seq_len]
valid_pos = (shift_labels != -100).view(-1)
valid_logits = shift_logits.view(-1, self.vocab_size)[valid_pos]
valid_labels = shift_labels.view(-1)[valid_pos]

non_eos_idx = torch.ones_like(valid_logits)
# for all positions, set the mask on the EOS token to 0
non_eos_idx[:, 2] = 0
# for positions where the label is EOS, set the maks on the EOS token to 1
final_idx = non_eos_idx.scatter_(1, valid_labels.unsqueeze(-1), True)

selected_logits = valid_logits.masked_fill(final_idx==0, -float('inf'))
loss = F.cross_entropy(selected_logits, valid_labels, reduction='mean')
outputs.loss = loss

return outputs
```