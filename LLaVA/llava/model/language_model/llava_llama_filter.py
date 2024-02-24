#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

import torch.nn.functional as F
import wandb

# plot
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import pdb


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # self.eos_res_cd = {}
        # self.eos_res = {}
        # self.eos_pred_log_probs = {}

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained('/datassd2/pretrained_models/llava-v1.5-7b')

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        conv_id: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        with torch.no_grad():
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

            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous() # [batch_size, seq_len, vocab_size]
            shift_labels = labels[..., 1:].contiguous() # [batch_size, seq_len]

            gpu_id = torch.distributed.get_rank()
            save_path = f'data_score_{gpu_id}.txt'
            '''
            IMPORTANT: Dataloader requires modification to include a unique conversation id `conv_id` for each data sample.
            '''
            with open(save_path, 'a') as f:
                for idx in range(shift_labels.size(0)):
                    valid_pos = (shift_labels[idx] != -100)
                    valid_logits = shift_logits[idx][valid_pos]
                    valid_labels = shift_labels[idx][valid_pos]
                    valid_probs = valid_logits.log_softmax(dim=-1).exp()
                    valid_one_minus_probs = torch.clamp(1 - valid_probs, min=1e-5)

                    # S_pos
                    eos_idx = valid_labels == 2
                    eos_probs = valid_probs[eos_idx, 2]
                    s_pos = -torch.log(eos_probs).sum()

                    # S_neg
                    non_eos_idx = valid_labels != 2
                    non_eos_one_minus_eos_probs = valid_one_minus_probs[non_eos_idx, 2]
                    s_neg = -torch.log(non_eos_one_minus_eos_probs).sum()

                    f.write(f'{conv_id[idx]},{s_pos},{s_neg}\n')
                    f.flush()

            outputs.loss *= 0
            return outputs


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
