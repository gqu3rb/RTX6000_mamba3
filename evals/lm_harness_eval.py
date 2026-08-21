import torch

import transformers
from transformers import AutoTokenizer

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
from lm_eval.__main__ import cli_evaluate


@register_model("mamba")
class MambaEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained="state-spaces/mamba-2.8b", max_length=2048, batch_size=None, device="cuda",
                 dtype=torch.float16, tokenizer=None, add_bos_token=False, logits_cache=True,
                 truncation=False):
        LM.__init__(self)
        # The open weights of mamba3 of the huggingface is composed of bfloat16 values
        # so this file is called by
        """
        python lm_harness_eval.py ... --dtype=bfloat16
        """
        # which passes the "bfloat16" as a string, but we need to convert it to a torch object
        # befor passing it to MambaLMHeadModel.from_pretrained
        dtype = get_dtype(dtype)
        self._model = MambaLMHeadModel.from_pretrained(pretrained, device=device, dtype=dtype)
        # when mamba/mamba2 is evaluated, set `tokenizer=None`, so the default
        # tokenizer "EleutherAI/gpt-neox-20b" is chosen
        tokenizer_name = tokenizer or "EleutherAI/gpt-neox-20b"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)
        # self.add_bos_token is used in 
        # ~/.conda/envs/mamba3/lib/python3.10/site-packages/lm_eval/models/huggingface.py
        # if self.add_bos_token doesn't exist, an AttributeError will be raised
        self.add_bos_token = add_bos_token
        # ~/.conda/envs/mamba3/lib/python3.10/site-packages/lm_eval/models/huggingface.py:908
        # will produce AttributeError if self.logits_cache doesn't exist
        self.logits_cache = logits_cache
        # self.truncation is read in
        # ~/.conda/envs/mamba3/lib/python3.10/site-packages/lm_eval/models/huggingface.py:1187
        # but is never used for running 7 benchmarks of Mamba3
        # It is used for free text generation. Leave it here is just a defensive action.
        self.truncation = truncation

        # lm_eval 0.4.3 reads these four attributes;
        # without them the run crashes with AttributeError after the evaluation has already
        # finished, losing the results.
        # This modifications are back compatible with 0.4.2, which never reads them.
        self.pretrained = pretrained
        self.revision = "main"
        self.peft = None
        self.delta = None

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError()


if __name__ == "__main__":
    cli_evaluate()
