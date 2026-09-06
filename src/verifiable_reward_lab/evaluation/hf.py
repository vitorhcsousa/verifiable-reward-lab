"""
The transformers-backed generator: the only file that knows about weights.

Kept apart from the eval loop so that loop stays testable without a model,
and so the day the GRPO policy generates its own rollouts it can satisfy the
same protocol without inheriting any of this.

`AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained`
are typed as unions wide enough to include None, and since transformers 4.50
`generate` no longer lives on `PreTrainedModel`. Rather than chase those
names - they change between releases and would be a silent break on the next
upgrade - the two objects are cast to protocols that declare exactly the
slice of the library this file uses. The type checker then checks the code
against what the code actually needs, and the protocols double as a written
list of the dependency surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verifiable_reward_lab.evaluation.gsm8k_eval import Generation
from verifiable_reward_lab.evaluation.prompt import STOP

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from verifiable_reward_lab.evaluation.config import EvalConfig


class _Tokenizer(Protocol):
    """What this file needs from a tokenizer."""

    padding_side: str
    pad_token: Any
    eos_token: Any
    pad_token_id: int | None

    def __call__(
        self, text: list[str], *, return_tensors: str, padding: bool
    ) -> Mapping[str, Tensor]: ...

    def batch_decode(
        self, sequences: Tensor, *, skip_special_tokens: bool
    ) -> list[str]: ...


class _CausalLM(Protocol):
    """What this file needs from a checkpoint."""

    config: Any

    def to(self, device: torch.device) -> object: ...

    def eval(self) -> object: ...

    def generate(self, **kwargs: object) -> Tensor: ...


class HFGenerator:
    """Greedy decoding from a pinned Hugging Face checkpoint."""

    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        self.tokenizer = cast(
            "_Tokenizer",
            AutoTokenizer.from_pretrained(
                config.model_name, revision=config.model_revision
            ),
        )
        # prompts differ in length and generation continues from the right
        # edge, so padding has to go on the left or the model continues from
        # padding instead of from the question
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # resolved once, and loudly: batching needs a real pad id, and a
        # None reaching the comparison below would silently count every
        # token as new instead of failing
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            msg = f"{config.model_name} has no pad token and no eos token to borrow"
            raise ValueError(msg)
        self.pad_id = pad_id

        self.model = cast(
            "_CausalLM",
            AutoModelForCausalLM.from_pretrained(
                config.model_name,
                revision=config.model_revision,
                dtype=getattr(torch, config.dtype),
            ),
        )
        self.model.to(self.device)
        self.model.eval()

    @property
    def resolved_revision(self) -> str:
        """The commit the hub actually served, for summary.json.

        `main` moves. Reading back what was loaded is what lets the protocol
        pin a sha it has seen rather than one it hopes for.
        """
        served = getattr(self.model.config, "_commit_hash", None)
        return str(served) if served else self.config.model_revision

    def generate(self, prompts: Sequence[str]) -> list[Generation]:
        """Complete each prompt greedily, in order."""
        encoded = self.tokenizer(list(prompts), return_tensors="pt", padding=True)
        batch = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            out = self.model.generate(
                **batch,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.pad_id,
                # stop where truncate_completion would have cut anyway. the
                # verdicts are identical; what changes is that a base model
                # no longer spends 150 tokens per example continuing the
                # pattern past its own answer.
                stop_strings=[STOP],
                tokenizer=self.tokenizer,
            )

        prompt_len = batch["input_ids"].shape[1]
        new_ids = out[:, prompt_len:]
        texts = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)

        # pad and eos are the same id on this tokenizer, so this undercounts
        # each sequence by its eos. it is a throughput figure, not a claim.
        lengths = new_ids.ne(self.pad_id).sum(dim=1).tolist()

        return [
            Generation(text=text, n_new_tokens=int(n))
            for text, n in zip(texts, lengths, strict=True)
        ]
