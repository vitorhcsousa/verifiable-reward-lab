import statistics
import time

import torch
from torch import Tensor

from rlvr_from_scratch.model.transformer import DecoderTransformer, TransformerConfig

SEED = 0
PROMPT_LEN = 16
N_RUNS = 5
NEW_TOKENS = (128, 256)


def build() -> tuple[DecoderTransformer, torch.Tensor]:
    torch.manual_seed(SEED)
    cfg = TransformerConfig(
        vocab_size=1000, d_model=256, n_layers=4, n_heads=4, max_seq_len=512
    )

    model = DecoderTransformer(cfg).eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, PROMPT_LEN))

    return model, prompt


def time_once(
    model: DecoderTransformer, prompt: Tensor, n_new: int, use_cache: bool
) -> float:
    t0 = time.perf_counter()
    model.generate(
        idx=prompt, max_new_tokens=n_new, do_sample=False, use_cache=use_cache
    )
    return n_new / (time.perf_counter() - t0)


def main() -> None:
    model, prompt = build()
    print("| n_new | use_cache | median tok/s | min-max |")
    print("|---|---|---|---|")

    for n_new in NEW_TOKENS:
        for use_cache in (False, True):
            time_once(model, prompt, n_new, use_cache)
            rates = [time_once(model, prompt, n_new, use_cache) for _ in range(N_RUNS)]
            med = statistics.median(rates)
            print(
                f"| {n_new} | {use_cache} | {med:.1f} | {min(rates):.1f}-{max(rates):.1f} |"
            )


if __name__ == "__main__":
    main()
