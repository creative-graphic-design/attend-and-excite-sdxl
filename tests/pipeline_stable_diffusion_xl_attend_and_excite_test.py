import pathlib
from typing import List

import pytest
import torch
from diffusers import StableDiffusionXLPipeline

from attend_and_excite_sdxl import StableDiffusionXLAttendAndExcitePipeline


@pytest.fixture
def root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def save_base_dir(root_dir: pathlib.Path) -> pathlib.Path:
    save_dir = root_dir / "generated"
    return save_dir


@pytest.fixture
def sd_model_id() -> str:
    return "stable-diffusion-v1-5/stable-diffusion-v1-5"


@pytest.fixture
def sdxl_model_id() -> str:
    return "stabilityai/stable-diffusion-xl-base-1.0"


@pytest.fixture
def torch_dtype() -> torch.dtype:
    return torch.bfloat16


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def seed() -> int:
    return 19950815


TEST_EXAMPLES = (
    # ("A white car and red sheep", [3, 6], ["car", "sheep"]),
    # ("A Picasso painting in a garden", [3, 6], ["painting", "garden"]),
    ("A cat and a dog reading in the library", [2, 5, 9], ["cat", "dog", "library"]),
)


@pytest.mark.parametrize(
    argnames="prompt, token_indices, tokens",
    argvalues=TEST_EXAMPLES,
)
def test_sdxl_ae_pipeline(
    sdxl_model_id: str,
    torch_dtype: torch.dtype,
    device: torch.device,
    seed: int,
    prompt: str,
    token_indices: List[int],
    tokens: List[str],
    save_base_dir: pathlib.Path,
):
    assert len(token_indices) == len(tokens)

    save_dir = save_base_dir / "sdxl-ae"
    save_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionXLAttendAndExcitePipeline.from_pretrained(
        sdxl_model_id,
        torch_dtype=torch_dtype,
    )
    # pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)

    indices_tokens = pipe.get_indices(prompt)
    print(f"{indices_tokens=}")
    assert [indices_tokens[i] for i in token_indices] == [f"{t}</w>" for t in tokens]

    # assert indices_tokens[token_indices[0]] == "car</w>"
    # assert indices_tokens[token_indices[1]] == "sheep</w>"

    output = pipe(
        prompt=prompt,
        token_indices=token_indices,
        max_iter_to_alter=25,
        guidance_scale=7.5,
        generator=torch.manual_seed(seed),
    )

    image = output.images[0]
    image.save(f"sdxl-ae, {prompt=}.png")


@pytest.mark.parametrize(
    argnames="prompt, token_indices, tokens",
    argvalues=TEST_EXAMPLES,
)
def test_sdxl_pipeline(
    sdxl_model_id: str,
    torch_dtype: torch.dtype,
    device: torch.device,
    seed: int,
    prompt: str,
    token_indices: List[int],
    tokens: List[str],
):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        sdxl_model_id, torch_dtype=torch_dtype
    )
    # pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(
    #     sd_model_id, torch_dtype=torch_dtype
    # )
    pipe = pipe.to(device)

    output = pipe(
        prompt=prompt,
        guidance_scale=7.5,
        generator=torch.manual_seed(seed),
    )

    image = output.images[0]
    image.save(f"sdxl, {prompt=}.png")
