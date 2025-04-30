```shell
pip install git+https://github.com/creative-graphic-design/attend-and-excite-sdxl
```

```python
import torch

from attend_and_excite_sdxl import StableDiffusionXLAttendAndExcitePipeline

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = StableDiffusionXLAttendAndExcitePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)
pipe = pipe.to(device)

prompt = "A cat and a dog reading in the library"
token_indices = [2, 5, 9]

indices_tokens = pipe.get_indices(prompt)
print(f"{indices_tokens=}")

output = pipe(
    prompt=prompt,
    token_indices=token_indices,
    max_iter_to_alter=25,
    guidance_scale=7.5,
    generator=torch.manual_seed(42),
)

image = output.images[0]
image.save("generated.png")
```
