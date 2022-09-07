# Prompt Extractor v0.3

## Synopsys

This is a standalone, self-hosted, fast tool designed to help an aspiring hacker with reverse-engineering generative art found on the Net.
Specifically, the code uses two deep neural networks - BLIP captioning NN and CLIP image-text embedder NN to estimate the textual prompt.
The resulting caption can be used with generative text-to-image system like Stable Diffusion to generate a similar image.
No GPU is necessary - the tool runs pretty fast on old CPU on a hacker's laptop. Embeddings are cached for blazingly fast execution.

## Usage

Default parameters are geared towards quick reverse-engineering of stable diffusion prompts, thus CLIP/ViTL14 and BLIP-base are used.
```
python3 extract_prompt.py test_images/mecha_cyberpunk.jpg
```

```
chmod +x extract_prompt.py
./extract_prompt.py test_images/mecha_cyberpunk.jpg
```

BLIP-large can be used for slightly enhanced quality of the caption
```
python3 extract_prompt.py --caption_model=large test_images/mecha_cyberpunk.jpg
```

## Honorable mentions

Inspired by:
* @pharmapsychotic's prompt interrogator proof-of-concept colab https://github.com/pharmapsychotic/clip-interrogator
* and its extended fork by @WonkyGrub https://github.com/WonkyGrub/clip-interrogator-customdata

## TODO
* folders and image caching
* self-hosted web ui
* ?
