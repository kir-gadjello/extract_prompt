# Prompt Extractor v0.4

## Synopsys

This is a standalone, self-hosted, fast tool designed to help an aspiring hacker with reverse-engineering generative art found on the Net.
Specifically, the code uses two deep neural networks - BLIP captioning NN and CLIP image-text embedder NN to estimate the textual prompt.
The resulting caption can be used with generative text-to-image system like Stable Diffusion to generate a similar image.
No GPU is necessary - the tool runs pretty fast on old CPU on a hacker's laptop. Embeddings are cached for blazingly fast execution.

## Installation
* Clone the repository

    ```git clone https://github.com/kir-gadjello/extract_prompt; cd extract_prompt;```

    or download and extract the latest release zip

* If you have a decently recent debian-like Linux distro (for example, Ubuntu), you don't have to use pip, just install the packages from apt:

    ```sudo apt install python3-requests python3-tqdm python3-tabulate python3-pil python3-torch python3-torchvision```

* Otherwise, use pip:

    ```pip3 install -r requirements.txt```

## Usage

Default parameters are geared towards quick reverse-engineering of Stable Diffusion prompts, thus CLIP/ViTL14 and BLIP-base are used.
Still, during the first run the tool will have to download BLIP captioning model (896Mb) and CLIP text-image model (932Mb for ViTL14).
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

## Notes

The release contains pre-computed text embedding vectors for ViTL14 and the default provided set of prompt templates, to facilitate immediate tool usage with generative art derived from Stable Diffusion.
If you have concerns about PyTorch's pickle tensor storage format, you can safely remove `./cache/embs` directory. This action will incur one-time embedding recomputation on your machine, taking approximately 10 minutes if you run the system on a CPU.

You can add more specific prompt templates to `./templates/` directory, this incurs only a small one-time penalty of recomputing a few chunks of embedding vectors.

## Honorable mentions

Inspired by:
* @pharmapsychotic's prompt interrogator proof-of-concept colab https://github.com/pharmapsychotic/clip-interrogator
* and its extended fork by @WonkyGrub https://github.com/WonkyGrub/clip-interrogator-customdata

## TODO
* batch folder handling, image caching
* html-formatted image batch report
* self-hosted web ui
* ?
