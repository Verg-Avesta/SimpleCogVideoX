# SimpleCogVideoX
 A simple diffusers implementation of CogVideoX 1.5 I2V SFT.

 ## Prepartion
- Download the checkpoints of CogVideoX 1.5 I2V, from [CogVideoX 1.5 I2V](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V/tree/main).

## Requirements
- diffusers (from source)
- transformers>=4.46.2
- accelerate>=1.1.1
- imageio-ffmpeg>=0.5.1
- pip install git+https://github.com/huggingface/diffusers
- pip install --upgrade transformers accelerate diffusers imageio-ffmpeg

## Acknowledgements
Many thanks to the code bases from [diffusers](https://github.com/huggingface/diffusers), [SimpleSDM](https://github.com/haoningwu3639/SimpleSDM) and [finetrainers](https://github.com/a-r-r-o-w/finetrainers), and pre-trained model parameters from [CogVideoX 1.5 I2V](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V).
