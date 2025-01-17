<div align="center">

<h1>ReNeg: Learning Negative Embedding with Reward Guidance</h1>

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2406.11838-b31b1b.svg)](https://arxiv.org/abs/2412.19637)&nbsp;
<!-- <p align="center">
   📃 <a href="https://arxiv.org/abs/2304.05977" target="_blank">Paper</a> <br>
</p> -->
<br><br><image src="assets/teaser.png"/>
</div>

We present **ReNeg**, a **Re**ward-guided approach that directly learns **Neg**ative embeddings through gradient descent. The global negative embeddings learned using **ReNeg** exhibit strong generalization capabilities and can be seamlessly adaptable to text-to-image and even text-to-video models. Strikingly simple yet highly effective, **ReNeg** amplifies the visual appeal of outputs from base Stable Diffusion models.


If you find `ReNeg`'s open-source effort useful, please 🌟 us to encourage our following development!
## 🔧 Installation
```bash
conda create -n reneg python=3.8.5
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Clone the ReNeg repository 
git clone https://github.com/XiaominLi1997/ReNeg.git
cd ReNeg
pip install -r requirements.txt
```
## 🗄️ Models and Demos
Any text-conditioned generative model utilizing the same text encoder can share their negative embeddings. We provide the following ReNeg embeddings of common text encoders.
### 1. Models
<table>
  <tr>
    <td style="text-align:center;"><b>Text Encoder</b></td>
    <td style="widd: 150px; text-align: center;"><b>Model</b></td>
    <td style="width: 300px; text-align: center;"><b>Path</b></td>
  </tr>
  <tr>
    <td style="text-align: center;" rowspan="2">CLIP ViT-L/14</td>
    <td style="text-align: center;">SD1.4</td>
    <td style="text-align: center;"><a href="checkpoints/sd1.4_reneg_emb.bin">sd1.4_reneg_emb</a></td>
  </tr>
  <tr>
    <!-- <td style="text-align: center;">CLIP ViT-L/14</td> -->
    <td style="text-align: center;">SD1.5</td>
    <td style="text-align: center;"><a href="checkpoints/sd1.5_reneg_emb.bin">sd1.5_reneg_emb</a></td>
  </tr>
  <tr>
    <td style="text-align: center;">OpenCLIP-ViT/H</td>
    <td style="text-align: center;">SD2.1</td>
    <td style="text-align: center;"><a href="checkpoints/sd2.1_reneg_emb.bin">sd2.1_reneg_emb</a></td>
  </tr>
  <tr>
    <td style="text-align:center;">T5-v1.1-xxl</td>
    <td style="text-align:center;">Pixart-alpha</td>
    <td style="text-align: center;"><a href="checkpoints/pixart-alpha_reneg_emb.bin">pixart-alpha_reneg_emb</a></td>
  </tr>
</table>

</table>

### 2. Demos

#### Negative Embeddings of SD and Pixart-alpha:

<table class="center">
<tr>
  <td style="text-align:center;"><b>Text Encoder</b></td>
  <td style="text-align:center;"><b>Model</b></td>
  <td style="text-align:center;"><b>Results</b></td>
  <!-- <td style="text-align:center;" colspan="3"><b>Pos. Prompt + Neg. Prompt</b></td>
  <td style="text-align:center;" colspan="3"><b>Pos. Prompt + Our Neg. Emb.</b></td> -->
</tr>
<tr>
  <td style="text-align:center;" rowspan="2">CLIP ViT-L/14</td>
  <td style="text-align:center;">SD1.4</td>
  <td><img src=assets/sd1.4.png></td>
</tr>
<!-- <tr>
  <td style="text-align:center;" colspan="3">Pos. Prompt: A close-up portrait of a beautiful girl with an autumn leaves headdress and melting wax.</td>  -->

</tr>
<tr>
  <!-- <td style="text-align:center;">CLIP ViT-L/14</td> -->
  <td style="text-align:center;">SD1.5</td>
  <td><img src=assets/sd1.5.png></td>
</tr>
  <!-- <td width=25% style="text-align:center;">"A Terracotta Warrior is riding a horse through an ancient battlefield."</br> seed: 1455028</td>
  <td width=25% style="text-align:center;">"A Terracotta Warrior is playing golf in front of the Great Wall." </br> seed: 5804477</td>
  <td width=25% style="text-align:center;">"A Terracotta Warrior is walking cross the ancient army captured with a reverse follow cinematic shot." </br> seed: 653658</td> -->
</tr>
<tr>
  <td style="text-align:center;">OpenCLIP-ViT/H</td>
  <td style="text-align:center;">SD2.1</td>
  <td><img src=assets/sd2.1.png></td>
</tr>

<tr>
  <td style="text-align:center;">T5-v1.1-xxl</td>
  <td style="text-align:center;">Pixart-alpha</td>
  <td><img src=assets/pixart.png></td>
</tr>
</table>

#### Transfer of Negative Embeddings:
<table class="center">
<tr>
  <td style="text-align:center;"><b>Text Encoder</b></td>
  <td style="text-align:center;"><b>Transfer of Neg. Emb.</b></td>
  <td style="text-align:center;"><b>Results</b></td>

  <!-- <td style="text-align:center;" colspan="3"><b>Pos. Prompt + Neg. Prompt</b></td>
  <td style="text-align:center;" colspan="3"><b>Pos. Prompt + Our Neg. Emb.</b></td> -->
</tr>
<tr>
  <td rowspan="2" style="text-align:center;">OpenCLIP-ViT/H</td>
  <td style="text-align:center;">SD2.1 -> ZeroScope</td>
  <td><img src=assets/transfer/zeroscope.gif></td>

</tr>
<tr>
  <td style="text-align:center;">SD2.1 -> VideoCrafter2</td>
  <td><img src=assets/transfer/videocrafter2.gif></td>
</tr>
<tr>
  <td style="text-align:center;">T5-v1.1-xxl</td>
  <td style="text-align:center;">Pixart-alpha -> LTX-Video</td>
  <td><img src=assets/transfer/ltx-video.gif></td>
</tr>


<!-- <tr>
  <td><img src=assets/multi_videos_results/reference_videos.gif></td>
  <td><img src=assets/customized_appearance_results/A_Terracotta_Warrior_is_riding_a_bicycle_past_an_ancient_Chinese_palace_166357.gif></td>
  <td><img src=assets/customized_appearance_results/A_Terracotta_Warrior_is_lifting_weights_in_front_of_the_Great_Wall_5635982.gif></td>
  <td><img src=assets/customized_appearance_results/A_Terracotta_Warrior_is_skateboarding_9033688.gif></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">Reference videos for motion customization: "A person is riding a bicycle."</td>
  <td width=25% style="text-align:center;">"A Terracotta Warrior is riding a bicycle past an ancient Chinese palace."</br> seed: 166357.</td>
  <td width=25% style="text-align:center;">"A Terracotta Warrior is lifting weights in front of the Great Wall." </br> seed: 5635982</td>
  <td width=25% style="text-align:center;">"A Terracotta Warrior is skateboarding." </br> seed: 9033688</td>
</tr> -->
</table>

<!-- ## 2. Download Models
<table class="center">
<tr>
  <td style="text-align:center;"><b>Text Encoder</b></td>
  <td style="text-align:center;"><b>Model</b></td>
  <td style="text-align:center;"><b>Path of Negative Embedding</b></td>
  <!-- <td style="text-align:center;" colspan="3"><b>Pos. Prompt + Neg. Prompt</b></td>
  <td style="text-align:center;" colspan="3"><b>Pos. Prompt + Our Neg. Emb.</b></td> 
</tr>
<tr>
  <td style="text-align:center;">CLIP ViT-L/14</td>
  <td>SD1.4</td>
  <td></td>
</tr>
<tr>
  <td style="text-align:center;">CLIP ViT-L/14</td>
  <td>SD1.5</td>
  <td></td>
</tr>
<tr>
  <td style="text-align:center;">OpenCLIP-ViT/H</td>
  <td>SD2.1</td>
  <td></td>
</tr>
<tr>
  <td style="text-align:center;">T5-v1.1-xxl</td>
  <td>Pixart-alpha</td>
  <td></td>
</tr>
</table> -->

## 💻 Inference
You need to first specify the paths for `SD1.5` and `neg_emb`. By default, we place `neg_emb` under the `checkpoints/` directory.
```bash
python inference.py --model_path "your_sd1.5_path" --neg_embeddings_path "checkpoints/checkpoint.bin" --prompt "A girl in a school uniform playing an electric guitar."
```

To compare with the inference results using `neg_emb`, you can perform inference using only positive prompt, or use a specific negative prompt.
+ To perform **inference using only the pos_prompt**, you need to specify `args.prompt_type = only_pos`.
```bash
python inference.py --model_path "your_sd1.5_path" --prompt_type "only_pos" --prompt "A girl in a school uniform playing an electric guitar."
```
+ To perform **inference using pos_prompt + neg_prompt**, example negative prompts include: `distorted, ugly, blurry, low resolution, low quality, bad, deformed, disgusting, Overexposed, Simple background, Plain background, Grainy, Underexposed, too dark, too bright, too low contrast, too high contrast, Broken, Macabre, artifacts, oversaturated`
```bash
python inference.py --model_path "your_sd1.5_path" --prompt_type "neg_prompt" --prompt "A girl in a school uniform playing an electric guitar."
```

## 📋 Todo List
- [x] Inference code
- [ ] Training code
- [ ] Online Demo

## ❤️ Acknowledgements
This project is based on [ImageReward](https://github.com/THUDM/ImageReward) and [diffusers](https://github.com/huggingface/diffusers). Thanks for their awesome works.


## Citation

```
@misc{li2024reneg,
      title={ReNeg: Learning Negative Embedding with Reward Guidance},
      author={Xiaomin Li, Yixuan Liu, Takashi Isobe, Xu Jia, Qinpeng Cui, Dong Zhou, Dong Li, You He, Huchuan Lu, Zhongdao Wang, Emad Barsoum},
      year={2024},
      eprint={2412.19637},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
