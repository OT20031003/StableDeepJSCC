# Stable Diffusion VAE潜在空間版ADJSCC 実装レポート

## 1. 実装概要

FFHQ画像をStable DiffusionのVAE潜在空間へ写像し、その4チャネル潜在表現をADJSCCで無線伝送する学習・評価プログラムを追加した。

新規実装ファイルは次のとおり。

- `ADJSCC/adjscc_sd_vae_ffhq.py`
- `ADJSCC/adjscc_sd_img2img.py`
- 追加テスト: `ADJSCC/tests/test_pytorch_port.py`
- 追加テスト: `ADJSCC/tests/test_sd_adjscc_img2img.py`

既存の画像空間用 `DeepJSCC` は3チャネル入力かつSigmoid出力であるため使用していない。既存実装から次の部品だけを再利用した。

- `GFRModule`: 畳み込み、GDN/IGDN、PReLU
- `AFModule`: SNR条件付きAttention Featureモジュール
- `Channel`: 複素信号化、電力正規化、AWGN

実行環境はリポジトリ既存の `environment.yaml` で作られた `ldm` 環境をそのまま使用する。追加パッケージ、PyTorchの置換、NumPyの置換は不要である。

## 2. 処理パイプライン

実装した処理系列は次のとおり。

```text
FFHQ画像 x: [B, 3, 512, 512], 値域 [-1, 1]
    ↓ Frozen Stable Diffusion VAE Encoder
z = vae.get_first_stage_encoding(vae.encode_first_stage(x))
z: [B, 4, 64, 64]
    ↓ LatentAttentionEncoder
y: [B, transmit_channel_num, 16, 16]
    ↓ コードワード単位の平均電力正規化
    ↓ 複素AWGNチャネル
y_hat
    ↓ LatentAttentionDecoder
z_hat: [B, 4, 64, 64]
    ↓ Frozen Stable Diffusion VAE Decoder
x_hat: [B, 3, 512, 512]
```

ADJSCCの学習・単独評価時にはStable DiffusionのUNetとCLIPテキストEncoderは使用しない。`models/ldm/stable-diffusion-v1/model.ckpt` からキーが `first_stage_model.` で始まるVAE重みだけを抽出してロードする。後述するStable Diffusion付き推論では、同じcheckpointからUNet・CLIP・VAEを含むfull modelをロードする。

設定ファイルは `configs/stable-diffusion/v1-inference.yaml` を使用し、同設定の `scale_factor=0.18215` を適用する。

## 3. Stable Diffusion VAE

### 3.1 Encoder

教師潜在は要求された処理と同じ順序で生成する。

```python
z = vae.get_first_stage_encoding(vae.encode_first_stage(x)).detach()
```

VAE Encoderは完全にfreezeし、教師潜在生成は `torch.no_grad()` 内で行う。`get_first_stage_encoding()` はStable Diffusionと同じくVAE posteriorからsampleし、0.18215を乗じる。そのため、同じ画像でもepochごとに教師潜在がわずかに変化する。

### 3.2 Decoderと画像損失の勾配

元の `ldm.models.diffusion.ddpm.LatentDiffusion.decode_first_stage()` には `@torch.no_grad()` が付いている。この関数をそのまま画像L1損失に使うと、`x_hat` から `z_hat`、さらにADJSCCへの勾配が遮断される。

そのため本実装では、同じscale処理を行ったうえで凍結済みfirst-stage Decoderを直接呼ぶ。

```python
x_hat = first_stage_model.decode(z_hat / scale_factor)
```

VAEパラメータの `requires_grad` はすべてFalseだが、このDecoder forward自体は `no_grad()` で囲まない。これによりVAE重みは更新されず、`dL/dz_hat` だけが計算される。

チェックポイントを使った実測で次を確認した。

- VAE trainable parameters: 0
- `32×32 → [1,4,4,4] → 32×32` のshape一致
- `x_hat` から `z_hat` への勾配が有限
- 読み込んだStable Diffusion checkpointのglobal step: 470000

## 4. FFHQ Dataset

デフォルト入力ディレクトリは次のとおり。

```text
../datasets/ffhq_train_70k
```

実データを確認した結果、PNG画像が70,000枚あり、各画像は256×256 RGBだった。本実装のデフォルトは合意どおり512×512であるため、中央正方形crop後にLanczosで512×512へresizeする。元画像が256×256なので、512px学習は空間解像度を増やすものではなく、Stable Diffusion VAEの512px入出力を使用するためのupscaleである。

前処理は次のとおり。

1. ディレクトリ以下を再帰検索
2. RGB変換
3. 中央正方形crop
4. 512×512へresize
5. 学習セットのみ確率0.5で左右反転
6. `[0,1]` から `[-1,1]` へ変換

デフォルトでは `split_seed=0` で画像パスを決定論的にshuffleし、69,000枚を学習、1,000枚を検証に使用する。

## 5. 潜在ADJSCCモデル

### 5.1 Encoder

512px画像の場合、VAE潜在 `[B,4,64,64]` を入力する。

| 段 | 入力→出力チャネル | Kernel | Stride | 後処理 |
|---:|---|---:|---:|---|
| 1 | 4→256 | 9 | 2 | GDN、PReLU、AF(SNR) |
| 2 | 256→256 | 5 | 2 | GDN、PReLU、AF(SNR) |
| 3 | 256→256 | 5 | 1 | GDN、PReLU、AF(SNR) |
| 4 | 256→256 | 5 | 1 | GDN、PReLU、AF(SNR) |
| 5 | 256→`transmit_channel_num` | 5 | 1 | GDN |

デフォルト `transmit_channel_num=16` では出力は `[B,16,16,16]` になる。

### 5.2 電力正規化とAWGN

Encoder出力をflattenし、前半を実部、後半を虚部として複素コードワードへ変換する。各サンプルについて平均複素シンボル電力が1になるよう正規化し、指定SNRの複素AWGNを加える。

学習時のSNRは、デフォルトでサンプルごとに `[0,20] dB` の一様分布から生成する。検証時はデフォルト10 dB、独立評価では任意のSNR列を指定できる。

512px、`transmit_channel_num=16` の伝送量は次のとおり。

- VAE潜在scalar数: `4×64×64 = 16,384`
- チャネル実数値数: `16×16×16 = 4,096`
- 複素チャネルシンボル数: 2,048
- 潜在scalar当たり複素シンボル数: 0.125
- 元画像pixel当たり複素シンボル数: 0.0078125

### 5.3 Decoder

DecoderはEncoderの逆方向構成で、最後に4チャネル潜在を出力する。最終段にはIGDNを残すが、Sigmoidは使用しない。したがって `z_hat` は負値を含む非制限の潜在表現になる。

## 6. 損失関数と二段階学習

実装した損失は次のとおり。

```python
latent_loss = mse(z_hat, z)
image_loss = l1(x_hat, x)
loss = latent_loss_weight * latent_loss + image_loss_weight * image_loss
```

推奨する二段階学習は次のとおり。

1. 第1段階: 潜在MSEのみでADJSCCを学習
2. 第2段階: 第1段階のbest checkpointから、潜在MSE＋少量の画像L1でfine-tuning

第2段階では損失の尺度と最適化条件が変わるため、`--reset-optimizer --reset-best` を指定する。epoch番号は第1段階から継続する。

## 7. チェックポイントと出力

出力ディレクトリは次の構造になる。

```text
output_dir/
├── best.pt
├── last.pt
├── history.json
├── samples/
│   └── epoch_XXXX.png
└── evaluation/
    ├── metrics.json
    ├── snr_0dB.png
    ├── snr_5dB.png
    └── ...
```

`.pt` に保存するものは次だけである。

- ADJSCC `state_dict`
- Adam optimizer state
- epoch
- best validation loss
- 実行引数metadata
- checkpoint種別と、途中保存時のbatch数・進捗率

Stable Diffusion VAE、UNet、CLIPの重みはADJSCC checkpointへ重複保存しない。

`samples/*.png` と `evaluation/*.png` は、上段が原画像、下段が再構成画像のグリッドである。

通常の保存タイミングと割合指定保存は次のとおり。

- `last.pt`: 各epochの学習と、該当する場合は検証が完了した後に毎回上書き保存
- `last.pt`（途中保存）: `--save-every-percent N` を指定すると、各epochの約N%、2N%…でも同じファイルへ上書き保存
- `best.pt`: `--val-every` ごとの検証lossが過去最良を更新した場合だけ上書き保存
- `history.json`: 各epoch終了時に追記
- `samples/epoch_XXXX.png`: 検証を実施したepochで保存

`--save-every-percent` は0より大きく100以下の値を受け付け、デフォルトの`0`では従来どおりepoch終了時だけ保存する。例えば`--save-every-percent 10`では、10%、20%…90%付近と、通常のepoch終了時に保存する。100%時点は検証後の通常保存が担当する。途中checkpointを世代別に増やさず`last.pt`へ原子的に上書きするため、保存回数を増やしてもディスク使用量は増え続けない。

gradient accumulation中の未反映勾配はcheckpointに含まれないため、割合閾値を超えた後の最初のOptimizer step直後に保存する。したがって、`--accumulation-steps 8`では指定率より最大7バッチ遅れる場合がある。保存時には実際のbatch数と進捗率を表示する。

途中保存された`last.pt`には、完全に終了したepoch数とは別に、処理中epoch・batch数・進捗率を記録する。`--resume`でこのcheckpointを指定すると処理中だったepochを先頭からやり直す。バッチ位置からの厳密再開ではないが、途中まで学習したADJSCC重みとOptimizer状態は復元され、処理中epochを誤って完了扱いして飛ばすことはない。epoch終了後の`last.pt`からは従来どおり次epochへ進む。

`--log-every 100` は学習・検証・独立評価の進捗を100バッチごとに表示する。学習lossは直近100バッチの平均、検証指標はその時点までの累積値である。経過時間と現在までの平均速度から計算したETAも表示する。最後のバッチは指定間隔と一致しなくても表示され、`--log-every 0` で進捗表示を無効化できる。

## 8. フルコマンド

すべてリポジトリrootから実行する。

### 8.1 環境有効化

```bash
cd /mnt/d/stable-diffusion
conda activate ldm
```

### 8.2 第1段階: 潜在MSEのみ

以下はFFHQ 69,000枚を学習、1,000枚を検証し、256px、SNR -10～20 dBで100 epoch学習するフルコマンドである。`batch-size=1` と `accumulation-steps=8` により実効batch sizeを8にする。

```bash
python ADJSCC/adjscc_sd_vae_ffhq.py train \
  --data-dir ../datasets/ffhq_train_70k \
  --sd-config configs/stable-diffusion/v1-inference.yaml \
  --sd-checkpoint models/ldm/stable-diffusion-v1/model.ckpt \
  --output-dir ADJSCC/outputs/sd_vae_ffhq_stage1_C32 \
  --image-size 256 \
  --val-count 1000 \
  --split-seed 0 \
  --transmit-channel-num 32 \
  --feature-channels 256 \
  --snr-low-train -10 \
  --snr-up-train 20 \
  --snr-val -5 \
  --latent-loss-weight 1.0 \
  --image-loss-weight 0.0 \
  --learning-rate 0.0001 \
  --epochs 100 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --accumulation-steps 8 \
  --grad-clip 1.0 \
  --val-every 1 \
  --log-every 100 \
  --save-every-percent 10 \
  --num-workers 4 \
  --sample-count 4 \
  --device cuda
```

### 8.3 第2段階: 潜在MSE＋画像L1 fine-tuning

第1段階のbest checkpointから50 epoch追加学習する。

```bash
python ADJSCC/adjscc_sd_vae_ffhq.py train \
  --data-dir ../datasets/ffhq_train_70k \
  --sd-config configs/stable-diffusion/v1-inference.yaml \
  --sd-checkpoint models/ldm/stable-diffusion-v1/model.ckpt \
  --output-dir ADJSCC/outputs/sd_vae_ffhq_stage2 \
  --image-size 256 \
  --val-count 1000 \
  --split-seed 0 \
  --transmit-channel-num 16 \
  --feature-channels 256 \
  --snr-low-train -10 \
  --snr-up-train 20 \
  --snr-val -5 \
  --latent-loss-weight 0.5 \
  --image-loss-weight 0.5 \
  --learning-rate 0.00005 \
  --epochs 50 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --accumulation-steps 8 \
  --grad-clip 1.0 \
  --val-every 1 \
  --log-every 100 \
  --save-every-percent 10 \
  --num-workers 4 \
  --sample-count 4 \
  --resume ADJSCC/outputs/sd_vae_ffhq_stage1/best.pt \
  --reset-optimizer \
  --reset-best \
  --device cuda
```

### 8.4 中断した同一段階の学習再開

Optimizerとbest値も復元するため、`--reset-optimizer` と `--reset-best` は付けない。epoch終了checkpointの場合、`--epochs 20` は保存epochからさらに20 epoch実行する意味である。途中保存checkpointの場合は処理中だったepochを先頭から再実行し、そこから20 epoch実行する。

```bash
python ADJSCC/adjscc_sd_vae_ffhq.py train \
  --data-dir ../datasets/ffhq_train_70k \
  --sd-config configs/stable-diffusion/v1-inference.yaml \
  --sd-checkpoint models/ldm/stable-diffusion-v1/model.ckpt \
  --output-dir ADJSCC/outputs/sd_vae_ffhq_stage2 \
  --image-size 256 \
  --val-count 1000 \
  --split-seed 0 \
  --transmit-channel-num 16 \
  --feature-channels 256 \
  --snr-low-train -10 \
  --snr-up-train 20 \
  --snr-val 10 \
  --latent-loss-weight 1.0 \
  --image-loss-weight 0.1 \
  --learning-rate 0.00005 \
  --epochs 20 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --accumulation-steps 8 \
  --grad-clip 1.0 \
  --val-every 1 \
  --log-every 100 \
  --save-every-percent 10 \
  --num-workers 4 \
  --sample-count 4 \
  --resume ADJSCC/outputs/sd_vae_ffhq_stage2/last.pt \
  --device cuda
```

### 8.5 SNRスイープ評価

検証1,000枚についてSNRごとに10回評価し、平均値と再構成グリッドを保存する。

```bash
python ADJSCC/adjscc_sd_vae_ffhq.py eval \
  --data-dir ../datasets/ffhq_train_70k \
  --sd-config configs/stable-diffusion/v1-inference.yaml \
  --sd-checkpoint models/ldm/stable-diffusion-v1/model.ckpt \
  --output-dir ADJSCC/outputs/sd_vae_ffhq_stage2 \
  --checkpoint ADJSCC/outputs/sd_vae_ffhq_stage2/last.pt \
  --image-size 256 \
  --val-count 100 \
  --split-seed 0 \
  --transmit-channel-num 16 \
  --feature-channels 256 \
  --eval-snrs -10 -5 0 5 10 15 20 \
  --eval-repeats 1 \
  --latent-loss-weight 1.0 \
  --image-loss-weight 0.1 \
  --eval-batch-size 4 \
  --log-every 100 \
  --num-workers 4 \
  --sample-count 4 \
  --device cuda
```

評価時の `transmit-channel-num` と `feature-channels` はcheckpoint metadataから自動取得する。コマンドには実験条件を明示する目的で記載している。

### 8.6 CPU用最小スモークテスト

これは動作確認専用であり、品質評価には使用しない。FFHQ各1枚、32px、1 batchだけを処理する。

```bash
python ADJSCC/adjscc_sd_vae_ffhq.py train \
  --data-dir ../datasets/ffhq_train_70k \
  --sd-config configs/stable-diffusion/v1-inference.yaml \
  --sd-checkpoint models/ldm/stable-diffusion-v1/model.ckpt \
  --output-dir /tmp/adjscc_sd_vae_smoke \
  --image-size 32 \
  --val-count 1 \
  --limit-train-samples 1 \
  --limit-val-samples 1 \
  --transmit-channel-num 4 \
  --feature-channels 8 \
  --latent-loss-weight 1.0 \
  --image-loss-weight 0.0 \
  --epochs 1 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --num-workers 0 \
  --sample-count 1 \
  --max-train-batches 1 \
  --max-eval-batches 1 \
  --log-every 1 \
  --save-every-percent 10 \
  --device cpu
```

## 9. 評価指標

`eval` はSNRごとに次を `evaluation/metrics.json` へ保存する。

- `latent_mse`: `z_hat` と `z` の潜在MSE
- `latent_nmse`: 潜在二乗誤差 / 潜在エネルギー
- `image_l1`: VAE再構成画像のL1
- `image_mse`: `[0,1]` へ変換・clamp後の画像MSE
- `image_psnr`: data range 1で計算したPSNR
- `loss`: 指定した重み付き損失

## 10. 実施した検証

`conda run -n ldm` を明示して次を実施した。

1. Python構文検査
2. CLI `--help`
3. 4チャネルLatent ADJSCCのforward/backward
4. 17×19の非4倍潜在に対するshape復元
5. FFHQ 70,000枚の検出と69,000/1,000分割
6. FFHQ画像の指定解像度化と `[-1,1]` 正規化
7. 指定 `model.ckpt` からfirst-stage VAEだけを抽出
8. VAE encode/decode shape確認
9. 凍結VAE Decoderを介した `z_hat` 勾配確認
10. 潜在MSEのみの1 batch学習、検証、best/last保存
11. 第1段階checkpointから画像L1付きfine-tuning
12. 独立evalコマンドによるJSON、PNG出力
13. `--log-every` の指定間隔・最終バッチ進捗表示
14. 割合指定checkpointがOptimizer step直後だけで発火すること
15. Stable Diffusion付き推論のstrength境界、入力前処理、checkpoint metadata読込
16. 実full Stable Diffusion・実ADJSCC checkpointによるGPU推論
17. 全単体テスト15件

単体テスト結果は15件すべて成功した。

CPUスモークテストは32px、学習・検証各1枚なので、出力値は品質を表さない。経路確認時の例は次のとおり。

- 第1段階 train latent MSE: 0.46109772
- 第1段階 validation latent MSE: 0.40463364
- 第2段階 train image L1: 0.45075285
- 10 dB評価 image PSNR: 13.9392 dB

Stable Diffusion付きGPUスモークテストでは、64px入力、SNR 0 dB、`ddim_steps=2`、`strength=0.5` とし、次の全経路を確認した。

```text
実FFHQ画像 → full SD VAE Encoder → 実ADJSCC → AWGN
→ 実ADJSCC Decoder → CLIP conditioning → DDIM UNet 1 step
→ full SD VAE Decoder → PNG・JSON保存
```

## 11. Stable Diffusion付きADJSCC img2img推論

### 11.1 実装ファイルと処理系列

新規推論CLIは次のファイルである。

```text
ADJSCC/adjscc_sd_img2img.py
```

処理系列は次のとおり。

```text
入力画像 x: [1, 3, 256, 256], 値域 [-1, 1]
    ↓ Stable Diffusion VAE Encoder
初期潜在 z: [1, 4, 32, 32]
    ↓ 学習済みLatent ADJSCC Encoder
通信表現 y: [1, 16, 8, 8]
    ↓ 電力正規化
    ↓ AWGN（--snr-db）
受信表現 y_hat
    ↓ 学習済みLatent ADJSCC Decoder
受信潜在 z_hat: [1, 4, 32, 32]
    ↓ DDIM stochastic_encode（--strength）
拡散ノイズ付き潜在 z_t
    ↓ Stable Diffusion UNet + prompt conditioning
精製潜在 z_sd
    ↓ Stable Diffusion VAE Decoder
最終画像 x_sd: [N, 3, 256, 256]
```

学習CLIと異なり、この推論CLIは `models/ldm/stable-diffusion-v1/model.ckpt` からVAE、UNet、CLIP text Encoderを含むfull Stable Diffusion modelをロードする。Stable DiffusionとADJSCCの全パラメータをeval・freezeし、推論全体を `torch.no_grad()` で実行する。

入力は中央正方形crop後に `--image-size` へLanczos resizeし、`[-1,1]` に変換する。ADJSCC checkpoint内metadataから `transmit_channel_num` と `feature_channels` を自動取得する。途中保存checkpointを指定した場合は、処理中epochと進捗率を警告表示し、`metadata.json`にも記録する。

### 11.2 strengthと2種類のノイズ

本パイプラインには独立した2種類のノイズがある。

| 引数 | 対象 | 意味 |
|---|---|---|
| `--snr-db` | 通信路AWGN | 小さいほど通信ノイズが強い |
| `--strength` | Stable Diffusion img2img | 大きいほど `z_hat` に強い拡散ノイズを加え、入力構造を変更しやすい |

`scripts/img2img.py` と同じ考え方で、DDIM逆拡散step数を次のように決める。

```python
t_enc = int(strength * ddim_steps)
```

- `strength=0`: 拡散ノイズとUNet逆拡散を省略し、ADJSCC受信潜在をそのままVAE復号
- `0 < strength < 1`: `t_enc` stepだけノイズ付加・逆拡散
- `strength=1`: 全DDIM stepを使用し、入力潜在の情報を最も強く崩す

元の `img2img.py` は `strength=1` のとき配列末尾を越える可能性があるため、新規実装ではノイズ付加indexを `t_enc-1` とする。これにより0～1の全範囲を有効にし、逆拡散step数とも整合させた。`strength`の実効分解能は `1/ddim_steps` である。

複数の `--num-samples` を指定した場合、ADJSCC/AWGNは入力画像に対して1回だけ実行する。その同じ `z_hat` を複製し、サンプルごとに異なるStable Diffusion拡散ノイズを加える。このため、複数出力間の差は通信路ではなくStable Diffusion側の確率性による。

主要な生成引数は次のとおり。

| 引数 | デフォルト | 用途 |
|---|---:|---|
| `--strength` | 0.35 | img2img拡散ノイズ量、0～1 |
| `--snr-db` | 0 | ADJSCCのAWGN SNR |
| `--ddim-steps` | 50 | DDIM schedule総step数 |
| `--ddim-eta` | 0 | DDIM samplingの追加確率性 |
| `--guidance-scale` | 5 | promptへのClassifier-Free Guidance強度。1ならCFGなし |
| `--num-samples` | 4 | 同じ通信結果から生成する画像数 |
| `--seed` | 42 | VAE posterior、AWGN、拡散ノイズの乱数seed |

### 11.3 フル推論コマンド

次は256px画像をSNR -5 dBで伝送し、`strength=0.35`でStable Diffusion img2img処理を行い、4枚生成するフルコマンドである。

```bash
cd /mnt/d/stable-diffusion
conda activate ldm

python ADJSCC/adjscc_sd_img2img.py \
  --init-img ../datasets/ffhq_train_70k/00000.png \
  --prompt "" \
  --negative-prompt "" \
  --output-dir ADJSCC/outputs/sd_adjscc_img2img/snr_20_strength_035_C32 \
  --adjscc-checkpoint ADJSCC/outputs/sd_vae_ffhq_stage1_C32/last.pt \
  --sd-config configs/stable-diffusion/v1-inference.yaml \
  --sd-checkpoint models/ldm/stable-diffusion-v1/model.ckpt \
  --image-size 256 \
  --snr-db 20 \
  --strength 0.75 \
  --ddim-steps 50 \
  --ddim-eta 0.0 \
  --guidance-scale 5.0 \
  --num-samples 4 \
  --seed 42 \
  --precision autocast \
  --device cuda
```

品質比較では同じ `--seed`、`--snr-db`、promptを保ち、`--strength` と出力ディレクトリだけを変更する。epoch完了後の最良モデルが存在する場合は、`last.pt`の代わりに `best.pt`を推奨する。

### 11.4 保存結果

指定した `--output-dir` には次を保存する。

```text
output_dir/
├── input.png
├── adjscc_received.png
├── sample_000.png
├── sample_001.png
├── sample_002.png
├── sample_003.png
├── comparison_grid.png
└── metadata.json
```

- `input.png`: 前処理済み入力画像
- `adjscc_received.png`: Stable Diffusion処理前の `z_hat` をVAE復号した通信結果
- `sample_XXX.png`: Stable Diffusion処理後の最終画像
- `comparison_grid.png`: 上段が入力、中段がADJSCC受信再構成、下段が最終画像
- `metadata.json`: prompt、SNR、strength、seed、checkpoint情報、潜在shape、実DDIM step数、全出力名

出力ファイル名は同一ディレクトリでは上書きされるため、SNRやstrengthごとに異なる `--output-dir` を指定する。

### 11.5 推論時の注意

- full Stable Diffusion、ADJSCC、生成batchを同時にGPUへ置く。CUDA OOM時は最初に `--num-samples 1`へ下げる。
- `--precision autocast`はCUDA時のみ有効で、CPUでは自動的にfull precisionとなる。
- `--guidance-scale 1`ではnegative promptを使用せず、UNetのbatch倍増も起きない。
- `strength=0`ではprompt conditioningとUNetを実行しないため、promptは出力に影響しない。
- safety checkerやwatermarkは追加していない。

## 12. 注意事項

- 70,000枚を使う本学習は実行していない。実装検証はCPUの最小構成で行った。
- 512pxで画像L1を使う第2段階は、凍結VAE Decoderのbackwardが必要なので第1段階よりVRAM使用量が大きい。
- CUDA OOM時は `batch-size` を1のまま保ち、`accumulation-steps` を増やす。必要なら `feature-channels` を下げるが、別アーキテクチャになるためcheckpoint互換性はなくなる。
- 本実装はPyTorch 1.11互換のfloat32処理を使用し、AMPは導入していない。
- VAE posterior sampleとAWGNの双方が確率的なので、評価値には揺らぎがある。厳密な比較では `eval-repeats` を増やす。
- 第2段階開始時は損失が変わるため、`--reset-optimizer --reset-best` を推奨する。
