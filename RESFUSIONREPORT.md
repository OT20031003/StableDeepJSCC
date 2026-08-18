# Latent Resfusion 実装レポート

## 1. 実装概要

`memo/easy.tex` に記載した次の系を実装した。

```text
画像
  -> Stable Diffusion VAE encoder
  -> clean scaled latent z0
  -> ADJSCC encoder -> AWGN channel -> ADJSCC decoder
  -> degraded latent z_hat0
  -> 0～5 step Latent Resfusion
  -> 対応するStable Diffusion raw timestepへ直接handover
  -> Stable Diffusion reverse process
  -> Stable Diffusion VAE decoder
  -> 復元画像
```

追加したファイルは次の4個である。

| ファイル | 役割 |
|---|---|
| `ADJSCC/latent_resfusion.py` | Resfusion U-Net、T=12 LinearPro schedule、forward/reverse式、SD timestep対応、checkpoint読込み |
| `ADJSCC/train_latent_resfusion.py` | Latent Resfusion専用の学習CLI |
| `ADJSCC/infer_latent_resfusion_handover.py` | 0～5 stepの各状態からStable Diffusionへ直接handoverする推論CLI |
| `ADJSCC/tests/test_latent_resfusion.py` | 数式、schedule、mapping、U-Net、checkpoint、CLIの単体テスト |

学習と推論は別ファイルに分離した。両者で一致させる必要があるモデル・schedule・checkpoint処理だけを `latent_resfusion.py` に共有している。

## 2. 学習実装

### 2.1 学習pair

既存のStable Diffusion VAEと学習済みADJSCCを固定し、各画像からオンラインで次のpairを作る。

```text
clean target      : z0     = scaled Stable Diffusion VAE latent
degraded condition: z_hat0 = ADJSCC(z0, SNR)
prior residual    : R      = z_hat0 - z0
```

VAEのscale factorは設定ファイルの `0.18215` を使用する。VAEとADJSCCは常にevaluation modeかつ勾配なしで実行し、更新するのはLatent Resfusion U-Netだけである。latentにRGB用の `[-1,1]` clampや画像用range変換は行わない。

### 2.2 Network

公式Resfusion repository内の `RDDM_Unet` を再利用した。

- state `v_t`: 4 channel
- degraded condition `z_hat0`: 4 channel
- U-Net input: 8 channel
- predicted residual noise: 4 channel
- default width: `dim=64`, `dim_mults=(1,2,4,8)`
- SNR専用embedding、追加head、bridge networkはなし

### 2.3 Forward processとtarget

Resfusionのforward stateは次式で生成する。

```text
v_t = sqrt(alpha_bar_t) z0
    + (1 - sqrt(alpha_bar_t)) (z_hat0 - z0)
    + sqrt(1 - alpha_bar_t) epsilon
```

学習targetは原論文のresidual noiseだけである。

```text
eta_t = epsilon
      + ((1 - sqrt(alpha_t)) sqrt(1 - alpha_bar_t) / beta_t)
        (z_hat0 - z0)
```

各sampleについて `t` を一様に `{0,1,2,3,4}` から取り、`epsilon` を標準正規分布から取る。最適化目的は次のMSEだけである。

```text
loss = MSE(eta_theta(v_t, z_hat0, t), eta_t)
```

residual loss、bridge loss、score loss、統計loss、画像lossは追加していない。gradient clippingやmixed precisionは数値計算・最適化の制御であり、loss項は増やさない。

### 2.4 Resfusion schedule

公式実装と同じ `T=12` の `LinearProScheduler` を再現した。`sqrt(alpha_bar)=0.5` のacceleration pointで切るため、使用するResfusion timestepは `4,3,2,1,0` の5個になる。

初期状態は次式である。

```text
v_4 = 0.5 z_hat0 + sqrt(0.75) epsilon
```

reverse stepも公式のresidual-noise式とposterior varianceを使用し、最後の `t=0` だけ確率noiseを0にする。

## 3. Stable Diffusionへの直接handover

### 3.1 6個のhandover状態

`k` を完了済みResfusion denoise回数とすると、次の全状態を選択できる。

| `k` | Resfusion state | Resfusion U-Net呼出し回数 |
|---:|---|---:|
| 0 | 初期状態 `v_4` | 0 |
| 1 | `v_3` | 1 |
| 2 | `v_2` | 2 |
| 3 | `v_1` | 3 |
| 4 | `v_0` | 4 |
| 5 | 最終clean推定 `v_-1` | 5 |

選択した `k` までしかResfusion U-Netを実行しない。例えば `--handover-step 0` はResfusion U-Netを一度も呼ばない。

### 3.2 Timestep matching

対応値は固定配列だけに依存せず、ロードしたStable Diffusionモデルの実際の `alphas_cumprod` から毎回計算する。各Resfusion状態と各SD raw timestepについて、次の2係数の二乗距離が最小になるSD timestepを選ぶ。

```text
1 - sqrt(alpha_bar)
sqrt(1 - alpha_bar)
```

このrepositoryのStable Diffusion v1 scheduleでは、`easy.tex` と同じ対応になる。

| 完了Resfusion step `k` | Resfusion `alpha_bar` | SD raw timestep `tau_k` | SD reverse U-Net回数 |
|---:|---:|---:|---:|
| 0 | 0.250000 | 520 | 521 |
| 1 | 0.310431 | 475 | 476 |
| 2 | 0.575518 | 309 | 310 |
| 3 | 0.833902 | 146 | 147 |
| 4 | 0.991667 | 9 | 10 |
| 5 | 1.000000 | 0 | 1 |

### 3.3 Direct handoverの内容

handover時は現在のResfusion stateをそのままStable Diffusion latentとして渡す。

```text
z_SD[tau_k] <- v^(k)
```

次の処理は行わない。

- bridge/correction network
- 係数によるtensor補正
- Stable Diffusion側でのre-noise
- residual subtraction
- handover用の追加学習loss

Stable Diffusion側ではraw timestep `tau_k` を含めて0まで降ろすため、upstream `DDIMSampler.decode` を `t_start=tau_k+1`、`use_original_steps=True` で呼ぶ。`--ddim-eta 0` なら、このreverse processは追加の確率noiseを使わない。

## 4. 実行環境

repository rootで実行する。

```bash
cd /mnt/d/stable-diffusion
```

既存の `ldm` conda環境で、公式RDDM U-Netを含むCLIのimportとテストを確認済みである。推論時の画質評価には `lpips==0.1.4`、`DISTS-pytorch==0.1`、`pytorch-msssim==1.0.0` を使用し、いずれも `environment.yaml` に含めている。以下ではshellのactivate状態に依存しない `conda run` を使用する。

## 5. 学習コマンド

既存のC32 ADJSCC checkpointを使う標準的な学習例は次のとおりである。

```bash
python ADJSCC/train_latent_resfusion.py \
  --data-dir ../datasets/ffhq_train_70k \
  --sd-config configs/stable-diffusion/v1-inference.yaml \
  --sd-checkpoint models/ldm/stable-diffusion-v1/model.ckpt \
  --adjscc-checkpoint ADJSCC/outputs/sd_vae_ffhq_stage1_C32/best.pt \
  --output-dir ADJSCC/outputs/latent_resfusion_5step_C32 \
  --image-size 256 \
  --val-count 1000 \
  --split-seed 0 \
  --transmit-channel-num 32 \
  --feature-channels 256 \
  --snr-low-train -10 \
  --snr-up-train 20 \
  --snr-val 0 \
  --model-dim 64 \
  --dim-mults 1,2,4,8 \
  --resnet-block-groups 8 \
  --learning-rate 0.00011 \
  --weight-decay 0 \
  --epochs 100 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --accumulation-steps 1 \
  --grad-clip 1 \
  --val-every 1 \
  --log-every 100 \
  --num-workers 4 \
  --precision autocast \
  --device cuda
```

生成物は次のとおりである。

```text
ADJSCC/outputs/latent_resfusion_5step_C32/
  best.pt      # validation residual-noise MSEが最良のcheckpoint
  last.pt      # 最終epoch checkpoint
  history.json # epochごとのtrain/validation residual-noise MSE
```

`best.pt` と `last.pt` にはU-Net architecture、T=12/5-step設定、optimizer state、ADJSCC checkpoint情報を保存する。

### 学習再開

次の例は `last.pt` の次から20 epochを追加する。`--epochs` は総epoch数ではなく、今回追加で実行するepoch数である。

```bash
python ADJSCC/train_latent_resfusion.py \
  --data-dir ../datasets/ffhq_train_70k \
  --adjscc-checkpoint ADJSCC/outputs/sd_vae_ffhq_stage1_C32/best.pt \
  --output-dir ADJSCC/outputs/latent_resfusion_5step_C32 \
  --resume ADJSCC/outputs/latent_resfusion_5step_C32/last.pt \
  --epochs 200 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --accumulation-steps 1 \
  --device cuda
```

checkpoint内のarchitecture metadataを優先してU-Netを復元する。optimizerを初期化したい場合は `--reset-optimizer`、best値も初期化したい場合は `--reset-best` を付ける。

## 6. 推論コマンド

### 6.1 1個のhandover段階を評価

次はResfusionを3 step実行した `v_1` を、対応するStable Diffusion raw timestep 146へ直接渡す例である。

```bash
python ADJSCC/infer_latent_resfusion_handover.py \
  --init-img ADJSCC/outputs/sd_adjscc_img2img/snr_20_strength_035_C32/input.png \
  --adjscc-checkpoint ADJSCC/outputs/sd_vae_ffhq_stage1_C32/best.pt \
  --resfusion-checkpoint ADJSCC/outputs/latent_resfusion_5step_C32/best.pt \
  --sd-config configs/stable-diffusion/v1-inference.yaml \
  --sd-checkpoint models/ldm/stable-diffusion-v1/model.ckpt \
  --output-dir ADJSCC/outputs/resfusion_handover_k3 \
  --image-size 256 \
  --snr-db -10 \
  --handover-step 5 \
  --prompt "" \
  --guidance-scale 1 \
  --ddim-eta 0 \
  --num-samples 1 \
  --device cuda
```

`--handover-step` は0～5を指定できる。prompt guidanceを使用する場合は、例えば `--prompt "a high quality portrait" --negative-prompt "artifacts" --guidance-scale 7.5` とする。

### 6.2 0～5の全段階を一度に評価

```bash
python ADJSCC/infer_latent_resfusion_handover.py \
  --init-img ADJSCC/outputs/sd_adjscc_img2img/snr_20_strength_035_C32/input.png \
  --adjscc-checkpoint ADJSCC/outputs/sd_vae_ffhq_stage1_C32/best.pt \
  --resfusion-checkpoint ADJSCC/outputs/latent_resfusion_5step_C32/best.pt \
  --output-dir ADJSCC/outputs/resfusion_handover_all \
  --snr-db 0 \
  --all-handover-steps \
  --guidance-scale 1 \
  --ddim-eta 0 \
  --device cuda
```

各handover状態からStable Diffusionを独立に実行するため、全段階ではSD U-Netを合計 `521+476+310+147+10+1=1465` 回呼ぶ。特に `k=0` はraw timestep 520から開始するため、Resfusion自体は0回でもStable Diffusion側は521回であり、短時間の推論ではない。

### 6.3 推論出力

推論先には次を保存する。

```text
input.png
adjscc_received.png
summary_grid.png
metrics.json
metadata.json
handover_k{K}_sd_t{TAU}/
  resfusion_state.png
  sample_000.png
  comparison_grid.png
  metrics.json
```

各 `comparison_grid.png` は上からGT、ADJSCC受信画像、Resfusion state decode、Stable Diffusion sampleの順である。同じディレクトリの `metrics.json` にはGT以外の各画像をGTと比較したPSNR、LPIPS、DISTS、MS-SSIMを行・列との対応付きで記録する。複数sampleの場合は各sampleの値と平均値を保存する。出力rootの `metrics.json` は全handover段階の集約である。PSNRとMS-SSIMは表示用RGB `[0,1]`（data range 1）、LPIPSはAlexNet版を使用する。

`metadata.json` には実行引数、checkpoint情報、実際に計算した6段階のtimestep mapping、選択段階、出力path、および各metrics fileへのpathを記録する。

## 7. 検証

実行した検証コマンドは次のとおりである。

```bash
conda run -n ldm python -m py_compile \
  ADJSCC/latent_resfusion.py \
  ADJSCC/train_latent_resfusion.py \
  ADJSCC/infer_latent_resfusion_handover.py \
  ADJSCC/tests/test_latent_resfusion.py

conda run -n ldm python -m unittest discover -s ADJSCC/tests -v

conda run -n ldm python ADJSCC/train_latent_resfusion.py --help
conda run -n ldm python ADJSCC/infer_latent_resfusion_handover.py --help
```

結果はADJSCC全体で32 tests passedである。新規テストでは特に次を確認した。

- T=12 LinearPro scheduleが5 stepへ切られること
- forward stateとresidual-noise targetが `easy.tex` の閉形式に一致すること
- 初期状態が `0.5 z_hat0 + sqrt(0.75) epsilon` であること
- SD mappingが `[520,475,309,146,9,0]` になること
- 指定handover stepでResfusion計算が停止すること
- U-Netが8 channel入力、4 channel出力でbackpropagation可能なこと
- checkpointからarchitectureとweightを復元できること
- SD raw timestepを含めてreverseするため `t_start=tau+1` となること
- samplerがexact raw scheduleを使用して選択timestepから復元すること
- synthetic 1 epochでResfusion U-Netだけを更新できること
- 学習CLIと推論CLIが分離されていること
- 4画質指標が表示用RGB tensorをGTと比較し、複数sampleの平均にも4指標すべてが含まれること

検証環境ではCUDA deviceを利用できなかったため、実checkpointを使ったGPU学習と全Stable Diffusion推論は実行していない。数式・model forward/backward・checkpoint・CLI・既存ADJSCCとの回帰テストまではCPUで確認済みである。
