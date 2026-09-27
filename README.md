# vae-anomaly-detection
Image anomaly detection using Variational Autoencoder (VAE) implemented with PyTorch

# Autoencoder Experiments & Image Anomaly Detection

PyTorchを用いてAutoencoder（AE）を基礎から実装し、モデル構造や潜在次元、パラメータ数が再構成性能に与える影響を検証したプロジェクトです。

基礎的なFully Connected Autoencoderから実験を開始し、Convolutional Autoencoderへの拡張、ハイパーパラメータ探索、モデル容量に関する実験を行いました。

さらに、学習した知識を画像異常検知へ応用し、MVTec ADを用いた異常検知モデルを実装しています。

---

## Project Overview

本プロジェクトでは、以下の流れでAutoencoderについて実装・検証しました。

1. Fully Connected Autoencoderの実装
2. Convolutional Autoencoderへの拡張
3. 潜在次元による再構成性能の比較
4. Optunaを用いたハイパーパラメータ探索
5. モデルのパラメータ数と汎化性能の関係を検証
6. MVTec ADを用いた画像異常検知への応用

単にモデルを実装するだけではなく、モデル構造や容量を変更しながらTrain/Test Lossを比較し、モデルの表現能力と汎化性能の関係について実験しました。

---

## 1. Fully Connected Autoencoder

MNISTを用いて、全結合層によるAutoencoderを実装しました。

### Architecture

```text
Input: 28 × 28 = 784

Encoder
784
 ↓
512
 ↓
256
 ↓
128

Decoder
128
 ↓
256
 ↓
512
 ↓
784
```

Activation functionにはReLUを使用し、Decoderの出力にはSigmoidを使用しています。

### Training

* Dataset: MNIST
* Loss: Mean Squared Error (MSE)
* Optimizer: Adam
* Learning Rate: 0.001
* Batch Size: 64
* Epochs: 10

学習後、元画像と再構成画像を比較してAutoencoderが入力画像の特徴を圧縮・再構成できていることを確認しました。

---

## 2. Convolutional Autoencoder

画像の空間的特徴をより効果的に学習するため、CNNを利用したConvolutional Autoencoderへ拡張しました。

### Encoder

```text
Input (1 × 28 × 28)

Conv2d
1 → 16
 ↓
Conv2d
16 → 32
 ↓
Conv2d
32 → 64
```

DecoderではConvTranspose2dを使用して元の画像サイズへ復元します。

Fully Connected AEとConvAEの双方について、再構成画像とLossを比較しました。

---

## 3. Latent Dimension Experiment

Autoencoderの潜在次元を変更し、モデルの表現能力と再構成性能の関係を検証しました。

検証したlatent dimension：

```text
2, 4, 8, 16, 32, 64,
128, 256, 512, 784,
1024, 2048
```

各モデルについてTrain LossとTest Lossを計測しました。

### Result

![Latent Dimension Experiment](results/double_descent_grid_search_plot.png)

実験では、latent dimensionを2から64まで増加させることでTest Lossが大きく低下しました。

特にlatent dimension = 64では、

```text
Train Loss : 0.005856
Test Loss  : 0.006224
```

となりました。

64以降では、latent dimensionを増加させてもTest Lossの大幅な改善は見られず、おおむね0.006〜0.007付近で推移しました。

この結果から、今回のモデル・学習条件では、潜在表現の容量を一定以上増加させても再構成性能の改善が限定的になることを確認しました。

---

## 4. Model Capacity / Double Descent Experiment

潜在次元だけではなく、ネットワークのwidthを変更することでモデル全体のパラメータ数を変化させ、モデル容量とTrain/Test Lossの関係について検証しました。

検証したwidth：

```text
8, 16, 32, 64, 128,
256, 512, 1024, 2048, 4096
```

パラメータ数は約1.5万〜4,000万まで変化します。

### Result

![Parameter Count Experiment](results/grid_search_parameter_count_plot.png)

多くの条件では、パラメータ数の増加に伴ってTrain/Test Lossが低下しました。

一方、約1,187万パラメータのモデル（width = 2048）ではTrain/Test Lossがともに大きく増加し、その後の約4,052万パラメータのモデル（width = 4096）では再び低下する挙動が観測されました。

ただし、この結果だけからDouble Descentが発生したと断定することはできないため、学習条件や最適化の影響を含めた追加検証が必要だと考えています。

---

## 5. Hyperparameter Optimization

Optunaを用いてAutoencoderのハイパーパラメータ探索も実装しました。

探索対象：

* Learning Rate
* Latent Dimension

Validation Lossを目的関数として、より良いハイパーパラメータの探索を行います。

---

## 6. Image Anomaly Detection

Autoencoderの応用として、産業用画像異常検知データセット **MVTec AD** を用いた異常検知を実装しました。

使用カテゴリ：

```text
zipper
```

### Method

Convolutional Autoencoderを正常画像から学習させ、入力画像と再構成画像のMSEをAnomaly Scoreとして使用します。

```text
Input Image
      ↓
Convolutional Autoencoder
      ↓
Reconstructed Image
      ↓
Reconstruction Error
      ↓
Anomaly Score
      ↓
Normal / Anomaly
```

正常画像のAnomaly Scoreから閾値を設定し、正常・異常を分類します。

### Evaluation

以下の指標を用いて性能を評価しています。

* Reconstruction Error
* Anomaly Score
* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score

---

## Technologies

* Python
* PyTorch
* torchvision
* NumPy
* Matplotlib
* scikit-learn
* Optuna
* CUDA

---

## Repository Structure

```text
autoencoder-experiments/
│
├── README.md
├── requirements.txt
│
├── mnist/
│   ├── autoencoder.py
│   ├── conv_autoencoder.py
│   └── optuna_search.py
│
├── experiments/
│   ├── latent_dimension_search.py
│   └── parameter_count_search.py
│
├── anomaly_detection/
│   └── mvtec_autoencoder.py
│
├── results/
│   ├── double_descent_grid_search_plot.png
│   └── grid_search_parameter_count_plot.png
│
└── data/
    └── .gitkeep
```

---

## Future Work

今後は以下の検証・改善を予定しています。

* 複数seedでの再実験
* Double Descent現象の再現性検証
* モデル容量と最適化条件の関係の調査
* 異常判定閾値の改善
* MVTec ADの他カテゴリへの適用
* 潜在空間の可視化
* Variational Autoencoder（VAE）との比較
* 他の異常検知手法との比較

---

## Author

Kaito Hayashi
Department of Electrical and Electronic Engineering
Ritsumeikan University
