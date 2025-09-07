;; Guix manifest for Associative memory models development

(specifications->manifest
 '(;; Core Python
   "python"
   ;; Runtime dependencies
   "python-pytorch"
   "python-torchvision"
   "python-pytorch-geometric"
   "python-transformers"
   "python-einops"
   "python-tqdm"
   "python-hydra-core"
   "python-pandas"
   "python-pesq"
   "python-pystoi"
   "python-torchaudio"
   "python-torchvggish"
   "python-torchmetrics"
   "python-pyarrow"
   "python-librosa"
   "python-soundfile"
   "python-decord"
   "python-huggingface-hub"
   ;; Visualization and analysis
   "python-pillow"
   "python-matplotlib"
   "python-scikit-learn"
   ;; Development dependencies
   "python-pytest"
   "python-pytest-cov"
   "python-ruff"
   "node-pyright"))
