;; Guix manifest for Associative memory models development

(specifications->manifest
 '(;; Core Python
   "python"
   ;; Runtime dependencies
   "python-pytorch"
   "python-torchvision"
   "python-pytorch-geometric"
   "python-einops"
   "python-tqdm"
   "python-hydra-core"
   "python-pandas"
   "python-pyarrow"
   "python-librosa"
   "python-soundfile"
   "python-decord"
   ;; Development dependencies
   "python-pytest"
   "python-pytest-cov"
   "python-ruff"
   "node-pyright"))
