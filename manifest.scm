;; Guix manifest for Associative memory models development

(specifications->manifest
 '(;; Core Python
   "python"
   
   ;; Runtime dependencies
   "python-pytorch-cuda"
   "python-torchvision-cuda"
   "python-pytorch-geometric"
   "python-einops"
   "python-tqdm"
   "python-hydra-core"
   "python-pandas"
   "python-pyarrow"
   
   ;; Audio/Video processing for GRID dataset
   "python-librosa"
   "python-soundfile"
   "python-decord"  ;; Video reading for MovieChat dataset
   
   ;; Development dependencies
   "python-pytest"
   "python-pytest-cov"
   "python-ruff"
   "node-pyright"
   
   ;; Build system
   "poetry"))
