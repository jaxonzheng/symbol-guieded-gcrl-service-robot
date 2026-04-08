## Acknowledgements

This work builds upon the Stable Contrastive Reinforcement Learning (SCRL) implementation by Chongyi Zheng:
https://github.com/chongyi-zheng/stable_contrastive_rl

mamba create -y --prefix /work/rleap1/jaxon.cheng/venvs/stable_contrastive_rl \
  -c conda-forge -c pytorch -c nvidia \
  python=3.9 "numpy<2" mujoco=3.1.6 opencv glfw pyopengl pip \
  pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=11.8

mamba install -y -c pytorch -c nvidia pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=11.8

pip install jax==0.4.7 "jaxlib==0.4.7+cuda11.cudnn82" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install click urchin hello-robot-stretch-urdf gym Pillow joblib python-dateutil absl-py pybullet pygame sk-video pandas matplotlib

pip install pyyaml

