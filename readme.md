## Vox-MSD in nnssl Framework
Inspired by our recently published work - [Vox-MMSD](https://github.com/HiLab-git/Vox-MMSD), we design a self-supervised framework Vox-MSD for OpenMind pre-training. This is the nnssl version of Vox-MSD.

This code is used for SSL3D challenge. The command in run.sh is used for runing Vox-MSD under ResEncL and PrimusM.

## Bug！
Due to some unknown reason, I found that the dino loss would become nan sometimes, which I haven't met in my Vox-MMSD framework. And the timing of the nan appearing was completely random, and I didn't manage to find where the problem was. I chose to stop the experiment after the nan appeared and restart the training from the last checkpoint.