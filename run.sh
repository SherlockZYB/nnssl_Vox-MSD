CUDA_VISIBLE_DEVICES=0 nnssl_train 745 onemmiso -tr VoxMSDEvaTrainer -p nnsslPlans -pretrained_weights /home/data/ZhouFF/SSL3D/nnssl_pretrained_models/PrimusM-OpenMind-MAE/checkpoint_final.pth
CUDA_VISIBLE_DEVICES=0 nnssl_train 745 onemmiso -tr VoxMSDTrainer -p nnsslPlans -pretrained_weights /home/data/ZhouFF/SSL3D/nnssl_pretrained_models/ResEncL-OpenMind-MAE/checkpoint_final.pth
