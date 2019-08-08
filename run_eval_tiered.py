import os

def run_exp(num_batch=1000, shot=1, query=15, lr=0.0001, lr2=0.001, lr3=0.00001, base_lr=0.01, update_step=10, gamma=0.5):
    #num_batch = 1000
    max_epoch = 100
    #shot = 1
    #query = 15
    way = 5
    #lr = 0.0001
    #lr2 = 0.001
    step_size = 10
    #gamma = 0.5
    gpu = 2
    label = 'exp6'
    eval_weights = '/BS/sun_project_multimodal/work/yyliu_project/lcc-new/Mtl-PyTorch-12-0-0/logs/meta/TieredImageNet_ResNet_MTL_shot5_way5_query15_step10_gamma0.5_lr0.0001_lr20.001_lr31e-05_batch100_maxepoch100_baselr0.01_updatestep100_stepsize10_exp6/max_acc.pth'
    
    the_command = 'python3 main.py' \
        + ' --max_epoch=' + str(max_epoch) \
        + ' --num_batch=' + str(num_batch) \
        + ' --shot=' + str(shot) \
        + ' --train_query=' + str(query) \
        + ' --way=' + str(way) \
        + ' --lr=' + str(lr) \
        + ' --lr2=' + str(lr2) \
        + ' --step_size=' + str(step_size) \
        + ' --gamma=' + str(gamma) \
        + ' --gpu=' + str(gpu) \
        + ' --base_lr=' + str(base_lr) \
        + ' --update_step=' + str(update_step) \
        + ' --lr3=' + str(lr3) \
        + ' --label=' + label \
        + ' --eval_weights=' + eval_weights \
        + ' --dataset=TieredImageNet' \
        + ' --phase=meta_eval' \
        + ' --val_query=15' 

    os.system(the_command)

run_exp(num_batch=100, shot=5, query=15, lr=0.0001, lr2=0.001, lr3=0.00001, base_lr=0.01, update_step=100, gamma=0.5)

