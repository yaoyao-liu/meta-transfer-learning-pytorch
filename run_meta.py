import os

def run_exp(num_batch=1000, shot=1, query=15, lr=0.0001, lr2=0.001, lr3=0.00001, base_lr=0.01, update_step=10, gamma=0.5):
    max_epoch = 100
    way = 5
    step_size = 10
    gpu = 0
    
    the_command = 'python3 main.py' \
        + ' --max_epoch=' + str(max_epoch) \
        + ' --num_batch=' + str(num_batch) \
        + ' --shot=' + str(shot) \
        + ' --train_query=' + str(query) \
        + ' --way=' + str(way) \
        + ' --meta_lr1=' + str(lr) \
        + ' --meta_lr2=' + str(lr2) \
        + ' --step_size=' + str(step_size) \
        + ' --gamma=' + str(gamma) \
        + ' --gpu=' + str(gpu) \
        + ' --base_lr=' + str(base_lr) \
        + ' --update_step=' + str(update_step) 

    os.system(the_command + ' --phase=meta_train')
    os.system(the_command + ' --phase=meta_eval')

run_exp(num_batch=100, shot=1, query=15, lr=0.0001, lr2=0.001, lr3=0.00001, base_lr=0.01, update_step=100, gamma=0.5)
run_exp(num_batch=100, shot=5, query=15, lr=0.0001, lr2=0.001, lr3=0.00001, base_lr=0.01, update_step=100, gamma=0.5)
