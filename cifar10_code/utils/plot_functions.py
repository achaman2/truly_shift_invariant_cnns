import numpy as np
import matplotlib.pyplot as plt 


def plot_save_train_val_list(list_to_be_plotted, path, numpy_flag = False):
    
    f = plt.figure()
    plt.plot(list_to_be_plotted)
    f.savefig(path + '.pdf')
    plt.close('all')
    
    if numpy_flag ==True:
        
        np.save(path + '.npy', list_to_be_plotted)
        
    
    
    