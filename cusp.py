import numpy as np
import matplotlib.pyplot as plt

def train_data_gen():
    N = 20000
    ll_ = np.random.uniform(-2.5,2.5,N)
    xx_ = np.random.uniform(-2,2,N)

    mu_ = xx_**3 - ll_*xx_

    data = []
    for i in range(0,xx_.shape[0]):
        if np.abs(mu_[i])<= 2:
            data.append([mu_[i], ll_[i], xx_[i]])
    data = np.array(data)
    
    data_0 = []
    for i in range(0, data.shape[0]):
        if np.abs(data[i,0] - 0) <= 0.05:
            data_0.append(data[i,:])
        
    data_0 = np.array(data_0)
    
    data_3 = []
    data_1 = []

    for i in range(0, data.shape[0]):
        if data[i,0] < 2*data[i,1]/3*np.sqrt(data[i,1]/3) and data[i,0] > -2*data[i,1]/3*np.sqrt(data[i,1]/3):
            data_3.append(data[i,:])
        else:
            data_1.append(data[i,:])

        
    data_1 = np.array(data_1)
    data_3 = np.array(data_3)
    
    return data, data_0, data_1, data_3

def plot_cusp(data_1, data_3):
    fig = plt.figure(figsize=(5,6))
    ax = fig.add_subplot(111,projection = '3d')
    p = ax.scatter3D(*data_1.T, s=0.1, c='b', zorder=1)#, cmap='winter')
    p = ax.scatter3D(*data_3.T, s=0.1, c='k', zorder=1)#, cmap='winter')

    ax.set_ylabel(r'$\mu$', labelpad=10, fontsize=14)
    ax.set_xlabel(r'$\lambda$', labelpad=10, fontsize=14)
    ax.set_zlabel(r'$x$', labelpad=-9, fontsize=14)
    ax.tick_params(axis='z', which='major', pad=-2)

    ax.set_xlim([-2.5,2.5])
    ax.set_ylim([-2.5,2.5])
    ax.set_zlim([-2.5,2.5])
    ax.view_init(20,135)
    plt.show()