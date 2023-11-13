import numpy as np

import torch
import matplotlib.pyplot as plt
lst1=np.load('c10_log_probs.npy',allow_pickle=True)
lst2=np.load('svhn_log_probs.npy',allow_pickle=True)

def logsumexp(logits):
    return torch.logsumexp(logits, dim=1, keepdim=False)
for idx in range(54):
    if(len(lst1[idx])==0):
        continue
    else:
        a=logsumexp(torch.tensor(lst1[idx]))
        #print(a)
        b = logsumexp(torch.tensor(lst2[idx]))
        #print(a)
        import seaborn as sns
        ax=sns.distplot(a.tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3},
                     label='ID')
        sns.distplot(b.tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3},
                     label='OOD')
        ax.set(xlabel='Log-Density', ylabel='Density',title='DDU Plot with Gate No ' + str(idx))
        plt.legend()
        fig = ax.get_figure()
        fig.savefig('Figs/gate_'+str(idx)+'.png')
        fig.clf()

        # print(name,' ',dataset)
        # print(tmp_lst1)
        # print(tmp_lst2)
        '''import matplotlib.pyplot as plt
        plt.legend(prop={'size': 16}, title='Evidential NN')
        plt.title('DDU Plot with Gate No ' + str(idx) )
        plt.xlabel('Log-Density')
        plt.ylabel('Density')
        plt.show()
        plt.savefig('Figs/gate_'+str(idx)+'.png')
        ax.clf()'''

'''print(len(lst1))
print(len(lst2))

import seaborn as sns

sns.distplot(lst1, hist=False, kde=True,
             kde_kws={'shade': True, 'linewidth': 3},
             label='OOD')
sns.distplot(lst2, hist=False, kde=True,
             kde_kws={'shade': True, 'linewidth': 3},
             label='ID')

# print(name,' ',dataset)
# print(tmp_lst1)
# print(tmp_lst2)


plt.legend(prop={'size': 16}, title='Evidential NN')
#plt.title('Exit No Density Plot with PGD and FGSM Attack for ' + dataset + ' and  ' + name + ' model')
plt.xlabel('Energy Score')
plt.ylabel('Density')
plt.show()'''