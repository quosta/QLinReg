import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

savedir = "/home/rugantio/Downloads/"

df = pd.read_csv("./ris/results.csv")
df.qa_t = df.bqm_t + df.qa_t
df.sa_t = df.bqm_t + df.sa_t

palette = ['#000000','#009900','#0000ff','#ff0000','#6699ff','#ff9966']


dashes = {'linreg_r2':(2,1),
          'sgd_r2':(1,1),
          'sa_r2':'',
          'qa_r2':'',
          'sa_ada_r2':'',
          'qa_ada_r2':''}

df = df[1:]
timings = df[['linreg_t', 'sgd_t', 'sa_t', 'qa_t','sa_ada_t', 'qa_ada_t']]
timings.index = df.features.astype(int)

fig, ax = plt.subplots(figsize=(9,6))
sns.lineplot(data=timings,dashes=False, palette = palette, ax = ax)
ax.set_yscale('log')
ax.set_xticks(timings.index)
ax.set_xlabel("Features")
ax.set_ylabel("Time (s)")
ax.legend(loc='lower right', labels=['Closed-Form','SGD','SA','QA','SA-Adaptive','QA-Adaptive'])
ax.grid()
fig.savefig(savedir + 'timings2.svg', bbox_inches='tight')

fig2, ax2 = plt.subplots(figsize=(9,6))
r2 = df[['linreg_r2', 'sgd_r2', 'sa_r2', 'qa_r2', 'sa_ada_r2','qa_ada_r2']]
r2.index = df.features.astype(int)
sns.lineplot(data=r2,dashes=dashes, palette = palette, ax = ax2)
ax2.set_xticks(r2.index)
ax2.set_xlabel("Features")
ax2.set_ylabel("$R^2$")

ax2.legend(loc='lower left', labels=['Closed-Form','SGD','SA','QA','SA-Adaptive','QA-Adaptive'])
ax2.grid()
fig2.savefig(savedir + 'r2_.svg', bbox_inches='tight')

# improv = pd.DataFrame({'sa_improvement': 100 * (r2.sa_ada_r2 - r2.sa_r2),
#                        'qa_improvement': 100 * (r2.qa_ada_r2 - r2.qa_r2)},
#                        index=df.features.astype(int))

# ax3 = sns.lineplot(data=improv,dashes=False, palette = ['#0000ff','#ff0000'])
# ax3.set_xticks(improv.index)
# ax3.set_xlabel("Features")
# ax3.set_ylabel("Improvement (%)")
# ax3.legend(loc='upper right', labels=['SA','QA'])
# ax3.grid()
