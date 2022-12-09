import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

tags = [1,2,3,4,5,6,7,8] 
names = ["man-made terrain", "natural terrain", "high vegetation","low vegetation","buildings","hard scape","scanning artefacts","cars"]

tummls =[x + y + z for x, y,z in zip([ 396451, 211536, 1182876, 14685, 1371671, 47672, 60775, 76031],[ 378413, 87079, 544798, 77591, 1471600, 49776, 58748, 19286],[ 272215,0,42025,0,1118141,10477,21466,66174])]


print(tummls)
print([x / sum(tummls) for x in tummls])
tummls = [x / sum(tummls) for x in tummls]
semantic3D = [0.1688, 0.1051, 0.2394, 0.0357, 0.3600, 0.0409, 0.0383, 0.0117]


c = ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink','spring', 'summer']

X_axis = np.arange(len(tummls))

palette = sns.color_palette("Spectral_r", 8)  # Spectral, inferno, Spectral_r

fig = plt.figure()
ax = plt.subplot(111)
for i in range(0,len(names)):
    ax.bar(X_axis[i], tummls[i], color=palette[i],label=names[i])
# plt.bar(X_axis + 0.2, semantic3D, 0.4, label = 'Semantic3D', color=c)
# ax.legend( names, loc='center left', bbox_to_anchor=(1, 0.5))
print(palette)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)
for index,data in enumerate(tummls):
    plt.text(x=index-0.4 , y =data+0.01 , s=f"{data*100:.2f}"+"%" , fontdict=dict(fontsize=12))
plt.title("TUM MLS")
plt.show()

fig = plt.figure()
ax = plt.subplot(111)
for i in range(0,len(names)):
    ax.bar(X_axis[i], semantic3D[i], color=palette[i],label=names[i])

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)
for index,data in enumerate(semantic3D):
    plt.text(x=index-0.4, y =data+0.01 , s=f"{data*100:.2f}"+"%" , fontdict=dict(fontsize=12))
plt.title("Semantic3D")
plt.show()