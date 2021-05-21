import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pandas import *
from sklearn.model_selection import train_test_split as train
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

path = 'wine.csv'
data = read_csv(path, delimiter=",")
print(data)

x = data.values[::, 1:14]
y = data.values[::, 0:1].ravel()
x_train, x_test, y_train, y_test = train(x, y, test_size=0.4)

clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
clf.fit(x_train, y_train)
print("Score: " + str(clf.score(x_test, y_test)))

x_train = x_train[::, (0, 10)]
x_test = x_test[::, (0, 10)]
clf.fit(x_train, y_train)

map_bold = ListedColormap(['#FFA07A', '#DC143C', '#483D8B'])
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=map_bold)
sns.kdeplot(x_test[:, 0], x_test[:, 1], c=y_test, hue=y_test)

x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
y_min, y_max = x_train[:, 1].min() - 0.2, x_train[:, 1].max() + 0.2
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.title("Score: %.0f percents" % (clf.score(x_test[::, (0, 1)], y_test) * 100))
plt.xlabel("Alcohol")
plt.ylabel("Hue")
plt.show()
