# KNN
import pandas as pd

df = pd.read_csv('Bank_Personal_Loan_Modelling.csv', encoding='utf-8')
df.info()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df['Income_Category'] = pd.cut(df['Income'], bins=[-1, 30000, 70000, float('inf')], labels=['Low', 'Medium', 'High'])

X = df.drop(['Income', 'Income_Category'], axis=1)
y = df['Income_Category']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy * 100)
