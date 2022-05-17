import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('titanic.csv')

x = df[df.columns.difference(['Survived'])]
y = df['Survived']

classifier = RandomForestClassifier()
classifier.fit(x, y)

print(x[2:4])
print(classifier.predict(x[2:4]))

joblib.dump(classifier, 'titanic_classifier.pkl')