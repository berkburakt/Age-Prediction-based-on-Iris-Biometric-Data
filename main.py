import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def execute(attributeLength, classLength, trainingFile, testingFile):
    X_train, y_train = readFromFile(trainingFile)

    X_test, y_test = readFromFile(testingFile)

    model = Sequential()
    model.add(Dense(12, input_dim=attributeLength, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(classLength, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=150, batch_size=64)

    _, accuracy = model.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy * 100))


def readFromFile(files):
    X = []
    y = []

    matches = ['@RELATION', '@ATTRIBUTE', '@DATA']
    temp_X = []
    for file in files:
        temp_X = []
        y = []
        with open(file) as my_file:
            for line in my_file:
                if not any(x in line for x in matches):
                    result = [x.strip() for x in line.split(',')]
                    if len(result) > 1:
                        temp_X.append(result[:len(result) - 1])
                        y.append([result[len(result) - 1]])
        if len(X) == 0:
            X = temp_X

    if len(files) > 1:
        res_X = []
        for i in range(len(X)):
            a = X[i]
            b = temp_X[i]
            x = a + b
            res_X.append(x)

        X = res_X

    X = np.array(X, dtype=np.float128)
    y = np.array(y, dtype=np.float128)

    sc = StandardScaler()
    X = sc.fit_transform(X)

    ohe = OneHotEncoder()
    y = ohe.fit_transform(y).toarray()

    return [X, y]


def main():
    execute(5, 3, ['IrisGeometicFeatures_TrainingSet.txt'], ['IrisGeometicFeatures_TestingSet.txt'])
    # execute(9600, 3, ['IrisTextureFeatures_TrainingSet.txt'], ['IrisTextureFeatures_TestingSet.txt'])
    # execute(9605, 3, ['IrisGeometicFeatures_TrainingSet.txt', 'IrisTextureFeatures_TrainingSet.txt'], ['IrisTextureFeatures_TestingSet.txt', 'IrisGeometicFeatures_TestingSet.txt'])


if __name__ == '__main__':
    main()
