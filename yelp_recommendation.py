import os
import sys
from pyspark import SparkContext, SparkConf
import time
import json
import math
import numpy as np
import xgboost as xgb
import csv
from sklearn.metrics import mean_squared_error

N = 7
alpha = 0.001


def writeToCSV(output_file_name, predictions):
    with open(output_file_name, 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar='\\')
        writer.writerow(["user_id", "business_id", "prediction"])
        for row in predictions:
            writer.writerow(row)


def getCoRatedUsers(u1, u2):
    userList1 = set(map(lambda x: x[0], u1))
    userList2 = set(map(lambda x: x[0], u2))

    return userList1.intersection(userList2)


def getCoRatedUserList(target, neighbours, businessUserDict):
    coRateDict = {}
    neighbourList = list(map(lambda x: x[0], neighbours))
    targetUsersList = businessUserDict[target]

    for neighbour in neighbourList:
        neighbourUserList = businessUserDict[neighbour]
        coRatedUsers = getCoRatedUsers(targetUsersList, neighbourUserList)
        if len(coRatedUsers) > 30:
            coRateDict[(target, neighbour)] = coRatedUsers

    return coRateDict


def getRating(coRatedUsers, userRatingList):
    ratings = list()
    for user in coRatedUsers:
        for uid_rate in userRatingList:
            if user == uid_rate[0]:
                ratings.append(uid_rate[1])

    return ratings


def calcW(avg_b1, avg_b2, rating_b1, rating_b2):
    numerator = sum(map(lambda pair: (pair[0] - avg_b1) * (pair[1] - avg_b2), zip(rating_b1, rating_b2)))
    denominator = math.sqrt(sum(map(lambda rate: pow((rate - avg_b1), 2), rating_b1))) * math.sqrt(
        sum(map(lambda rate: pow((rate - avg_b2), 2), rating_b2)))

    if numerator == 0 or denominator == 0:
        return 0

    return numerator / denominator


def getPearsonCoeff(target, neighbours, businessUserDict, businessAvgRating, coRateUserList):
    b1 = target
    u1 = businessUserDict[b1]
    avg_b1 = businessAvgRating[b1]
    result = list()

    for pairs in coRateUserList.items():
        coRatedUsers = list(pairs[1])
        b2 = pairs[0][1]
        u2 = businessUserDict[b2]
        avg_b2 = businessAvgRating[b2]
        rating_b1 = getRating(coRatedUsers, u1)
        rating_b2 = getRating(coRatedUsers, u2)
        weight = calcW(avg_b1, avg_b2, rating_b1, rating_b2)
        for item in neighbours:
            if item[0] == b2:
                result.append(tuple((item[1], weight)))

    return result


def adjustedRating(rating):
    if rating < 1:
        return 1.0
    elif rating > 5:
        return 5.0
    else:
        return rating


def getPrediction(businessList, businessAvgRating, businessUserDict):
    target = businessList[0]
    neighbours = businessList[1]

    coRateUserList = getCoRatedUserList(target, neighbours, businessUserDict)

    if len(coRateUserList) > 0:
        result = getPearsonCoeff(target, neighbours, businessUserDict, businessAvgRating, coRateUserList)
        result = list(filter(lambda x: x[1] > 0.0001, result))
        ratingWeightList = sorted(result, key=lambda item: item[1], reverse=True)[:N]

        numerator = sum(map(lambda x: x[0] * x[1], ratingWeightList))
        denominator = sum(map(lambda x: abs(x[1]), ratingWeightList))

        if numerator == 0 or denominator == 0:
            return tuple((target, businessAvgRating[target]))

        predictedRating = numerator / denominator

    else:
        predictedRating = businessAvgRating[target]

    return tuple((target, adjustedRating(predictedRating)))


def itemBasedCF(trainRDD, testRDD):
    trainUsersIndex = trainRDD.map(lambda x: x[0]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()
    revUsersIndex = {v: k for k, v in trainUsersIndex.items()}

    trainBusinessIndex = trainRDD.map(lambda x: x[1]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()
    revBusinessIndex = {v: k for k, v in trainBusinessIndex.items()}

    utilityMatrix = trainRDD.map(lambda x: (trainUsersIndex[x[0]], (trainBusinessIndex[x[1]], x[2]))) \
        .groupByKey().mapValues(list)

    businessAvgRating = trainRDD.map(lambda x: (trainBusinessIndex[x[1]], x[2])) \
        .groupByKey().mapValues(list) \
        .map(lambda x: (x[0], sum(x[1]) / len(x[1]))).collectAsMap()

    businessUserDict = trainRDD.map(lambda x: (trainBusinessIndex[x[1]], (trainUsersIndex[x[0]], x[2]))) \
        .groupByKey().mapValues(list).collectAsMap()

    invalidPairs = testRDD.filter(
        lambda pair: trainUsersIndex.get(pair[0], -1) == -1 or trainBusinessIndex.get(pair[1], -1) == -1) \
        .map(lambda x: (x[0], x[1], 3.8)).collect()

    testRDD = testRDD.map(lambda pair: (trainUsersIndex.get(pair[0], -1), trainBusinessIndex.get(pair[1], -1))) \
        .filter(lambda pair: pair[0] != -1 and pair[1] != -1)

    predictions = testRDD.leftOuterJoin(utilityMatrix) \
        .mapValues(lambda x: getPrediction(x, businessAvgRating, businessUserDict)) \
        .map(lambda x: (revUsersIndex[x[0]], revBusinessIndex[x[1][0]], float(x[1][1]))).collect()

    predictions.extend(invalidPairs)

    writeToCSV()

    return predictions


def modelBased(trainRDD, testRDD, userFeatures, businessFeatures):
    trainUtilityMatrix = trainRDD.map(lambda x: (x[0], x[1], userFeatures.get(x[0]), businessFeatures.get(x[1]), x[2])) \
        .map(lambda x: (x[0], x[1], x[2][0], x[2][1], x[3][0], x[3][1], x[4])).collect()

    testUtilityMatrix = testRDD.map(lambda x: (x[0], x[1], userFeatures.get(x[0]), businessFeatures.get(x[1]))) \
        .map(lambda x: (x[0], x[1], x[2][0], x[2][1], x[3][0], x[3][1])).collect()

    arrayTrain = np.array(trainUtilityMatrix)
    arrayTest = np.array(testUtilityMatrix)

    X_train = np.array(arrayTrain[:, [2, 3, 4, 5]])
    y_train = np.array(arrayTrain[:, 6], dtype="float")

    X_test = np.array(arrayTest[:, [2, 3, 4, 5]])

    model = xgb.XGBRegressor(objective='reg:linear')
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    predList = list(pred)

    temp = testRDD.collect()
    user, bID = zip(*temp)
    predictions = list(zip(user, bID, predList))

    return predictions


def main():
    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    configuration = SparkConf()
    configuration.set("spark.driver.memory", "4g")
    configuration.set("spark.executor.memory", "4g")
    sc = SparkContext.getOrCreate(configuration)
    sc.setLogLevel("ERROR")

    start = time.time()

    rdd = sc.textFile(os.path.join(folder_path, 'yelp_train.csv'))
    header = rdd.first()
    trainRDD = rdd.filter(lambda line: line != header) \
        .map(lambda x: (x.split(',')[0], x.split(',')[1], float(x.split(',')[2])))

    rdd = sc.textFile(test_file_name)
    header = rdd.first()
    testRDD = rdd.filter(lambda line: line != header).map(lambda x: (x.split(',')[0], x.split(',')[1]))

    userFeatures = sc.textFile(os.path.join(folder_path, 'user.json')).map(lambda x: json.loads(x)) \
        .map(lambda x: (x['user_id'], (int(x['review_count']), float(x['average_stars'])))).collectAsMap()

    businessFeatures = sc.textFile(os.path.join(folder_path, 'business.json')).map(lambda x: json.loads(x)) \
        .map(lambda x: (x['business_id'], (int(x['review_count']), float(x['stars'])))).collectAsMap()

    itemBasedPredictions = itemBasedCF(trainRDD, testRDD)

    modelBasedPredictions = modelBased(trainRDD, testRDD, userFeatures, businessFeatures)

    item = sc.parallelize(itemBasedPredictions).map(lambda x: ((x[0], x[1]), x[2])).collectAsMap()
    model = sc.parallelize(modelBasedPredictions).map(lambda x: ((x[0], x[1]), x[2])).collectAsMap()

    finalScore = []
    for uid_bid, rating in item.items():
        finalScore.append(tuple((uid_bid[0], uid_bid[1], (rating * alpha) + (model.get(uid_bid) * (1 - alpha)))))

    writeToCSV(output_file_name, finalScore)

    end = time.time()
    print("Duration:", end - start)

    valRDD = sc.textFile(os.path.join(folder_path, 'yelp_val.csv'))
    y_true = valRDD.filter(lambda line: line != header).map(lambda x: float(x.split(',')[2])).collect()
    y_pred = list(map(lambda x: x[2], finalScore))
    print("RMSE:", mean_squared_error(y_true, y_pred, squared=False))


if __name__ == '__main__':
    main()