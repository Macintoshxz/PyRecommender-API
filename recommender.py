#!/usr/bin/env python

from pyspark.sql import *
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS
import json
import sys
import yaml
from operator import add

with open("config.yaml") as configFile:
    config = yaml.load(configFile)

# Global variables
schemaColumnsToPaths = {"user_id":config["metrics"]["path_to_userid"],\
                        "app_id":config["metrics"]["path_to_appid"]}
schemaColumnNames = schemaColumnsToPaths.keys()
appsToCodes = {}
codesToApps = {}
appCounter = 0
throwawayFlags = ["improper_format"]

def userAppJSONParser(line):
    '''
    Map a line (string representing a JSON object) to a list of tuples (each value representing respective column in schema)
    
    Args:
        line:               (str) line formatted as a single JSON object
    '''
    try:
        flatJSON = jsonFlatten(json.loads(line), schemaColumnsToPaths)
    except KeyError:
        # Do not add entries with missing fields
        return "improper_format"

    # Translate app name to app ID
    global appCounter
    if flatJSON["app_id"] not in appsToCodes:
        appsToCodes[flatJSON["app_id"]] = appCounter
        codesToApps[appCounter] = flatJSON["app_id"]
        appCounter += 1
    flatJSON["app_id"] = appsToCodes[flatJSON["app_id"]]
    
    # Set the 'rating' attribute
    flatJSON["rating"] = 1
    
    return ((flatJSON["user_id"], flatJSON["app_id"]), flatJSON["rating"])

def jsonFlatten(jsonDict, newKeysToValPaths):
    '''
    Flatten a nested dictionary. (Used for nested JSON files)
    
    Args:
        jsonDict:           (dict) to flatten
        newKeysToValPaths:  (dict<str, list>) map new custom keys (str) to paths (list) in jsonDict that hold the original data
                                to be stored. Paths are in the format [key1, key2,...] (where key2 is in a dictionary nested under
                                key1).
    
                                Example: to map "date" in flattened dictionary to jsonDict['path']['to']['data'], insert the following
                                (key, value) pair: ("date", ['path', 'to', 'data'])
    
    Returns:
        A flattened dictionary containing elements under jsonDict[rootKey] and data defined by key/path pairs in newKeysToValPaths
    '''
    newDict = {}

    # Add key/val pairs to newDict as specified by newKeysToValPaths
    for key, dictPath in newKeysToValPaths.items():
        newDict[key] = extractVal(jsonDict, dictPath)

    return newDict

def extractVal(d, path):
    '''
    Recursively extract values from a dictionary given a path.

    Args:
        d:                  (dict) to extract from
        path                (list) containing a sequences of nested keys to follow (just like a regular filepath, except for dicts)
    '''
    if(len(path) == 1):
        rawValue = d[path[0]]
        # If value is a dictionary, stringify it to serialize it
        return rawValue if not(isinstance(rawValue, dict)) else json.dumps(rawValue)
    else:
        return extractVal(d[path[0]], path[1:])

def recommendApps(user_id, model, ratingRDD, N):
    '''
    Use a prepared ALS-trained matrix model to predict the top N apps for a user.

    Args:
        user_id:            (int) user's ID as it appears in the prediction matrix
        model               (MatrixFactorizationModel) containing predicted values for missing entries in the matrix
        ratingRDD           (RDD) the original rating RDD used to make the matrix model
        N                   (int) number of top apps to calculate
    '''
    user_ID = int(user_id)

    # Only consider apps which haven't been downloaded
    oldApps = set(ratingRDD.filter(lambda p: p[0] == user_ID).map(lambda p: int(p[1])).collect())
    appsToPredict = sc.parallelize([(user_ID, appCode) for appCode in codesToApps.keys() if appCode not in oldApps])

    # Predict the "ratings", sort them in descending order, and return the first N (or all, if N > # apps)
    predictions = model.predictAll(appsToPredict)\
                        .map(lambda p: [p[2],p[1]])\
                        .sortByKey(False, config["spark"]["num_tasks"])\
                        .take(min(N, len(appsToPredict)))\
                        .map(lambda p: codesToApps[p[1]]).collect()
    # Return a list of app names
    return predictions


if __name__ == "__main__":

    # Create the spark/sparkSQL contexts
    spConf = SparkConf().setAppName(config["spark"]["app_name"])\
                        .setMaster("local["+str(config["spark"]["num_threads"])+"]")
    sc = SparkContext(conf=spConf)

    # Load JSON data from a text file and parse/reduce counts
    appDataRDD = sc.textFile(config["metrics"]["file_path"])
    ratingRDD = appDataRDD.map(userAppJSONParser)\
                            .filter(lambda x: x not in throwawayFlags)\
                            .reduceByKey(add, config["spark"]["num_tasks"])\
                            .map(lambda p: [p[0][0], p[0][1], p[1]])

    # Create MatrixFactorizationModel with Spark's ALS training algorithm
    model = ALS.train(ratingRDD, config["ALS"]["num_features"], config["ALS"]["num_ALS_iterations"])

    # Take command line arguments for:
    #   [0]: number of apps desired
    #   [1:]: user_id's for whom to generate recommendations
    if len(sys.argv) < 2:
        print "First argument must be number of apps to return\n" + "Second argument must be a user ID"
        sys.exit()
    for user_id in sys.argv[1:]:
        print recommendApps(user_id, model, ratingRDD, sys.argv[0])
