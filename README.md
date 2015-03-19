# Marketplace Recommendation Engine (Work In Progress)

A recommender API (for cluster computing) created for the Amplify Marketplace to recommend apps to users based on a machine learning technique called Collaborative Filtering with Alternating Least Squares (a process to fill in missing entries in a "User-App" rating matrix to estimate a user's affinity for an app). This project heavily relies on Spark and builds on principles of distributed computation. This project was deployed on AWS EC2 servers, but the API allows for portability to other services (see below).

Currently, the only metric we use to determine a "rating" is the number of installations of an app by a user. Hopefully this will be broadened in the future (to increase dimensionality of feature vectors). Also, the algorithm runs completely offline. A growing user base may necessitate an online version of the algorithm.

## Before You Get Started
 - Ensure the following are installed on your system:
       - Scala
       - Python
 - Run `pip install requirements.txt`
 - Download Apache Spark for your machine (https://spark.apache.org/downloads.html)
 - If running on a cluster, be sure to acquire a key (e.g. .pem file) for access to your cluster
 - Be sure to change and understand the fields in `config.yaml` below.

## Setting up `config.yaml`
- `file_path`: This should be the path to your properly formatted metrics data in a JSON file. **Note**: Depending on your cluster, the absolute path to the metrics should be preceded by the appropriate filesystem prefix (e.g. Amazon S3 uses `s3://`, HDFS uses `file://`, etc.)
- `path_to_userid`: Each properly formatted app metric should contain some sort of integer user ID. Since JSON metric data may be formatted differently as time goes on, you should specify the nested "path" (in the JSON object) to the user ID attribute.
    - Example: If your JSON metrics are formatted as such:
 `{"d":{"user_id": 001, "otherData": 123,...}...}` then you should put `["d", "user_id"]` for this field since `user_id` is nested under `d`.
- `path_to_appid`: Each properly formatted app metric should contain some sort of app ID (for now, it is a String). See above for information about this field.
- `num_threads`: This indicates how many threads Spark will use to run parallel computations on your data. The larger the number, the more efficient execution, at the cost of power and memory. If running on a large cluster, consider making this number larger.
- `num_tasks`: Indicates the number of threads Spark will use to sort/group RDDs (see above)
- `num_features`: This attribute is analogous to the rank of the matrix we choose. We reduce the rank of our matrix to compress data, keeping only the most salient features; however, the lower the number, the less accurate the matrix factorization will be in estimating the actual rating matrix. The larger the number, we lose memory and efficiency. Experiment with this number to find optimal accuracy when predicting recommendations. If using a kernel, add the number of extra dimensions to this number.
- `num_ALS_iterations`: The number of times the Alternating Least Squares algorithm will run. Running it too many times will slow runtime as well as overfit data.

### Running the script
`recommender.py` is an executable script. Simply run `$  ./recommender.py` on the command line with the following arguments:
- The number of top-rated apps you'd like to view
- Any number of valid user ID's to query results for
    * Example: `$ ./recommender.py 5 123456 789100` will return the top `5` recommendations for users `123456` and `789100`.

### TODO/Bugs
- The `sortedByKey` function call will run into an error in the `recommendApps` function. If we switch to Spark SQL to post-process the RDDs, this may be a non-issue.
    - Once this bug is resolved, fix other transformation errors on the RDD in `recommendApps`
- Extracting data is specific to a key-value format described above and only JSON compatible
- Experiment with current data (the historic data is quite old)
- Integrate other features in the matrix
- After adding more features, look into kernel tricks for faster computation
- Experiment with PCA/SVD techniques for online learning
- Separate out different distributions (e.g. grade, school, district, class, etc.) using SparkSQL to get better results
- Data visualization instead of command line outputs

### Version
0.0.1
