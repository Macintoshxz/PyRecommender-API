# Marketplace Recommendation Engine

A recommender created for the Amplify Marketplace to recommend apps to users based on a machine learning technique called "Collaborative Filtering" (a process to fill in missing entries in a "User-App" rating matrix to estimate a user's affinity for an app). This project heavily relies on Spark and builds on principles of distributed computation. 

Currently, the only metric we use to determine a "rating" is the number of installations. Hopefully this will be broadened in the future. Also, the algorithm runs completely offline. A growing user base may necessitate an online version of the algorithm.

Please email any questions to shraman@mit.edu, and I will be happy to clarify anything to keep this project going!

Author: Shraman Ray Chaudhuri (Extern, January 2015), MIT Class of 2017

Mentor: Joe Quadrino
## Before You Get Started
 - Ensure the following are installed on your system:
       - Scala
       - Python
 - Run `pip install requirements.txt`
 - Download Apache Spark for your machine (https://spark.apache.org/downloads.html)
 - If running on a cluster, be sure to get `data-emr-key.pem` for access privileges (from Joe)

## Setting up `config.yaml`
- `file_path`: This should be the path to your (properly formatted) metrics data. **Note**: Depending on your cluster, the absolute path to the metrics should be preceded by the appropriate filesystem prefix (e.g. Amazon S3 uses `s3://`, HDFS uses `file://`, etc.)
- `path_to_userid`: Each properly formatted app metric should contain some sort of integer user ID. Since JSON metric data may be formatted differently as time goes on, you should specify the nested "path" (in the JSON object) to the user ID attribute.
        - **Example**: If your JSON metrics are formatted as such:
 `{"d":{"user_id": 001, "otherData": 123,...}...}` then you should put `["d", "user_id"]` for this field since `user_id` is nested under `d`.
- `path_to_appid`: Each properly formatted app metric should contain some sort of app ID (for now, it is a String). See above for information about this field.
- `num_threads`: This indicates how many threads Spark will use to run parallel computations on your data. The larger the number, the more efficient execution, at the cost of power and memory. If running on a large cluster, consider making this number larger.
- `num_tasks`: Indicates the number of threads Spark will use to sort/group RDDs (see above)
- `num_features`: This attribute is analogous to the rank of the matrix we choose (a.k.a. the number of singular values in our matrix factorization). We reduce the rank of our matrix to compress data, keeping only the most salient features; however, the lower the number, the less accurate the matrix factorization will be in representing the actual User-App rating matrix. The larger the number, we lose memory and efficiency. Experiment with this number to find optimal accuracy when predicting recommendations.
- `num_ALS_iterations`: The number of times the Alternating Least Squares algorithm will run. Running it too many times will slow runtime as well as overfit data. Again, experiment with this.
### Running the script
`recommender.py` is an executable script. Simply run `$  ./recommender.py` on the command line with the following arguments:
- The number of top-rated apps you'd like to view
- Any number of valid user ID's to query results for
**Example**: `$ ./recommender.py 5 123456 789100` will return the top `5` recommendations for users `123456` and `789100`.
### Version
0.0.1