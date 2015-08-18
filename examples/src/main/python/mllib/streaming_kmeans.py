#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
 Create a queue of RDDs that will be mapped/reduced one at a time in
 1 second intervals.

 To run this example use
    `$ bin/spark-submit examples/src/main/python/streaming/queue_stream.py
"""
import sys
import time

from pyspark import SparkContext
from pyspark.mllib.clustering import StreamingKMeans
from pyspark.streaming import StreamingContext

if __name__ == "__main__":

    sc = SparkContext(appName="PythonStreamingKMeans")
    ssc = StreamingContext(sc, 1)

    """Test that prediction happens on the updated model."""
    stkm = StreamingKMeans(decayFactor=0.0, k=2)
    stkm.setInitialCenters([[0.0], [1.0]], [1.0, 1.0])

    # Since decay factor is set to zero, once the first batch
    # is passed the clusterCenters are updated to [-0.5, 0.7]
    # which causes 0.2 & 0.3 to be classified as 1, even though the
    # classification based in the initial model would have been 0
    # proving that the model is updated.

    # TRAINING DATA

    # (1) original:
    # The below "batches" with 2 sets of elements is the original one from the test,
    # and running with it produces the expected results.
    # batches = [[[-0.5], [0.6], [0.8]], [[0.2], [-0.1], [0.3]]]

    # (2a) batch 1 only: This produces the same results as the original.
    # batches = [[[-0.5], [0.6], [0.8]]]

    # (2b) batch 2 only: This produces the results from the test failure.
    batches = [[[0.2], [-0.1], [0.3]]]

    batches = [sc.parallelize(batch) for batch in batches]
    input_stream = ssc.queueStream(batches)

    # TEST DATA (same as original training data)

    pBatches = batches = [[[-0.5], [0.6], [0.8]], [[0.2], [-0.1], [0.3]]]
    pBatches = [sc.parallelize(batch) for batch in pBatches]
    p_input_stream = ssc.queueStream(pBatches)

    # RUN STREAMING JOB

    predict_results = []

    def collect(rdd):
        rdd_collect = rdd.collect()
        if rdd_collect:
            predict_results.append(rdd_collect)

    stkm.trainOn(input_stream)
    predict_stream = stkm.predictOn(p_input_stream)
    predict_stream.foreachRDD(collect)

    ssc.start()
    time.sleep(6)
    ssc.stop(stopSparkContext=True, stopGraceFully=True)

    # PRINT STUFF

    for r in predict_results:
        print ", ".join(map(lambda x: str(x), r)) + "\n"
