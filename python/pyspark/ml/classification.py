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

import warnings

from pyspark import since
from pyspark.ml.util import keyword_only
from pyspark.ml.wrapper import JavaEstimator, JavaModel
from pyspark.ml.param.shared import *
from pyspark.mllib.common import inherit_doc


__all__ = ['LogisticRegression', 'LogisticRegressionModel']


@inherit_doc
class LogisticRegression(JavaEstimator, HasFeaturesCol, HasLabelCol, HasPredictionCol, HasMaxIter,
                         HasRegParam, HasTol, HasProbabilityCol, HasRawPredictionCol,
                         HasElasticNetParam, HasFitIntercept, HasStandardization, HasThresholds,
                         HasWeightCol):
    """
    Logistic regression.
    Currently, this class only supports binary classification.

    >>> from pyspark.sql import Row
    >>> from pyspark.mllib.linalg import Vectors
    >>> df = sc.parallelize([
    ...     Row(label=1.0, weight=2.0, features=Vectors.dense(1.0)),
    ...     Row(label=0.0, weight=2.0, features=Vectors.sparse(1, [], []))]).toDF()
    >>> lr = LogisticRegression(maxIter=5, regParam=0.01, weightCol="weight")
    >>> model = lr.fit(df)
    >>> model.coefficients
    DenseVector([5.5...])
    >>> model.intercept
    -2.68...
    >>> test0 = sc.parallelize([Row(features=Vectors.dense(-1.0))]).toDF()
    >>> result = model.transform(test0).head()
    >>> result.prediction
    0.0
    >>> result.probability
    DenseVector([0.99..., 0.00...])
    >>> result.rawPrediction
    DenseVector([8.22..., -8.22...])
    >>> test1 = sc.parallelize([Row(features=Vectors.sparse(1, [0], [1.0]))]).toDF()
    >>> model.transform(test1).head().prediction
    1.0
    >>> lr.setParams("vector")
    Traceback (most recent call last):
        ...
    TypeError: Method setParams forces keyword arguments.

    .. versionadded:: 1.3.0
    """

    # a placeholder to make it appear in the generated doc
    threshold = Param(Params._dummy(), "threshold",
                      "Threshold in binary classification prediction, in range [0, 1]." +
                      " If threshold and thresholds are both set, they must match.")

    @keyword_only
    def __init__(self, featuresCol="features", labelCol="label", predictionCol="prediction",
                 maxIter=100, regParam=0.1, elasticNetParam=0.0, tol=1e-6, fitIntercept=True,
                 threshold=0.5, thresholds=None, probabilityCol="probability",
                 rawPredictionCol="rawPrediction", standardization=True, weightCol=None):
        """
        __init__(self, featuresCol="features", labelCol="label", predictionCol="prediction", \
                 maxIter=100, regParam=0.1, elasticNetParam=0.0, tol=1e-6, fitIntercept=True, \
                 threshold=0.5, thresholds=None, probabilityCol="probability", \
                 rawPredictionCol="rawPrediction", standardization=True, weightCol=None)
        If the threshold and thresholds Params are both set, they must be equivalent.
        """
        super(LogisticRegression, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.ml.classification.LogisticRegression", self.uid)
        #: param for threshold in binary classification, in range [0, 1].
        self.threshold = Param(self, "threshold",
                               "Threshold in binary classification prediction, in range [0, 1]." +
                               " If threshold and thresholds are both set, they must match.")
        self._setDefault(maxIter=100, regParam=0.1, tol=1E-6, threshold=0.5)
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)
        self._checkThresholdConsistency()

    @keyword_only
    @since("1.3.0")
    def setParams(self, featuresCol="features", labelCol="label", predictionCol="prediction",
                  maxIter=100, regParam=0.1, elasticNetParam=0.0, tol=1e-6, fitIntercept=True,
                  threshold=0.5, thresholds=None, probabilityCol="probability",
                  rawPredictionCol="rawPrediction", standardization=True, weightCol=None):
        """
        setParams(self, featuresCol="features", labelCol="label", predictionCol="prediction", \
                  maxIter=100, regParam=0.1, elasticNetParam=0.0, tol=1e-6, fitIntercept=True, \
                  threshold=0.5, thresholds=None, probabilityCol="probability", \
                  rawPredictionCol="rawPrediction", standardization=True, weightCol=None)
        Sets params for logistic regression.
        If the threshold and thresholds Params are both set, they must be equivalent.
        """
        kwargs = self.setParams._input_kwargs
        self._set(**kwargs)
        self._checkThresholdConsistency()
        return self

    def _create_model(self, java_model):
        return LogisticRegressionModel(java_model)

    @since("1.4.0")
    def setThreshold(self, value):
        """
        Sets the value of :py:attr:`threshold`.
        Clears value of :py:attr:`thresholds` if it has been set.
        """
        self._paramMap[self.threshold] = value
        if self.isSet(self.thresholds):
            del self._paramMap[self.thresholds]
        return self

    @since("1.4.0")
    def getThreshold(self):
        """
        Gets the value of threshold or its default value.
        """
        self._checkThresholdConsistency()
        if self.isSet(self.thresholds):
            ts = self.getOrDefault(self.thresholds)
            if len(ts) != 2:
                raise ValueError("Logistic Regression getThreshold only applies to" +
                                 " binary classification, but thresholds has length != 2." +
                                 "  thresholds: " + ",".join(ts))
            return 1.0/(1.0 + ts[0]/ts[1])
        else:
            return self.getOrDefault(self.threshold)

    @since("1.5.0")
    def setThresholds(self, value):
        """
        Sets the value of :py:attr:`thresholds`.
        Clears value of :py:attr:`threshold` if it has been set.
        """
        self._paramMap[self.thresholds] = value
        if self.isSet(self.threshold):
            del self._paramMap[self.threshold]
        return self

    @since("1.5.0")
    def getThresholds(self):
        """
        If :py:attr:`thresholds` is set, return its value.
        Otherwise, if :py:attr:`threshold` is set, return the equivalent thresholds for binary
        classification: (1-threshold, threshold).
        If neither are set, throw an error.
        """
        self._checkThresholdConsistency()
        if not self.isSet(self.thresholds) and self.isSet(self.threshold):
            t = self.getOrDefault(self.threshold)
            return [1.0-t, t]
        else:
            return self.getOrDefault(self.thresholds)

    def _checkThresholdConsistency(self):
        if self.isSet(self.threshold) and self.isSet(self.thresholds):
            ts = self.getParam(self.thresholds)
            if len(ts) != 2:
                raise ValueError("Logistic Regression getThreshold only applies to" +
                                 " binary classification, but thresholds has length != 2." +
                                 " thresholds: " + ",".join(ts))
            t = 1.0/(1.0 + ts[0]/ts[1])
            t2 = self.getParam(self.threshold)
            if abs(t2 - t) >= 1E-5:
                raise ValueError("Logistic Regression getThreshold found inconsistent values for" +
                                 " threshold (%g) and thresholds (equivalent to %g)" % (t2, t))


class LogisticRegressionModel(JavaModel):
    """
    Model fitted by LogisticRegression.

    .. versionadded:: 1.3.0
    """

    @property
    @since("1.4.0")
    def weights(self):
        """
        Model weights.
        """

        warnings.warn("weights is deprecated. Use coefficients instead.")
        return self._call_java("weights")

    @property
    @since("1.6.0")
    def coefficients(self):
        """
        Model coefficients.
        """
        return self._call_java("coefficients")

    @property
    @since("1.4.0")
    def intercept(self):
        """
        Model intercept.
        """
        return self._call_java("intercept")


if __name__ == "__main__":
    import doctest
    from pyspark.context import SparkContext
    from pyspark.sql import SQLContext
    globs = globals().copy()
    # The small batch size here ensures that we see multiple batches,
    # even in these small test examples:
    sc = SparkContext("local[2]", "ml.classification tests")
    sqlContext = SQLContext(sc)
    globs['sc'] = sc
    globs['sqlContext'] = sqlContext
    (failure_count, test_count) = doctest.testmod(
        globs=globs, optionflags=doctest.ELLIPSIS)
    sc.stop()
    if failure_count:
        exit(-1)
