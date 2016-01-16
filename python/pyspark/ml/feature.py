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

import sys
if sys.version > '3':
    basestring = str

from pyspark import since
from pyspark.rdd import ignore_unicode_prefix
from pyspark.ml.param.shared import *
from pyspark.ml.util import keyword_only
from pyspark.ml.wrapper import JavaEstimator, JavaModel, JavaTransformer, _jvm
from pyspark.mllib.common import inherit_doc
from pyspark.mllib.linalg import _convert_to_vector

__all__ = ['HashingTF', 'StandardScaler', 'StandardScalerModel', 'StringIndexer',
           'StringIndexerModel', 'Tokenizer']


@inherit_doc
class HashingTF(JavaTransformer, HasInputCol, HasOutputCol, HasNumFeatures):
    """
    .. note:: Experimental

    Maps a sequence of terms to their term frequencies using the
    hashing trick.

    >>> df = sqlContext.createDataFrame([(["a", "b", "c"],)], ["words"])
    >>> hashingTF = HashingTF(numFeatures=10, inputCol="words", outputCol="features")
    >>> hashingTF.transform(df).head().features
    SparseVector(10, {7: 1.0, 8: 1.0, 9: 1.0})
    >>> hashingTF.setParams(outputCol="freqs").transform(df).head().freqs
    SparseVector(10, {7: 1.0, 8: 1.0, 9: 1.0})
    >>> params = {hashingTF.numFeatures: 5, hashingTF.outputCol: "vector"}
    >>> hashingTF.transform(df, params).head().vector
    SparseVector(5, {2: 1.0, 3: 1.0, 4: 1.0})

    .. versionadded:: 1.3.0
    """

    @keyword_only
    def __init__(self, numFeatures=1 << 18, inputCol=None, outputCol=None):
        """
        __init__(self, numFeatures=1 << 18, inputCol=None, outputCol=None)
        """
        super(HashingTF, self).__init__()
        self._java_obj = self._new_java_obj("org.apache.spark.ml.feature.HashingTF", self.uid)
        self._setDefault(numFeatures=1 << 18)
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    @since("1.3.0")
    def setParams(self, numFeatures=1 << 18, inputCol=None, outputCol=None):
        """
        setParams(self, numFeatures=1 << 18, inputCol=None, outputCol=None)
        Sets params for this HashingTF.
        """
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)


@inherit_doc
class StandardScaler(JavaEstimator, HasInputCol, HasOutputCol):
    """
    .. note:: Experimental

    Standardizes features by removing the mean and scaling to unit variance using column summary
    statistics on the samples in the training set.

    >>> from pyspark.mllib.linalg import Vectors
    >>> df = sqlContext.createDataFrame([(Vectors.dense([0.0]),), (Vectors.dense([2.0]),)], ["a"])
    >>> standardScaler = StandardScaler(inputCol="a", outputCol="scaled")
    >>> model = standardScaler.fit(df)
    >>> model.mean
    DenseVector([1.0])
    >>> model.std
    DenseVector([1.4142])
    >>> model.transform(df).collect()[1].scaled
    DenseVector([1.4142])

    .. versionadded:: 1.4.0
    """

    # a placeholder to make it appear in the generated doc
    withMean = Param(Params._dummy(), "withMean", "Center data with mean")
    withStd = Param(Params._dummy(), "withStd", "Scale to unit standard deviation")

    @keyword_only
    def __init__(self, withMean=False, withStd=True, inputCol=None, outputCol=None):
        """
        __init__(self, withMean=False, withStd=True, inputCol=None, outputCol=None)
        """
        super(StandardScaler, self).__init__()
        self._java_obj = self._new_java_obj("org.apache.spark.ml.feature.StandardScaler", self.uid)
        self.withMean = Param(self, "withMean", "Center data with mean")
        self.withStd = Param(self, "withStd", "Scale to unit standard deviation")
        self._setDefault(withMean=False, withStd=True)
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    @since("1.4.0")
    def setParams(self, withMean=False, withStd=True, inputCol=None, outputCol=None):
        """
        setParams(self, withMean=False, withStd=True, inputCol=None, outputCol=None)
        Sets params for this StandardScaler.
        """
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    @since("1.4.0")
    def setWithMean(self, value):
        """
        Sets the value of :py:attr:`withMean`.
        """
        self._paramMap[self.withMean] = value
        return self

    @since("1.4.0")
    def getWithMean(self):
        """
        Gets the value of withMean or its default value.
        """
        return self.getOrDefault(self.withMean)

    @since("1.4.0")
    def setWithStd(self, value):
        """
        Sets the value of :py:attr:`withStd`.
        """
        self._paramMap[self.withStd] = value
        return self

    @since("1.4.0")
    def getWithStd(self):
        """
        Gets the value of withStd or its default value.
        """
        return self.getOrDefault(self.withStd)

    def _create_model(self, java_model):
        return StandardScalerModel(java_model)


class StandardScalerModel(JavaModel):
    """
    .. note:: Experimental

    Model fitted by StandardScaler.

    .. versionadded:: 1.4.0
    """

    @property
    @since("1.5.0")
    def std(self):
        """
        Standard deviation of the StandardScalerModel.
        """
        return self._call_java("std")

    @property
    @since("1.5.0")
    def mean(self):
        """
        Mean of the StandardScalerModel.
        """
        return self._call_java("mean")


@inherit_doc
class StringIndexer(JavaEstimator, HasInputCol, HasOutputCol, HasHandleInvalid):
    """
    .. note:: Experimental

    A label indexer that maps a string column of labels to an ML column of label indices.
    If the input column is numeric, we cast it to string and index the string values.
    The indices are in [0, numLabels), ordered by label frequencies.
    So the most frequent label gets index 0.

    >>> stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
    >>> model = stringIndexer.fit(stringIndDf)
    >>> td = model.transform(stringIndDf)
    >>> sorted(set([(i[0], i[1]) for i in td.select(td.id, td.indexed).collect()]),
    ...     key=lambda x: x[0])
    [(0, 0.0), (1, 2.0), (2, 1.0), (3, 0.0), (4, 0.0), (5, 1.0)]
    >>> inverter = IndexToString(inputCol="indexed", outputCol="label2", labels=model.labels())
    >>> itd = inverter.transform(td)
    >>> sorted(set([(i[0], str(i[1])) for i in itd.select(itd.id, itd.label2).collect()]),
    ...     key=lambda x: x[0])
    [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'a'), (4, 'a'), (5, 'c')]

    .. versionadded:: 1.4.0
    """

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, handleInvalid="error"):
        """
        __init__(self, inputCol=None, outputCol=None, handleInvalid="error")
        """
        super(StringIndexer, self).__init__()
        self._java_obj = self._new_java_obj("org.apache.spark.ml.feature.StringIndexer", self.uid)
        self._setDefault(handleInvalid="error")
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    @since("1.4.0")
    def setParams(self, inputCol=None, outputCol=None, handleInvalid="error"):
        """
        setParams(self, inputCol=None, outputCol=None, handleInvalid="error")
        Sets params for this StringIndexer.
        """
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def _create_model(self, java_model):
        return StringIndexerModel(java_model)


class StringIndexerModel(JavaModel):
    """
    .. note:: Experimental

    Model fitted by StringIndexer.

    .. versionadded:: 1.4.0
    """
    @property
    @since("1.5.0")
    def labels(self):
        """
        Ordered list of labels, corresponding to indices to be assigned.
        """
        return self._java_obj.labels


@inherit_doc
class IndexToString(JavaTransformer, HasInputCol, HasOutputCol):
    """
    .. note:: Experimental

    A :py:class:`Transformer` that maps a column of indices back to a new column of
    corresponding string values.
    The index-string mapping is either from the ML attributes of the input column,
    or from user-supplied labels (which take precedence over ML attributes).
    See L{StringIndexer} for converting strings into indices.

    .. versionadded:: 1.6.0
    """

    # a placeholder to make the labels show up in generated doc
    labels = Param(Params._dummy(), "labels",
                   "Optional array of labels specifying index-string mapping." +
                   " If not provided or if empty, then metadata from inputCol is used instead.")

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, labels=None):
        """
        __init__(self, inputCol=None, outputCol=None, labels=None)
        """
        super(IndexToString, self).__init__()
        self._java_obj = self._new_java_obj("org.apache.spark.ml.feature.IndexToString",
                                            self.uid)
        self.labels = Param(self, "labels",
                            "Optional array of labels specifying index-string mapping. If not" +
                            " provided or if empty, then metadata from inputCol is used instead.")
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    @since("1.6.0")
    def setParams(self, inputCol=None, outputCol=None, labels=None):
        """
        setParams(self, inputCol=None, outputCol=None, labels=None)
        Sets params for this IndexToString.
        """
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    @since("1.6.0")
    def setLabels(self, value):
        """
        Sets the value of :py:attr:`labels`.
        """
        self._paramMap[self.labels] = value
        return self

    @since("1.6.0")
    def getLabels(self):
        """
        Gets the value of :py:attr:`labels` or its default value.
        """
        return self.getOrDefault(self.labels)


@inherit_doc
@ignore_unicode_prefix
class Tokenizer(JavaTransformer, HasInputCol, HasOutputCol):
    """
    .. note:: Experimental

    A tokenizer that converts the input string to lowercase and then
    splits it by white spaces.

    >>> df = sqlContext.createDataFrame([("a b c",)], ["text"])
    >>> tokenizer = Tokenizer(inputCol="text", outputCol="words")
    >>> tokenizer.transform(df).head()
    Row(text=u'a b c', words=[u'a', u'b', u'c'])
    >>> # Change a parameter.
    >>> tokenizer.setParams(outputCol="tokens").transform(df).head()
    Row(text=u'a b c', tokens=[u'a', u'b', u'c'])
    >>> # Temporarily modify a parameter.
    >>> tokenizer.transform(df, {tokenizer.outputCol: "words"}).head()
    Row(text=u'a b c', words=[u'a', u'b', u'c'])
    >>> tokenizer.transform(df).head()
    Row(text=u'a b c', tokens=[u'a', u'b', u'c'])
    >>> # Must use keyword arguments to specify params.
    >>> tokenizer.setParams("text")
    Traceback (most recent call last):
        ...
    TypeError: Method setParams forces keyword arguments.

    .. versionadded:: 1.3.0
    """

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        """
        __init__(self, inputCol=None, outputCol=None)
        """
        super(Tokenizer, self).__init__()
        self._java_obj = self._new_java_obj("org.apache.spark.ml.feature.Tokenizer", self.uid)
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    @since("1.3.0")
    def setParams(self, inputCol=None, outputCol=None):
        """
        setParams(self, inputCol="input", outputCol="output")
        Sets params for this Tokenizer.
        """
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)


if __name__ == "__main__":
    import doctest
    from pyspark.context import SparkContext
    from pyspark.sql import Row, SQLContext
    globs = globals().copy()
    # The small batch size here ensures that we see multiple batches,
    # even in these small test examples:
    sc = SparkContext("local[2]", "ml.feature tests")
    sqlContext = SQLContext(sc)
    globs['sc'] = sc
    globs['sqlContext'] = sqlContext
    testData = sc.parallelize([Row(id=0, label="a"), Row(id=1, label="b"),
                               Row(id=2, label="c"), Row(id=3, label="a"),
                               Row(id=4, label="a"), Row(id=5, label="c")], 2)
    globs['stringIndDf'] = sqlContext.createDataFrame(testData)
    (failure_count, test_count) = doctest.testmod(globs=globs, optionflags=doctest.ELLIPSIS)
    sc.stop()
    if failure_count:
        exit(-1)
