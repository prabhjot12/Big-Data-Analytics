import re
import operator
from pyspark.sql import SQLContext, functions, types

sqlCt = SparkSession.builder.appName('Entity Resolution').getOrCreate()


class EntityResolution:
    def __init__(self, dataFile1, dataFile2, stopWordsFile):
        self.f = open(stopWordsFile, "r")
        self.stopWords = set(self.f.read().split("\n"))
        self.stopWordsBC = sc.broadcast(self.stopWords).value
        self.df1 = sqlCt.read.parquet(dataFile1).cache()
        self.df2 = sqlCt.read.parquet(dataFile2).cache()

    def preprocessDF(self, df, cols):
        stop = self.stopWordsBC

        def tokenizer(column):
            tokens = re.split(r'\W+', column)
            tokenList = []
            for token in tokens:
                if not ((token in ["", " ", "."]) | (token in stop)):
                    tokenList.append(token.lower())
            return tokenList

        df.createOrReplaceTempView("dataframe")
        newDF = sqlCt.sql("select *,concat(" + cols[0] + ", ' ', " + cols[1] + ") as Key from dataframe")
        my_udf_token = functions.udf(tokenizer, returnType=types.StringType())
        newDF = newDF.withColumn('joinKey', my_udf_token('Key')).drop('Key')
        return newDF

    def filtering(self, df1, df2):
        def formatter(column):
            val = str(column).strip("[")
            val = val
            return val.strip("]")

        def intConv(column):
            return column.strip(" ' ")

        my_udf_format = functions.udf(formatter, returnType=types.StringType())
        trimmedDf1 = df1.withColumn('formatted', my_udf_format('joinKey')).withColumnRenamed("id", "id1")
        trimmedDf2 = df2.withColumn('formatted', my_udf_format('joinKey')).withColumnRenamed("id", "id2")
        df1_compare = trimmedDf1.select('id1', 'joinKey', functions.explode \
            (functions.split(trimmedDf1.formatted, ","))).withColumnRenamed("joinKey", "joinKey1")
        df2_compare = trimmedDf2.select('id2', 'joinKey', functions.explode \
            (functions.split(trimmedDf2.formatted, ","))).withColumnRenamed("joinKey", "joinKey2")
        my_udf_int = functions.udf(intConv, returnType=types.StringType())
        df1_compare = df1_compare.withColumn('value', my_udf_int('col'))
        df2_compare = df2_compare.withColumn('value', my_udf_int('col'))
        joined = df1_compare.join(df2_compare, "value", "inner")
        candDF = joined.select('id1', 'joinKey1', 'id2', 'joinKey2')
        candDF = candDF.dropDuplicates()
        return candDF

    def verification(self, candDF, threshold):
        def setConv(column):
            column = (column.strip('[')).strip(']')
            val = column.split("][")
            jk1 = val[0].split(",")
            jk2 = val[1].split(",")
            set_temp1 = set()
            set_temp2 = set()
            for i in jk1:
                set_temp1.add(i.strip(" ' "))
            for j in jk2:
                set_temp2.add(j.strip(" ' "))
            return len(set_temp1.intersection(set_temp2)) / len(set_temp1.union(set_temp2))

        candDF.createOrReplaceTempView('cand')
        temp = sqlCt.sql('''select *,concat(joinKey1,joinKey2) as joined from cand''')
        my_udf_set = functions.udf(setConv, returnType=types.FloatType())
        temp = temp.withColumn("jaccard", my_udf_set("joined"))
        temp.createOrReplaceTempView('temp')
        resultDF = sqlCt.sql("select id1,joinKey1,id2,joinKey2,jaccard from temp where jaccard>= " + str(threshold))
        return resultDF

    def evaluate(self, result, groundTruth):
        T = 0
        R = len(result)
        A = len(groundTruth)
        for i in result:
            if (i in groundTruth):
                T += 1
        precision = T / R
        recall = T / A
        Fmeasure = (2 * precision * recall) / (precision + recall)
        return precision, recall, Fmeasure

    def jaccardJoin(self, cols1, cols2, threshold):

        newDF1 = self.preprocessDF(self.df1, cols1)
        newDF2 = self.preprocessDF(self.df2, cols2)
        print("Before filtering: %d pairs in total" % (self.df1.count() * self.df2.count()))

        candDF = self.filtering(newDF1, newDF2)
        print("After Filtering: %d pairs left" % (candDF.count()))

        resultDF = self.verification(candDF, threshold)
        print("After Verification: %d similar pairs" % (resultDF.count()))

        return resultDF

    def __del__(self):
        self.f.close()


if __name__ == "__main__":
    er = EntityResolution("Amazon_sample/part-r-*", "Google_sample/part-r-*", "stopwords.txt")
    amazonCols = ["title", "manufacturer"]
    googleCols = ["name", "manufacturer"]
    resultDF = er.jaccardJoin(amazonCols, googleCols, 0.5)
    result = resultDF.rdd.map(lambda row: (row.id1, row.id2)).collect()
    groundTruth = sqlCt.read.parquet("Amazon_Google_perfectMapping_sample/part-r-*") \
        .rdd.map(lambda row: (row.idAmazon, row.idGoogle)).collect()
    print("(precision, recall, fmeasure) = ", er.evaluate(result, groundTruth))



    # if __name__ == "__main__":
    #     er = EntityResolution("Amazon", "Google", "stopwords.txt")
    #     amazonCols = ["title", "manufacturer"]
    #     googleCols = ["name", "manufacturer"]
    #     resultDF = er.jaccardJoin(amazonCols, googleCols, 0.5)
    #     result = resultDF.rdd.map(lambda row: (row.id1, row.id2)).collect()
    #     groundTruth = spark.read.parquet("Amazon_Google_perfectMapping") \
    #                           .rdd.map(lambda row: (row.idAmazon, row.idGoogle)).collect()
    #     print ("(precision, recall, fmeasure) = ", er.evaluate(result, groundTruth))
