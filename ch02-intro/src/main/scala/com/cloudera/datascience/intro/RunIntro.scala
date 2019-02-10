/*
 * Copyright 2015 and onwards Sanford Ryza, Uri Laserson, Sean Owen and Joshua Wills
 *
 * See LICENSE file for further information.
 */

package com.cloudera.datascience.intro

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions._ // for lit(), first(), etc.

case class MatchData(
                      id_1: Int,
                      id_2: Int,
                      cmp_fname_c1: Option[Double],
                      cmp_fname_c2: Option[Double],
                      cmp_lname_c1: Option[Double],
                      cmp_lname_c2: Option[Double],
                      cmp_sex: Option[Int],
                      cmp_bd: Option[Int],
                      cmp_bm: Option[Int],
                      cmp_by: Option[Int],
                      cmp_plz: Option[Int],
                      is_match: Boolean
                    )

object RunIntro extends Serializable {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .config("spark.master", "local[*]")
      .config("spark.executor.memory", "6g")
      .appName("Intro")
      .getOrCreate
    import spark.implicits._

    spark.sparkContext.setLogLevel("ERROR")

    val preview = spark.read.csv("ch02-intro/linkage/block_*.csv")
    println("InferSchema false")
    preview.show()
    preview.printSchema()

    val parsed = spark.read
      .option("header", "true")
      .option("nullValue", "?")
      .option("inferSchema", "true")
      .csv("ch02-intro/linkage/block_*.csv")
    println("InferSchema true")
    parsed.show()
    parsed.printSchema()

    parsed.count()
    parsed.cache()
    println("Count match")
    parsed.groupBy("is_match").count().orderBy($"count".desc).show()

    println("Count match")
    parsed.createOrReplaceTempView("linkage")
    spark.sql(
      """
      SELECT is_match, COUNT(*) cnt
      FROM linkage
      GROUP BY is_match
      ORDER BY cnt DESC
    """).show()

    val summary = parsed.describe()

    println("Summary")
    summary.show()
    println("Summary")
    summary.select("summary", "cmp_fname_c1", "cmp_fname_c2").show()

    val matches = parsed.where("is_match = true")
    val misses = parsed.filter($"is_match" === false)
    val matchSummary = matches.describe()
    val missSummary = misses.describe()

    println("Summary match true")
    matchSummary.show()
    println("Summary match false")
    missSummary.show()

    val matchSummaryT = pivotSummary(matchSummary)
    val missSummaryT = pivotSummary(missSummary)

    println("Summary pivotal match true")
    matchSummaryT.show()
    println("Summary pivotal match false")
    missSummaryT.show()

    matchSummaryT.createOrReplaceTempView("match_desc")
    missSummaryT.createOrReplaceTempView("miss_desc")
    println("Join Summary pivotal match true and Summary pivotal match false\n" +
      "in order to find best features to classify on.\n" +
      "Good feature has two properties:\n" +
      "It tends to have significantly different values for nmatches and nonmatches\n" +
      "so col(delta) difference between two means will be large\n" +
      "And it occurs often enough in the data col(total) that we can rely on\n" +
      "SELECT a.field, a.count + b.count total, a.mean - b.mean delta\n"+
      "FROM match_desc a INNER JOIN miss_desc b ON a.field = b.field\n"+
      "WHERE a.field NOT IN (id_1, id_2)\n"+
      "ORDER BY delta DESC, total DESC")
    spark.sql(
      """
      SELECT a.field, a.count + b.count total, a.mean - b.mean delta
      FROM match_desc a INNER JOIN miss_desc b ON a.field = b.field
      WHERE a.field NOT IN ("id_1", "id_2")
      ORDER BY delta DESC, total DESC
    """).show()

    val matchData = parsed.as[MatchData]
    val scored = matchData.map { md =>
      (scoreMatchData(md), md.is_match)
    }.toDF("score", "is_match")

    println("Score:\n" +
      "Sums of the values of the obviously good features\n" +
      "cmp_plz,cmp_by, cmp_bd, cmp_lname_c1, and cmp_bm.")
    scored.orderBy($"score".desc).show(5)
    scored.orderBy($"score".asc).show(5)

    println("Count match")
    parsed.groupBy("is_match").count().orderBy($"count".desc).show()

    println("Cross tabulation\n" +
      "show trade of false-negative and false-positive\n" +
      "Applying the lower threshold of 2.0,\n" +
      "we can ensure that we capture all of the known matching records,\n" +
      "but at a substantial cost in terms of false positive (top-right cell)\n" +
      "")

    crossTabs(scored, 2.0).show()
  }

  def crossTabs(scored: DataFrame, t: Double): DataFrame = {
    scored.
      selectExpr(s"score >= $t as above", "is_match").
      groupBy("above").
      pivot("is_match", Seq("true", "false")).
      count()
  }

  case class Score(value: Double) {
    def +(oi: Option[Int]) = {
      Score(value + oi.getOrElse(0))
    }
  }

  def scoreMatchData(md: MatchData): Double = {
    (Score(md.cmp_lname_c1.getOrElse(0.0)) + md.cmp_plz +
      md.cmp_by + md.cmp_bd + md.cmp_bm).value
  }

  def pivotSummary(desc: DataFrame): DataFrame = {
    val lf = longForm(desc)
    println("Long form")
    lf.show()
    lf.groupBy("field").
      pivot("metric", Seq("count", "mean", "stddev", "min", "max")).
      agg(first("value"))
  }

  def longForm(desc: DataFrame): DataFrame = {
    import desc.sparkSession.implicits._ // For toDF RDD -> DataFrame conversion
    val schema = desc.schema
    desc.flatMap(row => {
      val metric = row.getString(0)
      (1 until row.size).map(i => (metric, schema(i).name, row.getString(i).toDouble))
    })
      .toDF("metric", "field", "value")
  }
}
