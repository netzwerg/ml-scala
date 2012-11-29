package scala

import io.Source
import breeze.classify.NaiveBayes
import breeze.linalg.Counter
import breeze.data.Example
import math._

/**
 * Classifies NY Times articles previously fetched via [[scala.NYTimesPull]]. Uses naive
 * Bayes classifier from <a href="https://github.com/dlwh/breeze">Breeze</a> library.
 *
 * Inspired by Hilary Mason's http://github.com/hmason/ml_class/blob/master/intro_web_data/classify.py
 */
object Classify extends App {

  // label constants (also corresponding to file names)
  val Arts = "arts"
  val Sports = "sports"

  // creating & training classifier
  val trainingData = readExamples(List(Arts, Sports))
  val classifier = new NaiveBayes.Trainer().train(trainingData)

  // classifying arbitrary sports example
  val sportsTestData = Counter.count("philadelphia", "christmas").mapValues(_.toDouble)
  val classifiedSports = classifier.classify(sportsTestData)
  println(normalizeScores(classifier.scores(sportsTestData)))
  assert(classifiedSports == Sports)

  // classifying arbitrary arts example
  val artsTestData = Counter.count("melodrama", "society").mapValues(_.toDouble)
  val classifiedArts = classifier.classify(artsTestData)
  println(normalizeScores(classifier.scores(artsTestData)))
  assert(classifiedArts == Arts)

  def readExamples(labels: Seq[String]) = {

    val langData = breeze.text.LanguagePack.English
    val tokenizer = langData.simpleTokenizer
    val stemmer = langData.stemmer.getOrElse(identity[String]_)

    for {
      label <- labels
      tokens = tokenizer(Source.fromFile(label).mkString).map(_.toLowerCase)
      tokenStems = tokens.map(stemmer)
      counter = Counter.count[String](tokens).mapValues(_.toDouble)
    } yield {
      Example(label, counter)
    }
  }

  def normalizeScores(scores: Counter[_,Double]) = {
    val nonLogScores = scores.mapValues(-exp(_))
    val scoreSum = nonLogScores.sum
    val normalizedScores = nonLogScores.mapValues(_ / scoreSum)
    normalizedScores
  }

}
