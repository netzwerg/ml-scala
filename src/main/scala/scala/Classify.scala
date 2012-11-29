package scala

import io.Source
import breeze.classify.NaiveBayes
import breeze.linalg.Counter
import breeze.data.Example

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
  assert(classifiedSports == Sports)

  // classifying arbitrary arts example
  val artsTestData = Counter.count("melodrama", "society").mapValues(_.toDouble)
  val classifiedArts = classifier.classify(artsTestData)
  assert(classifiedArts == Arts)

  def readExamples(labels: Seq[String]) = {

    val langData = breeze.text.LanguagePack.English
    val tokenizer = langData.simpleTokenizer

    for {
      label <- labels
      tokens = tokenizer(Source.fromFile(label).mkString).map(_.toLowerCase)
      counter = Counter.count[String](tokens).mapValues(_.toDouble)
    } yield {
      Example(label, counter)
    }
  }

}
