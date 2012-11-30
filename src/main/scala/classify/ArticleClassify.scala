package classify

import io.Source
import breeze.classify.NaiveBayes
import breeze.linalg.Counter
import breeze.data.Example
import math._

/**
 * Classifies NY Times articles previously fetched via [[classify.NYTimesPull]]. Uses a naive
 * Bayes classifier from the <a href="https://github.com/dlwh/breeze">Breeze</a> library.
 *
 * Inspired by Hilary Mason's http://github.com/hmason/ml_class/blob/master/intro_web_data/classify.py
 */
object ArticleClassify extends App {

  val PrettyPrint = "\n%s \nClassification probability for '%s'\n%s"

  // label constants (also corresponding to file names)
  val Arts = "arts"
  val Sports = "sports"

  // creating & training classifier
  val trainingData = createExamples(List(Arts, Sports))
  val classifier = new NaiveBayes.Trainer().train(trainingData)

  // classifying arbitrary CNN sports example
  val SportsSentence = "Though Gang Green's fans may have reached a breaking point, Bart Scott isn't willing to let them say whatever they want."
  val sportsTestData = createCounter(SportsSentence)
  assert(classifier.classify(sportsTestData) == Sports)
  val classifiedSportsProb = normalizeScores(classifier.scores(sportsTestData)).get(Sports).get
  println(PrettyPrint.format(Sports.toUpperCase, SportsSentence, classifiedSportsProb))

  // classifying arbitrary Guardian arts example
  val ArtsSentence = "Monet painted the olive trees and described the gardens at Villa Mariani in Bordighera, Italy, as 'pure magic.'"
  val artsTestData = createCounter(ArtsSentence)
  assert(classifier.classify(artsTestData) == Arts)
  val classifiedArtsProb = normalizeScores(classifier.scores(artsTestData)).get(Arts).get
  println(PrettyPrint.format(Arts.toUpperCase, ArtsSentence, classifiedArtsProb))

  def createExamples(labels: Seq[String]) = {
    for {
      label <- labels
      counter = createCounter(readFromFile("/" + label))
    } yield {
      Example(label, counter)
    }
  }

  def readFromFile(name: String) = Source.fromURL(getClass.getResource(name)).mkString

  def createCounter(s: String) = {
    val langData = breeze.text.LanguagePack.English
    val tokenizer = langData.simpleTokenizer
    val stemmer = langData.stemmer.getOrElse(identity[String] _)
    val tokens = tokenizer(s).map(_.toLowerCase)
    val tokenStems = tokens.map(stemmer)
    Counter.count[String](tokenStems).mapValues(_.toDouble)
  }

  def normalizeScores(scores: Counter[String, Double]) = {
    val nonLogScores = scores.mapValues(-exp(_))
    val scoreSum = nonLogScores.sum
    val normalizedScores = nonLogScores.mapValues(_ / scoreSum)
    normalizedScores
  }

}
