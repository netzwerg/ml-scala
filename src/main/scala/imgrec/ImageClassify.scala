package imgrec

import breeze.classify.SVM
import breeze.linalg._
import classify.ArticleClassify

/**
 * Classifies images into crosses 'x' and circles 'o'. Uses a support vector
 * machine from the <a href="https://github.com/dlwh/breeze">Breeze</a> library.
 */
object ImageClassify extends App {

  val CircleLabel = "o"
  val CrossLabel = "x"

  val circleTrainingData = ImageReader.readTrainingExamples(CircleLabel)
  val crossTrainingData = ImageReader.readTrainingExamples(CrossLabel)

  val trainer = new SVM.SMOTrainer[String, DenseVector[Double]](100)
  val classifier = trainer.train(circleTrainingData ++ crossTrainingData)

  val circleTestData = ImageReader.readPixelVector("/o/0.png")
  println(ArticleClassify.normalizeScores(classifier.scores(circleTestData)))
  assert(classifier.classify(circleTestData) == CircleLabel)

  val crossTestData = ImageReader.readPixelVector("/x/0.png")
  println(ArticleClassify.normalizeScores(classifier.scores(crossTestData)))
  assert(classifier.classify(crossTestData) == CrossLabel)

}
