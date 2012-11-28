package scala

import io.Source
import play.api.libs.json._
import java.io.PrintWriter

/**
 * Pulls 50 NY Times articles for "arts" and "sports" categories into dedicated files.
 *
 * Scala translation of Hilary Mason's http://github.com/hmason/ml_class/blob/master/intro_web_data/nytimes_pull.py
 */
object NYTimesPull extends App {

  val artsArticles = JsonReader.pull("[Top/Features/Arts]")
  FileWriter.dump("arts", artsArticles)

  val sportsArticles = JsonReader.pull("[Top/News/Sports]")
  FileWriter.dump("sports", sportsArticles)

}

object JsonReader {

  val ApiKey = "f7b4a1749764aec0364b215c354e3a0f:18:25759498"

  def pull10ArticlesByCategory(category: String, offset: Int) = {
    val Url = "http://api.nytimes.com/svc/search/v1/article?query=classifiers_facet:%s&api-key=%s&offset=%s".format(category, ApiKey, offset)
    Json.parse(Source.fromURL(Url).mkString)
  }

  def pull(category: String) = {
    val articleBodies = for {
      offset <- 0 until 5
      json = pull10ArticlesByCategory(category, offset)
      body = (json \ "results" \\ "body").map(_.as[String])
    } yield body
    articleBodies.flatten
  }

}

object FileWriter {

  def dump(fileName: String, contents: Seq[String]) {
    val out = new PrintWriter(fileName)
    contents.map(out.println(_))
  }

}