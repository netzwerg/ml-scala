package imgrec

import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import breeze.data.Example
import breeze.linalg.DenseVector

object ImageReader {

  def readTrainingExamples(label: String) = {
    for {
      fileIndex <- 1 to 9
    } yield {
      val fileName = "/" + label + "/" + fileIndex + ".png"
      val pixelVector = ImageReader.readPixelVector(fileName)
      Example(label, pixelVector)
    }
  }

  def readPixelVector(fileName: String): DenseVector[Double] = {
    val image = ImageIO.read(this.getClass.getResource(fileName))
    val pixels = extractPixelSeq(image).map(_.toDouble).toArray
    new DenseVector(pixels)
  }

  def extractPixelSeq(image: BufferedImage) = for {
    x <- 0 until image.getWidth
    y <- 0 until image.getHeight
    red = (image.getRGB(x, y) >> 16) & 0xFF
  } yield {
    if (red == 0) 0 else 1
  }

}
