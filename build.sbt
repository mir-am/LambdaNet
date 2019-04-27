name := "GradualTypingWithRL"

version := "0.1"


organization in ThisBuild := "mrvplusone.github.io"
scalaVersion in ThisBuild := "2.12.7"

resolvers in ThisBuild += Resolver.sonatypeRepo("snapshots")
val osClassifier = System.getProperty("os.name") match {
  case "Mac OS X" => "darwin-cpu-x86_64"
  case "Linux" => "linux-cpu-x86_64"
  case _ => throw new Error("Platform does not support tensorflow-scala.")
}

libraryDependencies ++= Seq(
  "com.lihaoyi" %% "fastparse" % "2.0.4",
  "org.scalacheck" %% "scalacheck" % "1.14.0",
  "org.scalatest" %% "scalatest" % "3.0.3" % Test,

  "ch.qos.logback" % "logback-classic" % "1.2.3",
  "com.typesafe.akka" %% "akka-actor" % "2.5.12",
  "com.typesafe.akka" %% "akka-testkit" % "2.5.12" % Test,
  "com.lihaoyi" %% "ammonite-ops" % "1.0.3",
  "org.json4s" %% "json4s-native" % "3.6.3",

  "com.github.daddykotex" %% "courier" % "1.0.0",
  
  //  "be.botkop" %% "numsca" % "0.1.4",
  // for building numsca
  "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta3",
  "com.typesafe.scala-logging" %% "scala-logging" % "3.7.2",

  // tensorflow dependencies
  "org.platanios" %% "tensorflow" % "0.4.2-SNAPSHOT" classifier osClassifier,

)