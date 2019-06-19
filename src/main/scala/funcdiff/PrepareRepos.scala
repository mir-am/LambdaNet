package funcdiff

import ammonite.ops._
import lambdanet.ProjectPath
import lambdanet.translation.ImportsResolution.{
  ErrorHandler,
  ModuleExports,
  PathMapping,
  SourceFileMissingError
}
import lambdanet.translation.{
  IRTranslation,
  ImportsResolution,
  PLangTranslation,
  PredicateGraphTranslation,
  QLangTranslation
}
import lambdanet.translation.PredicateGraph.PNodeAllocator
import lambdanet.utils.ProgramParsing
import lambdanet.utils.ProgramParsing.GProject

@SerialVersionUID(1)
case class LibDefs(
    baseCtx: ImportsResolution.ModuleExports,
    libAllocator: PNodeAllocator,
    libExports: Map[ProjectPath, ImportsResolution.ModuleExports]
)

object PrepareRepos {

  def parseLibDefs() = {
    val declarationsDir = pwd / up / "lambda-repos" / "declarations"

    println("parsing default module...")
    val (baseCtx, libAllocator, _) = QLangTranslation.parseDefaultModule()
    println("default module parsed")

    val libExports = {
      println("parsing library modules...")
      val GProject(_, modules, mapping, subProjects, devDependencies) =
        ProgramParsing
          .parseGProjectFromRoot(declarationsDir, declarationFileMod = true)

      println("parsing PModules...")
      val pModules =
        modules.map(m => PLangTranslation.fromGModule(m, libAllocator))

      println("imports resolution...")
      val handler = ErrorHandler(ErrorHandler.StoreError, ErrorHandler.StoreError)

      val exports = ImportsResolution.resolveExports(
        pModules,
        baseCtx,
        Map(),
        mapping,
        defaultPublicMode = true,
        errorHandler = handler,
        devDependencies,
        maxIterations = 5
      )

      val namedExports = subProjects.map {
        case (name, path) =>
          name -> exports.getOrElse(
            path,
            exports.getOrElse(
              path / "index", {
                Console.err.println(
                  s"Couldn't find Exports located at $path for $name, ignore this named project."
                )
                ModuleExports.empty
              }
            )
          )
      }
      handler.warnErrors()
      exports ++ namedExports
    }
    println("Declaration files parsed.")
    LibDefs(baseCtx, libAllocator, libExports)
  }

  def prepareProject(libDefs: LibDefs, root: Path) =
    SimpleMath.withErrorMessage(s"In project: $root") {
      import libDefs._

      val skipSet = Set("__tests__", "dist", "test", "tests")
      def filterTests(path: Path): Boolean = {
        path.segments.forall(!skipSet.contains(_))
      }

      val p = ProgramParsing.parseGProjectFromRoot(root, filter = filterTests)
      val allocator = new PNodeAllocator(forLib = false)
      val irTranslator = new IRTranslation(allocator)

      val errorHandler = ErrorHandler(ErrorHandler.ThrowError, ErrorHandler.StoreError)

//    println(s"LibExports key set: ${libExports.keySet}")
      val irModules = QLangTranslation
        .fromProject(
          p.modules,
          baseCtx,
          libExports,
          allocator,
          p.pathMapping,
          p.devDependencies,
          errorHandler
        )
        .map(irTranslator.fromQModule)
      val graph = PredicateGraphTranslation.fromIRModules(irModules)
      errorHandler.warnErrors()
      println(s"Project parsed: '$root'")
      graph.printStat()
      println(graph)
    }

  def main(args: Array[String]): Unit = {
    val loadFromFile = true //set to true to load declarations from the serialization file

    val libDefsFile = pwd / up / "lambda-repos" / "libDefs.serialized"

    val libDefs = if (loadFromFile) {
      println(s"loading library definitions from $libDefsFile...")
      val read = SimpleMath.readObjectFromFile[LibDefs](libDefsFile.toIO)
      println(s"library definitions loaded.")
      read
    } else {
      val defs = parseLibDefs()
      SimpleMath.saveObjectToFile(libDefsFile.toIO)(defs)
      println(s"library definitions saved to $libDefsFile")
      defs
    }

    val projectsDir = pwd / up / "lambda-repos" / "projects"

    (ls ! projectsDir).foreach(f => if (f.isDir) prepareProject(libDefs, f))
  }

}
