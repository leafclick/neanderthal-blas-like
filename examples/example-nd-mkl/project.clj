(defproject neanderthal-blas-like/example-nd-mkl "0.5.0-SNAPSHOT"
  :description "BLAS-like Extensions for Neanderthal, example"
  :url "https://github.com/katox/neanderthal-blas-like"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.12.3"]
                 [org.uncomplicate/neanderthal-base "0.61.0"]
                 [org.uncomplicate/neanderthal-mkl "0.61.0"]
                 [org.bytedeco/mkl "2025.3-1.5.13" :classifier "linux-x86_64-redist"]
                 [com.leafclick/neanderthal-blas-like "0.2.0"]]
  :profiles {:java8 {:jvm-opts ^:replace ["-XX:+TieredCompilation" "-XX:TieredStopAtLevel=1"]}}

  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"]
  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                       "--enable-native-access=ALL-UNNAMED"]
  :source-paths ["src"])
