(window.webpackJsonp=window.webpackJsonp||[]).push([[24],{298:function(t,a,e){"use strict";e.r(a);var n=e(14),s=Object(n.a)({},(function(){var t=this,a=t._self._c;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"migration-guide"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#migration-guide"}},[t._v("#")]),t._v(" Migration guide")]),t._v(" "),a("h2",{attrs:{id:"migrating-from-blacklab-1-7-to-2-0"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#migrating-from-blacklab-1-7-to-2-0"}},[t._v("#")]),t._v(" Migrating from BlackLab 1.7 to 2.0")]),t._v(" "),a("p",[t._v("BlackLab Server 2.0 contains one significant change to the API: metadata values are now always reported as lists, even if there is only one value. Because it is now possible to index multiple values for a single metadata field (for example, to associate two authors or two images with one document), this change was required.")]),t._v(" "),a("p",[t._v("Where BlackLab 2.0 does differ significantly is in its Java API and configuration files. If you use the Java API, for example because you've written your own DocIndexer class, this page helps you migrate it to 2.0.")]),t._v(" "),a("h3",{attrs:{id:"terminology"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#terminology"}},[t._v("#")]),t._v(" Terminology")]),t._v(" "),a("ul",[a("li",[t._v('"Searcher" -> "BlackLab index"')]),t._v(" "),a("li",[t._v('"complex field" -> "annotated field"')]),t._v(" "),a("li",[t._v('"property on a complex field" -> "annotation on an annotated field"')]),t._v(" "),a("li",[t._v('"an indexed alternative for a property" (e.g. case- and diacritics-insensitive) -> "an indexed sensitivity for an annotation"')])]),t._v(" "),a("p",[t._v('So, for example, an annotated field "contents" might have annotations "word", "lemma" and "pos" (part of speech), and the "word" annotation might have two sensitivities indexed: (case- and diacritics-) sensitive, and (case- and diacritics-) insensitive.')]),t._v(" "),a("h3",{attrs:{id:"migrating-the-configuration-file-s"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#migrating-the-configuration-file-s"}},[t._v("#")]),t._v(" Migrating the configuration file(s)")]),t._v(" "),a("p",[t._v("Usually you will use either a file "),a("code",[t._v("blacklab-server.yaml")]),t._v(" (for BlackLab Serer), or "),a("code",[t._v("blacklab.yaml")]),t._v(" (for e.g. IndexTool, QueryTool or other BlackLab applications). (JSON works too if you prefer)")]),t._v(" "),a("p",[t._v("A new, cleaner format was added in BlackLab 2.0. The old format still works, but it is a good idea to convert to the new format as the old format will eventually be removed.")]),t._v(" "),a("p",[t._v("For more information about the config file format, see "),a("RouterLink",{attrs:{to:"/server/configuration.html"}},[t._v("Configuration files")]),t._v(".")],1),t._v(" "),a("h3",{attrs:{id:"migrating-docindexers"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#migrating-docindexers"}},[t._v("#")]),t._v(" Migrating DocIndexers")]),t._v(" "),a("p",[t._v("If you have a custom implementation of DocIndexer for your own input format, please ensure that it has a default constructor. If instead if has a constructor that takes an "),a("code",[t._v("Indexer")]),t._v(", change "),a("code",[t._v("Indexer")]),t._v(" to "),a("code",[t._v("DocWriter")]),t._v(".")]),t._v(" "),a("h3",{attrs:{id:"method-naming"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#method-naming"}},[t._v("#")]),t._v(" Method naming")]),t._v(" "),a("p",[t._v("For many classes, methods were renamed from getSomeThing() to simply someThing(). While this may not be the convention in Java, it makes for less noisy, more natural-sounding code, especially when chaining methods. It also saves on typing. For example, compare these two examples:")]),t._v(" "),a("div",{staticClass:"language-java extra-class"},[a("pre",{pre:!0,attrs:{class:"language-java"}},[a("code",[a("span",{pre:!0,attrs:{class:"token class-name"}},[t._v("String")]),t._v(" luceneField "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" index\n    "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("getAnnotatedField")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"contents"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("getAnnotation")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"word"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("getSensitivity")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token class-name"}},[t._v("MatchSensitivity")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),a("span",{pre:!0,attrs:{class:"token constant"}},[t._v("SENSITIVE")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("getLuceneField")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token class-name"}},[t._v("String")]),t._v(" luceneField "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" index\n    "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("annotatedField")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"contents"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("annotation")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"word"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("sensitivity")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token class-name"}},[t._v("MatchSensitivity")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),a("span",{pre:!0,attrs:{class:"token constant"}},[t._v("SENSITIVE")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("luceneField")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n")])])]),a("h3",{attrs:{id:"important-renamed-packages-and-classes"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#important-renamed-packages-and-classes"}},[t._v("#")]),t._v(" Important renamed packages and classes")]),t._v(" "),a("p",[t._v("General:")]),t._v(" "),a("ul",[a("li",[t._v("Searcher -> BlackLabIndex")])]),t._v(" "),a("p",[t._v("Classes used while indexing:")]),t._v(" "),a("ul",[a("li",[t._v("ComplexField -> AnnotatedFieldWriter")]),t._v(" "),a("li",[t._v("ComplexFieldProperty -> AnnotationWriter")]),t._v(" "),a("li",[t._v("ComplexFieldUtil -> AnnotatedFieldNameUtil")])]),t._v(" "),a("p",[t._v("Index structure:")]),t._v(" "),a("ul",[a("li",[t._v("IndexStructure -> IndexMetadata")]),t._v(" "),a("li",[t._v("ComplexFieldDesc -> AnnotatedField")]),t._v(" "),a("li",[t._v("PropertyDesc -> Annotation")])]),t._v(" "),a("p",[t._v("Packages:")]),t._v(" "),a("ul",[a("li",[t._v("nl.inl.blacklab.search.indexstructure -> .search.indexmetadata")]),t._v(" "),a("li",[t._v("nl.inl.blacklab.externalstorage -> .contentstore")])]),t._v(" "),a("h3",{attrs:{id:"migrating-blacklab-programs"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#migrating-blacklab-programs"}},[t._v("#")]),t._v(" Migrating BlackLab programs")]),t._v(" "),a("p",[t._v("Methods:")]),t._v(" "),a("ul",[a("li",[t._v("instead of BlackLabIndex.open(), use BlackLab.open()")])])])}),[],!1,null,null,null);a.default=s.exports}}]);