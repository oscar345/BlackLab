(window.webpackJsonp=window.webpackJsonp||[]).push([[22],{296:function(t,e,a){"use strict";a.r(e);var s=a(14),r=Object(s.a)({},(function(){var t=this,e=t._self._c;return e("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[e("h1",{attrs:{id:"downloads"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#downloads"}},[t._v("#")]),t._v(" Downloads")]),t._v(" "),e("h2",{attrs:{id:"releases"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#releases"}},[t._v("#")]),t._v(" Releases")]),t._v(" "),e("p",[t._v("See the "),e("a",{attrs:{href:"https://github.com/INL/BlackLab/releases/",target:"_blank",rel:"noopener noreferrer"}},[t._v("GitHub releases page"),e("OutboundLink")],1),t._v(" for the complete list. This may also include development versions you can try out. If you're looking for the BlackLab library or commandline tools (i.e. not BlackLab Server), choose the version with libraries included.")]),t._v(" "),e("p",[t._v("Also see the "),e("RouterLink",{attrs:{to:"/development/changelog.html"}},[t._v("Change log")]),t._v(".")],1),t._v(" "),e("h2",{attrs:{id:"versions-and-compatibility"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#versions-and-compatibility"}},[t._v("#")]),t._v(" Versions and compatibility")]),t._v(" "),e("p",[t._v("Here's a list of BlackLab versions with their minimum Java version, the Lucene version they use and\nthe corpora they support.")]),t._v(" "),e("p",[t._v("The reason for not all older corpora being usable with a newer BlackLab version is mostly that Lucene drops support for older index formats.")]),t._v(" "),e("table",[e("thead",[e("tr",[e("th",[t._v("BlackLab")]),t._v(" "),e("th",{staticStyle:{"text-align":"left"}},[t._v("1st release")]),t._v(" "),e("th",[t._v("Java")]),t._v(" "),e("th",[t._v("Lucene")]),t._v(" "),e("th",[t._v("Solr")]),t._v(" "),e("th",[t._v("Supports corpora...")])])]),t._v(" "),e("tbody",[e("tr",[e("td",[t._v("4.x (future)")]),t._v(" "),e("td",{staticStyle:{"text-align":"left"}},[t._v("future")]),t._v(" "),e("td",[t._v("likely 11")]),t._v(" "),e("td",[t._v("likely 9")]),t._v(" "),e("td",[t._v("likely 9")]),t._v(" "),e("td",[t._v("created with BL 3-4")])]),t._v(" "),e("tr",[e("td",[t._v("3.x")]),t._v(" "),e("td",{staticStyle:{"text-align":"left"}},[t._v("Jul 2022")]),t._v(" "),e("td",[t._v("11")]),t._v(" "),e("td",[t._v("8")]),t._v(" "),e("td",[t._v("-")]),t._v(" "),e("td",[t._v("created with BL 3")])]),t._v(" "),e("tr",[e("td",[t._v("2.x")]),t._v(" "),e("td",{staticStyle:{"text-align":"left"}},[t._v("Jan 2020")]),t._v(" "),e("td",[t._v("8")]),t._v(" "),e("td",[t._v("5")]),t._v(" "),e("td",[t._v("-")]),t._v(" "),e("td",[t._v("created with BL 1.2-2.x")])]),t._v(" "),e("tr",[e("td",[t._v("1.7-1.9 (obsolete)")]),t._v(" "),e("td",{staticStyle:{"text-align":"left"}},[t._v("Jun 2018")]),t._v(" "),e("td",[t._v("8")]),t._v(" "),e("td",[t._v("5")]),t._v(" "),e("td",[t._v("-")]),t._v(" "),e("td",[t._v("created with BL 1.2-2.x")])]),t._v(" "),e("tr",[e("td",[t._v("1.0-1.2 (obsolete)")]),t._v(" "),e("td",{staticStyle:{"text-align":"left"}},[t._v("Apr 2014")]),t._v(" "),e("td",[t._v("6")]),t._v(" "),e("td",[t._v("3/5")]),t._v(" "),e("td",[t._v("-")]),t._v(" "),e("td",[t._v("created with BL 1.x")])])])]),t._v(" "),e("p",[t._v("You can stay on 2.x for now to avoid reindexing your corpora, but you'll miss out on performance improvements and new features. We do appreciate any help backporting bugfixes to this version.")]),t._v(" "),e("h2",{attrs:{id:"build-your-own"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#build-your-own"}},[t._v("#")]),t._v(" Build your own")]),t._v(" "),e("p",[t._v("To download and build the development version (bleeding edge, may be unstable), clone the repository and build it\nusing Maven:")]),t._v(" "),e("div",{staticClass:"language-bash extra-class"},[e("pre",{pre:!0,attrs:{class:"language-bash"}},[e("code",[e("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" clone git://github.com/INL/BlackLab.git\n"),e("span",{pre:!0,attrs:{class:"token builtin class-name"}},[t._v("cd")]),t._v(" BlackLab\n"),e("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# git checkout dev    # (the default branch)")]),t._v("\nmvn clean package\n")])])]),e("p",[t._v("To instead build the most recent release of BlackLab yourself:")]),t._v(" "),e("div",{staticClass:"language-bash extra-class"},[e("pre",{pre:!0,attrs:{class:"language-bash"}},[e("code",[e("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" checkout main\nmvn clean package\n")])])])])}),[],!1,null,null,null);e.default=r.exports}}]);