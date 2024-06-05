(window.webpackJsonp=window.webpackJsonp||[]).push([[51],{328:function(t,e,o){"use strict";o.r(e);var n=o(14),r=Object(n.a)({},(function(){var t=this,e=t._self._c;return e("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[e("h1",{attrs:{id:"document-contents"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#document-contents"}},[t._v("#")]),t._v(" Document contents")]),t._v(" "),e("p",[t._v("Retrieve the original input document, presuming it's stored in the corpus (it is by default).")]),t._v(" "),e("p",[e("strong",[t._v("URL")]),t._v(" : "),e("code",[t._v("/blacklab-server/<corpus-name>/docs/<pid>/contents")])]),t._v(" "),e("p",[e("strong",[t._v("Method")]),t._v(" : "),e("code",[t._v("GET")])]),t._v(" "),e("h4",{attrs:{id:"parameters"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#parameters"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),e("p",[t._v("All parameters are optional.  "),e("code",[t._v("wordstart")]),t._v(" and "),e("code",[t._v("wordend")]),t._v(" refer to token position in a document, 0-based.")]),t._v(" "),e("p",[t._v("Partial contents XML output will be wrapped in "),e("code",[t._v("<blacklabResponse/>")]),t._v(" element to ensure a single root element.")]),t._v(" "),e("table",[e("thead",[e("tr",[e("th",[t._v("Parameter")]),t._v(" "),e("th",[t._v("Description")])])]),t._v(" "),e("tbody",[e("tr",[e("td",[e("code",[t._v("patt")])]),t._v(" "),e("td",[t._v("Pattern to highlight in the document. "),e("code",[t._v("<hl>...</hl>")]),t._v(" tags will be added to highlight hits.")])]),t._v(" "),e("tr",[e("td",[e("code",[t._v("field")])]),t._v(" "),e("td",[t._v("Annotated field to get snippet for.  (default: main annotated field)"),e("br"),t._v("("),e("strong",[t._v("NOTE:")]),t._v(" in case of a parallel corpus, you can shorten this to only the version, e.g. "),e("code",[t._v("nl")]),t._v(" if you want to get the contents for field "),e("code",[t._v("contents__nl")]),t._v(")")])]),t._v(" "),e("tr",[e("td",[e("code",[t._v("wordstart")])]),t._v(" "),e("td",[t._v("First word position we want returned. -1 for document start."),e("br"),e("strong",[t._v("NOTE:")]),t._v(" when greater than -1, any content before the first word will "),e("em",[t._v("not")]),t._v(" be included in the response!")])]),t._v(" "),e("tr",[e("td",[e("code",[t._v("wordend")])]),t._v(" "),e("td",[t._v("First word position we don't want returned. -1 for document end."),e("br"),e("strong",[t._v("NOTE:")]),t._v(" when greater than -1, any content after the last word will "),e("em",[t._v("not")]),t._v(" be included in the response!")])]),t._v(" "),e("tr",[e("td",[e("code",[t._v("adjusthits")])]),t._v(" "),e("td",[t._v("(relations queries only) should query hits be adjusted so all matched relations are inside the hit? Default: "),e("code",[t._v("no")])])])])]),t._v(" "),e("h2",{attrs:{id:"success-response"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#success-response"}},[t._v("#")]),t._v(" Success Response")]),t._v(" "),e("p",[e("strong",[t._v("Code")]),t._v(" : "),e("code",[t._v("200 OK")])]),t._v(" "),e("h3",{attrs:{id:"content-examples"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#content-examples"}},[t._v("#")]),t._v(" Content examples")]),t._v(" "),e("p",[e("em",[t._v("(the original input document, be it XML or some other format)")])])])}),[],!1,null,null,null);e.default=r.exports}}]);