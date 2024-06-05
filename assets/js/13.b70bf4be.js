(window.webpackJsonp=window.webpackJsonp||[]).push([[13],{288:function(e,t,o){"use strict";o.r(t);var i=o(14),n=Object(i.a)({},(function(){var e=this,t=e._self._c;return t("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[t("h1",{attrs:{id:"blacklab-webservice-api-evolution"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#blacklab-webservice-api-evolution"}},[e._v("#")]),e._v(" Blacklab webservice API evolution")]),e._v(" "),t("div",{staticClass:"custom-block warning"},[t("p",{staticClass:"custom-block-title"},[e._v("OLDER CONTENT")]),e._v(" "),t("p",[e._v("This page contains ideas that are partially obsolete.\nSee "),t("RouterLink",{attrs:{to:"/server/rest-api/api-versions.html"}},[e._v("API versions")]),e._v(" for the current state of the API.")],1)]),e._v(" "),t("p",[e._v("The BLS API has quite a few quirks that can make it confusing and annoying to work with.")]),e._v(" "),t("p",[e._v("We intend to evolve the API over time, with new versions that gradually move away from the bad parts of the old API. This can be done using the "),t("code",[e._v("api")]),e._v(" parameter to switch between versions, or by adding endpoints or response keys, while supporting the old ones for a allow time to transition.")]),e._v(" "),t("p",[e._v("For a comparison between the different API versions currently available, see "),t("RouterLink",{attrs:{to:"/server/rest-api/api-versions.html"}},[e._v("API versions")]),e._v(".")],1),e._v(" "),t("p",[e._v("For some older ideas for example requests and responses, see "),t("RouterLink",{attrs:{to:"/development/api-redesign/API.html"}},[e._v("here")]),e._v(".")],1),e._v(" "),t("h2",{attrs:{id:"api-evolution-todo"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#api-evolution-todo"}},[e._v("#")]),e._v(" API evolution TODO")]),e._v(" "),t("p",[e._v("General guidelines:")]),e._v(" "),t("ul",[t("li",[e._v("Publish a clear and complete migration guide")]),e._v(" "),t("li",[e._v("Publish complete reference documentation")]),e._v(" "),t("li",[e._v("Use "),t("code",[e._v("corpus")]),e._v("/"),t("code",[e._v("corpora")]),e._v(" in favor of "),t("code",[e._v("index")]),e._v("/"),t("code",[e._v("indices")]),e._v(".")]),e._v(" "),t("li",[e._v("Be consistent: if information is given in multiple places, e.g. on the server info page as well as on the corpus info page, use the same structure and element names (except one page may give additional details).")]),e._v(" "),t("li",[e._v("Return helpful error messages."),t("br"),e._v("\n(if an illegal value is passed, explain or list legal values, and/or refer to online docs)")]),e._v(" "),t("li",[e._v("JSON should probably be our primary output format"),t("br"),e._v("\n(the XML structure should just be a dumb translation from JSON, for those who need it, e.g. to pass through XSLT). So e.g. no difference in concordance structure between JSON and XML)")]),e._v(" "),t("li",[e._v("Avoid custom encodings (e.g. strings with specific separator characters, such as used for HitProperty and related values); prefer a standard encoding such as JSON.")])]),e._v(" "),t("p",[e._v("Already fixed in v4/5:")]),e._v(" "),t("ul",[t("li",[e._v("Ensure correct data types, e.g. "),t("code",[e._v("fieldValues")]),e._v(" should have integer values, but are strings")]),e._v(" "),t("li",[e._v("Fix "),t("code",[e._v("blacklabBuildTime")]),e._v(" vs. "),t("code",[e._v("blackLabBuildTime")])]),e._v(" "),t("li",[e._v("Added "),t("code",[e._v("before")]),e._v("/"),t("code",[e._v("after")]),e._v(" in addition to "),t("code",[e._v("left")]),e._v("/"),t("code",[e._v("right")]),e._v(" for parameters (response structure unchanged)")]),e._v(" "),t("li",[e._v("Don't include static info on dynamic (results) pages."),t("br"),e._v("\n(e.g. don't send display names for all metadata fields with each hits results;\nthe client can request those once if needed)")]),e._v(" "),t("li",[e._v("Avoid attributes; use elements for everything.")]),e._v(" "),t("li",[e._v("Avoid dynamic XML element names"),t("br"),e._v("(e.g. don't use map keys for XML element names.\nNot an issue if we copy JSON structure)")]),e._v(" "),t("li",[e._v("add "),t("code",[e._v("/corpora/*")]),e._v(" endpoints. Avoid ambiguity with e.g. "),t("code",[e._v("/blacklab-server/input-formats")]),e._v(", and also provide a place to update the API in parallel. That is, these new endpoints will not be 100% compatible but use a newer, cleaner version.")])]),e._v(" "),t("p",[e._v("TODO v4:")]),e._v(" "),t("ul",[t("li",[e._v("Make functionality more orthogonal. E.g. "),t("code",[e._v("subcorpusSize")]),e._v(" can be included in grouped responses, but not in ungrouped ones.")]),e._v(" "),t("li",[e._v("Add a way to pass HitProperty as JSON in addition to custom encoding")])]),e._v(" "),t("p",[e._v("DONE IN /corpora ENDPOINTS (e.g. v5):")]),e._v(" "),t("ul",[t("li",[e._v("Replace "),t("code",[e._v("left")]),e._v("/"),t("code",[e._v("right")]),e._v(" in response with "),t("code",[e._v("before")]),e._v("/"),t("code",[e._v("after")]),t("br"),e._v("\n(makes more sense for RTL languages)")]),e._v(" "),t("li",[e._v("XML: same concordance structure as in JSON")]),e._v(" "),t("li",[e._v("Handle custom information better. "),t("br"),e._v("\nCustom information, ignored by Blacklab but useful for e.g. the frontend,\nlike displayName, uiType, etc. is polluting the response structure.\nWe should isolate it (e.g. in a "),t("code",[e._v("custom")]),e._v(" section for each field, annotation, etc.),\njust pass it along unchecked, and include it only if requested."),t("br"),e._v('\nThis includes the so-called "special fields" except for '),t("code",[e._v("pidField")]),e._v(" (so author, title, date).\n(Blacklab uses the "),t("code",[e._v("pidField")]),e._v(" to refer to documents)")]),e._v(" "),t("li",[e._v("Change confusing names."),t("br"),e._v("\n(e.g. the name "),t("code",[e._v("stoppedRetrievingHits")]),e._v(' prompts the question "why did you stop?".\n'),t("code",[e._v("limitReached")]),e._v(" might be easier to understand, especially if it's directly\nrelated to a configuration setting "),t("code",[e._v("hitLimit")]),e._v(")")]),e._v(" "),t("li",[e._v("Group related values."),t("br"),e._v("\n(e.g. numberOfHitsRetrieved / numberOfDocsRetrieved / stoppedRetrievingHits\nwould be better as a structure "),t("code",[e._v('"retrieved": { "hits": 100, "docs": 10, "reachedHitLimit": true }')]),e._v(" ).")]),e._v(" "),t("li",[e._v("Separate unrelated parts."),t("br"),e._v("\n(e.g. in DocInfo, arbitrary document metadata values such as "),t("code",[e._v("title")]),e._v(" or "),t("code",[e._v("author")]),e._v(" should probably be in a separate subobject, not alongside special values like "),t("code",[e._v("lengthInTokens")]),e._v(" and "),t("code",[e._v("mayView")]),e._v(". Also, "),t("code",[e._v("metadataFieldGroups")]),e._v(" shouldn't be alongside DocInfo structures.)")])]),e._v(" "),t("p",[e._v("DONE API v5:")]),e._v(" "),t("ul",[t("li",[e._v("remove "),t("code",[e._v("/blacklab-server/CORPUSNAME")]),e._v(" endpoints.")])]),e._v(" "),t("p",[e._v("TODO /corpora ENDPOINTS:")]),e._v(" "),t("ul",[t("li",[e._v("XML: When using "),t("code",[e._v("usecontent=orig")]),e._v(", don't make the content part of the XML anymore."),t("br"),e._v("\n(escape it using CDATA (again, same as in JSON). Also consider just returning both\nthe FI concordances as well as the original content (if requested), so the response\nstructure doesn't fundamentally change because of one parameter value)\n(optionally have a parameter to include it as part of the XML if desired, to simplify response handling?)")]),e._v(" "),t("li",[e._v("Return HitPropertyValues as JSON instead of current custom encoding?")])]),e._v(" "),t("p",[e._v("TODO v5:")]),e._v(" "),t("ul",[t("li",[e._v("remove old custom encodings for HitProperty in favour of the JSON format?")])]),e._v(" "),t("p",[e._v("Possible new endpoints/features:")]),e._v(" "),t("ul",[t("li",[e._v("If you're interested in stats like total number of results, subcorpus size, etc., it's kind of confusing to have to do "),t("code",[e._v("/hits?number=0&waitfortotal=true")]),e._v("; maybe have separate endpoints for this kind of application? (calculating stats vs. paging through hits)")])]),e._v(" "),t("p",[e._v("This might be harder to do without breaking compatibility:")]),e._v(" "),t("ul",[t("li",[e._v("Try to use consistent terminology between parameters, response and configuration files."),t("br"),e._v("\n(e.g. use the term "),t("code",[e._v("hitLimit")]),e._v(" everywhere for the same concept)")])]),e._v(" "),t("p",[e._v("Maybe?")]),e._v(" "),t("ul",[t("li",[e._v("Support Solr's common query parameters, e.g. "),t("code",[e._v("start")]),e._v(","),t("code",[e._v("rows")]),e._v(","),t("code",[e._v("fq")]),e._v(", etc.\nas the preferred version."),t("br"),e._v("\nSupport the "),t("code",[e._v("lowerCamelCase")]),e._v(" version of query parameter names for consistency\nwith responses and configuration options."),t("br"),e._v("\nSupport the old query parameter names (but issue deprecation warning when first\nencountered?)")]),e._v(" "),t("li",[e._v("Don't send "),t("code",[e._v("mayView")]),e._v(" for each document (until we implement such granular authorization), include it in corpus info. Although keeping it there doesn't hurt and prepares us for this feature.")]),e._v(" "),t("li",[e._v("Be stricter about parameter values."),t("br"),e._v("\n(if an illegal value is passed, return an error instead of silently using a default value)")]),e._v(" "),t("li",[e._v("Consider adding a JSON request option in addition to regular query parameters.\nThere should be an easy-to-use test interface so there's no need to\nmanually type URL-encoded JSON requests into the browser address bar.")])])])}),[],!1,null,null,null);t.default=n.exports}}]);