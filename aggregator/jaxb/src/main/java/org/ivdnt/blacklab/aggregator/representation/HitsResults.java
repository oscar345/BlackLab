package org.ivdnt.blacklab.aggregator.representation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlElementWrapper;
import javax.xml.bind.annotation.XmlRootElement;

import org.ivdnt.blacklab.aggregator.helper.JacksonUtil;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.JacksonException;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;

@XmlRootElement(name="blacklabResponse")
@XmlAccessorType(XmlAccessType.FIELD)
public class HitsResults {

    /** Use this to serialize this class to JSON */
    private static class Serializer extends JsonSerializer<List<DocInfo>> {
        @Override
        public void serialize(List<DocInfo> value, JsonGenerator jgen, SerializerProvider provider)
                throws IOException {

            if (value == null)
                return;
            jgen.writeStartObject();
            for (DocInfo el: value) {
                String pid = el.pid;
                if (pid == null)
                    pid = "UNKNOWN";
                jgen.writeObjectFieldStart(pid);
                for (Map.Entry<String, MetadataValues> v: el.metadata.entrySet()) {
                    jgen.writeArrayFieldStart(v.getKey());
                    for (String x: v.getValue().getValue()) {
                        jgen.writeString(x);
                    }
                    jgen.writeEndArray();
                }
                if (el.lengthInTokens != null)
                    jgen.writeNumberField("lengthInTokens", el.lengthInTokens);
                if (el.mayView != null)
                    jgen.writeBooleanField("mayView", el.mayView);
                jgen.writeEndObject();
            }
            jgen.writeEndObject();
        }
    }

    private static class Deserializer extends JsonDeserializer<List<DocInfo>> {
        @Override
        public List<DocInfo> deserialize(JsonParser parser, DeserializationContext deserializationContext)
                throws IOException, JacksonException {
            JsonToken token = parser.getCurrentToken();
            if (token != JsonToken.START_OBJECT)
                throw new RuntimeException("Expected START_OBJECT, found " + token);

            List<DocInfo> docInfos = new ArrayList<>();
            while (true) {
                token = parser.nextToken();
                if (token == JsonToken.END_OBJECT)
                    break;

                if (token != JsonToken.FIELD_NAME)
                    throw new RuntimeException("Expected END_OBJECT or FIELD_NAME, found " + token);
                String pid = parser.getCurrentName();
                if (pid.equals("metadataFieldGroups")) {
                    // Skip this part, which doesn't really belong but ended up here unfortunately.
                    parser.nextToken(); // START_ARRAY
                    while (parser.nextToken() != JsonToken.END_ARRAY) {
                        // Skip each metadata field group object (don't contain nested objects)
                        while (parser.nextToken() != JsonToken.END_OBJECT);
                    }
                    continue;
                }
                DocInfo docInfo = new DocInfo();
                docInfo.pid = parser.getCurrentName();
                docInfo.metadata = new LinkedHashMap<>();

                token = parser.nextToken();
                if (token != JsonToken.START_OBJECT)
                    throw new RuntimeException("Expected START_OBJECT, found " + token);
                while (true) {
                    token = parser.nextToken();
                    if (token == JsonToken.END_OBJECT)
                        break;

                    if (token != JsonToken.FIELD_NAME)
                        throw new RuntimeException("Expected END_OBJECT or FIELD_NAME, found " + token);
                    String fieldName = parser.getCurrentName();
                    token = parser.nextToken();
                    if (token == JsonToken.VALUE_NUMBER_INT) {
                        // Special lengthInTokens setting?
                        if (!fieldName.equals("lengthInTokens"))
                            throw new RuntimeException("Unexpected int in metadata");
                        docInfo.lengthInTokens = parser.getValueAsInt();
                    } else if (token == JsonToken.VALUE_TRUE || token == JsonToken.VALUE_FALSE) {
                        // Special mayView setting?
                        if (!fieldName.equals("mayView"))
                            throw new RuntimeException("Unexpected boolean in metadata");
                        docInfo.mayView = parser.getValueAsBoolean();
                    } else if (token == JsonToken.START_ARRAY) {
                        // A list of metadata values
                        List<String> values = JacksonUtil.readStringList(parser);
                        docInfo.metadata.put(fieldName, new MetadataValues(values));
                    }
                }
                docInfos.add(docInfo);
            }

            return docInfos;
        }
    }

    public SearchSummary summary;

    @XmlElementWrapper(name="hits")
    @XmlElement(name = "hit")
    @JsonProperty("hits")
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public List<Hit> hits;

    @XmlElementWrapper(name="docInfos")
    @XmlElement(name = "docInfo")
    @JsonProperty("docInfos")
    @JsonSerialize(using = Serializer.class)
    @JsonDeserialize(using = Deserializer.class)
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public List<DocInfo> docInfos;

    @XmlElementWrapper(name="hitGroups")
    @XmlElement(name = "hitGroup")
    @JsonProperty("hitGroups")
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public List<HitGroup> hitGroups;

    // required for Jersey
    public HitsResults() {}

    public HitsResults(SearchSummary summary, List<Hit> hits,
            List<DocInfo> docInfos) {
        this.summary = summary;
        this.hits = hits;
        this.docInfos = docInfos;
        this.hitGroups = null;
    }

    public HitsResults(SearchSummary summary, List<HitGroup> groups) {
        this.summary = summary;
        this.hits = null;
        this.docInfos = null;
        this.hitGroups = groups;
    }

    @Override
    public String toString() {
        return "HitsResults{" +
                "summary=" + summary +
                ", hits=" + hits +
                ", docInfos=" + docInfos +
                ", hitGroups=" + hitGroups +
                '}';
    }
}
