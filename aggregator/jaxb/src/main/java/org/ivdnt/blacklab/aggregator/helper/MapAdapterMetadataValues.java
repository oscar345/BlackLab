package org.ivdnt.blacklab.aggregator.helper;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import javax.xml.bind.JAXBElement;
import javax.xml.bind.annotation.adapters.XmlAdapter;
import javax.xml.namespace.QName;

import org.ivdnt.blacklab.aggregator.representation.MetadataValues;
import org.w3c.dom.Element;

/**
 * Helps us to (de)serialize a map of metadata values where the keys become element names in XML.
 *
 * Separate class because there may be more than one metadata value.
 */
public class MapAdapterMetadataValues extends XmlAdapter<MapWrapperMetadataValues, Map<String, MetadataValues>> {
    @Override
    public MapWrapperMetadataValues marshal(Map<String, MetadataValues> m) throws Exception {
        MapWrapperMetadataValues wrapper = new MapWrapperMetadataValues();
        List elements = new ArrayList();
        for (Map.Entry<String, MetadataValues> property : m.entrySet()) {

            elements.add(new JAXBElement<>(new QName(getCleanLabel(property.getKey())),
                    MetadataValues.class, property.getValue()));
        }
        wrapper.elements = elements;
        return wrapper;
    }

    @Override
    public Map<String, MetadataValues> unmarshal(MapWrapperMetadataValues v) throws Exception {
        Map<String, MetadataValues> returnval = new LinkedHashMap();
        for (Object o : v.elements) {
            Element e = (Element) o;

            List<String> values = new ArrayList<>();
            for (int count = 0; count < e.getChildNodes().getLength(); count++) {
                if (e.getChildNodes().item(count) instanceof Element) {
                    values.add(e.getChildNodes().item(count).getTextContent());
                }
            }
            returnval.put(e.getTagName(), new MetadataValues(values));
        }
        return returnval;
    }


    // Return a XML-safe attribute.  Might want to add camel case support
    private String getCleanLabel(String attributeLabel) {
        attributeLabel = attributeLabel.replaceAll("[()]", "").replaceAll("[^\\w\\s]", "_").replaceAll(" ", "_");
        return attributeLabel;
    }
}