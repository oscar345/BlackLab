package nl.inl.blacklab.search.indexmetadata;

import java.util.Map;

import nl.inl.blacklab.search.BlackLabIndex;

/** An annotated field */
public interface AnnotatedField extends Field {

	/**
	 * Get the annotations for this field.
	 *
	 * Properties are returned sorted according to the displayOrder defined in the
	 * index metadata, if any.
	 *
	 * @return the annotations
	 */
	Annotations annotations();
	
	default Annotation mainAnnotation() {
	    return annotations().main();
	}
	
	default Annotation annotation(String name) {
	    return annotations().get(name);
	}

    @Deprecated
	default boolean hasXmlTags() { return hasRelationAnnotation(); }

    boolean hasRelationAnnotation();

    Map<String, Map<String, Long>> getRelationsMap(BlackLabIndex index);

    RelationsStats getRelationsStats(BlackLabIndex index);

    String tokenLengthField();

    @Override
    default String contentsFieldName() {
        Annotation main = mainAnnotation();
        AnnotationSensitivity offsetsSensitivity = main.offsetsSensitivity();
        if (offsetsSensitivity == null)
            offsetsSensitivity = main.sensitivity(MatchSensitivity.SENSITIVE);
        return offsetsSensitivity.luceneField();
    }

    default Annotation punctAnnotation() {
		return hasXmlTags() ? annotation(AnnotatedFieldNameUtil.PUNCTUATION_ANNOT_NAME): null;
	}
}
