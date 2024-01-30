package nl.inl.blacklab.util;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.lang3.StringUtils;

/**
 * Takes care of serializing/deserializing Hit/DocProperties and -Values with
 * proper escaping.
 */
public final class PropertySerializeUtil {

    private final static String PART_SEPARATOR = ":";

    private final static String PART_SEPARATOR_ESC_REGEX = ":";

    private final static String MULTIPLE_SEPARATOR = ",";

    private final static String MULTIPLE_SEPARATOR_ESC_REGEX = ",";

    private PropertySerializeUtil() {
    }

    public static String escapePart(String part) {
        return part.replace("$", "$DL").replace(",", "$CM").replace(":", "$CL");
    }

    public static String unescapePart(String partEscaped) {
        return partEscaped.replace("$CL", ":").replace("$CM", ",").replace("$DL", "$");
    }

    public static String combineParts(String... parts) {
        return Arrays.stream(parts).map(part -> escapePart(part)).collect(Collectors.joining(PART_SEPARATOR));
    }

    public static String combineParts(List<String> parts) {
        return combineParts(parts.toArray(new String[0]));
    }

    /**
     * Split string on first part and the rest.
     *
     * NOTE: only the first part is unescaped; the rest should be split later and unescaped at that time!
     *
     * @param partsCombined serialized parts
     * @return an array of length 1 or 2, depending on whether there's more than one part
     */
    public static String[] splitPartFirstRest(String partsCombined) {
        String[] parts = partsCombined.split(PART_SEPARATOR_ESC_REGEX, 2);
        for (int i = 0; i < parts.length; i++) {
            parts[i] = i == 0 ? unescapePart(parts[i]) : parts[i]; // only unescape first part, not the rest!
        }
        return parts;
    }

    public static String[] splitParts(String partsCombined) {
        String[] parts = partsCombined.split(PART_SEPARATOR_ESC_REGEX, -1);
        for (int i = 0; i < parts.length; i++) {
            parts[i] = unescapePart(parts[i]);
        }
        return parts;
    }

    public static String combineMultiple(String... values) {
        return StringUtils.join(values, MULTIPLE_SEPARATOR);
    }

    public static String[] splitMultiple(String valueCombined) {
        return valueCombined.split(MULTIPLE_SEPARATOR_ESC_REGEX, -1);
    }

    public static boolean isMultiple(String serialized) {
        return serialized.contains(PropertySerializeUtil.MULTIPLE_SEPARATOR);
    }

    public interface SerializableProperty {
        String serialize();
    }

    public static String serializeMultiple(boolean reverse, List<? extends SerializableProperty> properties) {
        String[] values = new String[properties.size()];
        for (int i = 0; i < properties.size(); i++) {
            values[i] = properties.get(i).serialize();
        }
        return (reverse ? "-(" : "") + combineMultiple(values) + (reverse ? ")" : "");
    }
}
