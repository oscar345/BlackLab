package nl.inl.util;

import java.text.Normalizer;
import java.util.regex.Pattern;

import org.apache.commons.lang3.StringUtils;

/**
 * A collection of String-related utility methods and regular expression
 * patterns.
 */
public final class StringUtil {
    /** nbsp character (decimal 160 = hex A0) */
    public static final char CHAR_NON_BREAKING_SPACE = '\u00A0';

    public static final char CHAR_EM_SPACE = '\u2003';

    /** zero-width space character */
    public static final char CHAR_ZERO_WIDTH_SPACE = '\u200b';

    public static final char CHAR_SOFT_HYPHEN = '\u00AD';

    public static final char CHAR_DELETE = '\u007F';

    /** Matches one or more whitespace characters. */
    public static final String REGEX_WHITESPACE = "[\\p{javaSpaceChar}\n]+";

    /** Matches one or more whitespace characters. */
    private static final Pattern PATT_WHITESPACE = Pattern.compile(REGEX_WHITESPACE);

    private StringUtil() {
    }

    /** Any characters that should be escaped when constructing a regular expression matching a value */
    private static final Pattern PATT_REGEX_CHARACTERS = Pattern.compile("([|\\\\?*+()\\[\\]\\-^${}.])");

    /**
     * Escape regex special characters
     * 
     * (Pattern.quote() also does this, but this method is needed if you use a different regex
     *  engine from Java's, such as with Lucene)
     *
     * @param termStr the string to escape characters in
     * @return the escaped string
     */
    public static String escapeRegexCharacters(String termStr) {
        return PATT_REGEX_CHARACTERS.matcher(termStr).replaceAll("\\\\$1");
    }

    public static final Pattern PATT_LEADING_OR_TRAILING_WHITESPACE = Pattern.compile("^\\p{javaSpaceChar}+|\\p{javaSpaceChar}+$");

    /** Trim any Java space characters from start and end of string.
     *
     * Contrary to String.trim() this trims all space characters
     * ({@link Character#isSpaceChar(char)}), not "codepoint 32 and below".
     *
     * @param s the string to trim
     * @return the trimmed string
     */
    public static String trimWhitespace(String s) {
        return PATT_LEADING_OR_TRAILING_WHITESPACE.matcher(s).replaceAll("");
    }

    /**
     * Replace adjacent whitespace characters with a single space
     *
     * @param s source string
     * @return the result
     */
    public static String normalizeWhitespace(String s) {
        return PATT_WHITESPACE.matcher(s).replaceAll(" ");
    }

    /** Diacritical marks to be removed after decomposition */
    private static final Pattern PATT_DIACRITICAL_MARKS = Pattern.compile("\\p{InCombiningDiacriticalMarks}+");

    private static final Pattern PATT_DIACRITICAL_MARKS_AND_EM_SPACE =
            Pattern.compile("[\\p{InCombiningDiacriticalMarks}" + CHAR_EM_SPACE + "]+");

    /**
     * Removes diacritics (~= accents) from a string. The case will not be altered.
     *
     * For instance, '&agrave;' will be replaced by 'a'. Note that ligatures will be left as is.
     *
     * Also strips out 0xAD (also known as soft hyphen or &amp;shy;), which frequently causes
     * issues when comparing insensitively (and Collator ignores it as well).
     *
     * <pre>
     * StringUtils.stripAccents(null)                = null
     * StringUtils.stripAccents("")                  = ""
     * StringUtils.stripAccents("control")           = "control"
     * StringUtils.stripAccents("&eacute;clair")     = "eclair"
     * </pre>
     *
     * NOTE: this method was copied from Apache StringUtils. Changes:
     * precompiling the regular expression for efficiency; optionally removing em space
     *
     * @param input String to be stripped
     * @return input text with diacritics removed
     *
     * @since 3.0
     */
    // See also Lucene's ASCIIFoldingFilter (Lucene 2.9) that replaces accented characters by their unaccented equivalent (and uncommitted bug fix: https://issues.apache.org/jira/browse/LUCENE-1343?focusedCommentId=12858907&page=com.atlassian.jira.plugin.system.issuetabpanels%3Acomment-tabpanel#action_12858907).
    public static String stripAccents(final String input) {
        return stripAccents(input, false);
    }

    public static String stripAccents(final String input, boolean removeEmSpace) {
        if (input == null) {
            return null;
        }
        final StringBuilder decomposed = new StringBuilder(Normalizer.normalize(input, Normalizer.Form.NFD));
        convertRemainingAccentCharacters(decomposed);
        // Note that this doesn't correctly remove ligatures...
        Pattern patt = removeEmSpace ? PATT_DIACRITICAL_MARKS_AND_EM_SPACE : PATT_DIACRITICAL_MARKS;
        return patt.matcher(decomposed).replaceAll("");
    }

    private static final char CHAR_LATIN_UPPER_L_WITH_STROKE = '\u0141'; // Ł

    private static final char CHAR_LATIN_LOWER_L_WITH_STROKE = '\u0142'; // ł

    private static void convertRemainingAccentCharacters(StringBuilder decomposed) {
        for (int i = 0; i < decomposed.length(); i++) {
            if (decomposed.charAt(i) == CHAR_LATIN_UPPER_L_WITH_STROKE) {
                decomposed.deleteCharAt(i);
                decomposed.insert(i, 'L');
            } else if (decomposed.charAt(i) == CHAR_LATIN_LOWER_L_WITH_STROKE) {
                decomposed.deleteCharAt(i);
                decomposed.insert(i, 'l');
            }
        }
    }

    /**
     * A lowercase letter followed by an uppercase one, both matched in groups.
     */
    static final Pattern lowercaseCharFollowedByUppercase = Pattern.compile("(\\p{Ll})(\\p{Lu})");


    /**
     * Convert a string from a camel-case "identifier" style to a human-readable
     * version, by putting spaces between words, uppercasing the first letter and
     * lowercasing the rest.
     *
     * E.g. myCamelCaseString becomes "My camel case string".
     *
     * @param camelCaseString a string in camel case, i.e. multiple capitalized
     *            words glued together.
     * @param dashesToSpaces if true, also converts dashes and underscores to spaces
     * @return a human-readable version of the input string
     */
    public static String camelCaseToDisplayable(String camelCaseString, boolean dashesToSpaces) {
        String spaceified = camelCaseString;
        spaceified = lowercaseCharFollowedByUppercase.matcher(spaceified).replaceAll("$1 $2");
        if (dashesToSpaces)
            spaceified = spaceified.replaceAll("[\\-_]", " ");
        return StringUtils.capitalize(spaceified.toLowerCase());
    }

    /**
     * For a number n, return a string like "nth".
     *
     * So 1 returns "1st", 2 returns "2nd", and so on.
     *
     * @param docNumber number
     * @return ordinal for that number
     */
    public static String ordinal(int docNumber) {
        final String[] ordSuffix = { "th", "st", "nd", "rd" };
        int i = docNumber;
        if (i < 1 || i > 3)
            i = 0;
        return docNumber + ordSuffix[i];
    }

    /**
     * Convert wildcard string to regex string.
     *
     * Adapted from: http://stackoverflow.com/a/28758377
     *
     * @param wildcard wildcard pattern
     * @return equivalent regex pattern
     */
    public static String wildcardToRegex(String wildcard) {
        StringBuilder s = new StringBuilder(wildcard.length());
        s.append('^');
        for (int i = 0, is = wildcard.length(); i < is; i++) {
            char c = wildcard.charAt(i);
            switch (c) {
            case '*':
                s.append(".*");
                break;
            case '?':
                s.append(".");
                break;
            // escape special regexp-characters
            case '^': // escape character in cmd.exe
            case '(':
            case ')':
            case '[':
            case ']':
            case '$':
            case '.':
            case '{':
            case '}':
            case '|':
            case '\\':
                s.append("\\");
                s.append(c);
                break;
            default:
                s.append(c);
                break;
            }
        }
        s.append('$');
        return s.toString();
    }

    /**
     * Lowercase string and remove any diacritics.
     *
     * @param str string to desensitize
     * @return desensitized string
     */
    public static String desensitize(String str) {
        return stripAccents(str).toLowerCase();
    }

    public static String ord(int pass) {
        pass++;
        switch(pass) {
        case 1:
            return "1st";
        case 2:
            return "2nd";
        case 3:
            return "3rd";
        default:
            return pass + "th";
        }
    }

    private static final String REGEX_REMOVE_UNPRINTABLES = "[" + CHAR_ZERO_WIDTH_SPACE + CHAR_SOFT_HYPHEN + "]";

    private static final Pattern PATT_REMOVE_UNPRINTABLES = Pattern.compile(REGEX_REMOVE_UNPRINTABLES);

    /**
     * Remove unprintable characters and normalize to canonical unicode composition.
     * @param value string to sanitize
     * @return sanitized string
     */
    public static String sanitizeAndNormalizeUnicode(String value) {
        value = PATT_REMOVE_UNPRINTABLES.matcher(value).replaceAll("");
        return Normalizer.normalize(value, Normalizer.Form.NFC);
    }

    public static String removeCharsIgnoredByInsensitiveCollator(String s) {
        return s.replaceAll("[\t\n\r" + CHAR_EM_SPACE + CHAR_NON_BREAKING_SPACE + CHAR_DELETE + "]", "");
    }
}
