package nl.inl.blacklab.forwardindex;

import java.nio.charset.StandardCharsets;
import java.text.Collator;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import nl.inl.blacklab.forwardindex.Collators.CollatorVersion;
import nl.inl.blacklab.search.indexmetadata.MatchSensitivity;

public class TestCollatorsZalgo {

    /** A "Zalgo" text string, containing lots of combining diacritics marks. */
    private static String zalgo1;

    /** A "Zalgo" text string, containing lots of combining diacritics marks. */
    private static String zalgo2;

    private static String fromIntArrayOfByteNumbers(int[] ax) {
        List<Byte> l = new ArrayList<>();
        for (int i: ax) {
            l.add((byte)i);
        }
        byte [] b = new byte[l.size()];
        for (int i = 0; i < l.size(); i++) {
            b[i] = l.get(i);
        }
        return new String(b, StandardCharsets.UTF_8);
    }

    private static Collator getDefaultEnglishCollator() {
        return Collator.getInstance(new Locale("en", "GB"));
    }

    private static Collator getBlackLabCollator() {
        Collator coll = getDefaultEnglishCollator();
        Collators colls = new Collators(coll, CollatorVersion.V2);
        return colls.get(MatchSensitivity.INSENSITIVE);
    }

    @BeforeClass
    public static void setUp() {
        int[] zalgoBytes1 = { 204, 184, 205, 129, 204, 134, 205, 131, 205, 128, 204, 191, 204, 148, 204, 136, 204, 129, 204, 191, 204, 189, 205, 160, 205, 130, 205, 151, 204, 143, 205, 155, 204, 143, 205, 130, 204, 129, 205, 157, 205, 134, 204, 132, 205, 138, 205, 157, 205, 151, 205, 155, 205, 130, 204, 133, 204, 145, 204, 144, 204, 145, 204, 144, 204, 189, 205, 128, 204, 131, 204, 131, 204, 141, 204, 146, 205, 128, 205, 151, 204, 146, 204, 136, 204, 191, 204, 191, 204, 129, 205, 157, 204, 142, 205, 132, 205, 155, 205, 151, 205, 144, 204, 144, 204, 132, 205, 151, 204, 141, 204, 189, 204, 154, 205, 132, 204, 145, 204, 132, 204, 128, 204, 135, 204, 144, 204, 147, 205, 151, 205, 145, 204, 139, 205, 144, 204, 149, 204, 131, 204, 155, 205, 155, 205, 132, 204, 144, 205, 128, 204, 134, 204, 142, 205, 140, 204, 139, 205, 129, 204, 149, 205, 139, 205, 155, 204, 135, 204, 129, 204, 176, 204, 159, 204, 178, 205, 147, 204, 151, 205, 135, 204, 162, 205, 141, 204, 187, 205, 141, 205, 156, 205, 156, 204, 169, 205, 133, 204, 159, 205, 141, 205, 154, 204, 177, 205, 133, 204, 166, 204, 160, 204, 163, 204, 185, 204, 178, 204, 165, 204, 171, 204, 185, 204, 177, 204, 171, 205, 156, 204, 168, 204, 179, 204, 167, 204, 152, 204, 173, 205, 137, 204, 179, 204, 160, 204, 161, 205, 133, 205, 135, 205, 137, 204, 175, 204, 157, 204, 159, 204, 159, 204, 157, 204, 185, 204, 186, 205, 156, 204, 151, 205, 147, 204, 175, 204, 163, 204, 173, 204, 153, 204, 153, 204, 175, 205, 135, 204, 163, 205, 141, 204, 164, 205, 153, 204, 188, 205, 154, 205, 153, 205, 135, 204, 167, 205, 156, 205, 137, 108, 204, 184, 204, 128, 205, 140, 204, 145, 205, 146, 204, 136, 204, 142, 205, 151, 204, 190, 204, 140, 204, 136, 205, 160, 204, 147, 205, 144, 205, 157, 204, 140, 204, 132, 205, 138, 204, 138, 204, 128, 204, 130, 204, 138, 204, 128, 204, 147, 205, 138, 205, 157, 204, 131, 204, 136, 205, 155, 204, 132, 204, 138, 204, 139, 205, 157, 204, 139, 204, 146, 204, 142, 204, 139, 204, 132, 204, 146, 204, 133, 205, 144, 205, 155, 205, 130, 205, 157, 204, 154, 204, 141, 204, 189, 204, 142, 204, 146, 204, 132, 205, 131, 204, 138, 205, 146, 204, 128, 205, 132, 205, 146, 204, 144, 205, 160, 205, 130, 204, 144, 204, 149, 205, 132, 205, 157, 205, 160, 204, 141, 204, 149, 204, 154, 205, 139, 204, 142, 204, 129, 205, 147, 204, 187, 204, 185, 204, 177, 204, 170, 205, 142, 205, 136, 204, 187, 204, 150, 204, 173, 204, 164, 204, 159, 205, 137, 204, 159, 205, 154, 204, 186, 204, 167, 204, 175, 204, 159, 205, 154, 205, 156, 204, 158, 205, 137, 204, 177, 204, 159, 205, 149, 205, 156, 205, 154, 204, 187, 204, 151, 204, 157, 204, 165, 204, 150, 204, 169, 205, 133, 205, 148, 204, 176, 204, 178, 204, 171, 204, 172, 205, 133, 204, 172, 204, 174, 204, 161, 205, 154, 204, 176, 204, 151, 204, 159, 204, 186, 204, 157, 204, 152, 204, 162, 204, 170, 204, 166, 204, 188, 205, 153, 204, 156, 204, 152, 204, 185, 204, 156, 204, 174, 205, 142, 204, 185, 204, 169, 204, 159, 205, 148, 205, 149, 204, 160, 204, 157, 204, 156, 205, 149, 111, 204, 181, 204, 140, 204, 129, 205, 160, 205, 138, 204, 130, 204, 155, 204, 144, 204, 148, 204, 146, 205, 145, 204, 131, 204, 154, 205, 155, 205, 131, 205, 144, 204, 190, 205, 140, 204, 133, 205, 151, 205, 151, 205, 152, 205, 140, 205, 132, 205, 160, 204, 135, 204, 138, 204, 145, 204, 147, 204, 147, 204, 144, 205, 134, 204, 136, 204, 133, 204, 142, 204, 154, 205, 129, 204, 139, 204, 133, 204, 145, 204, 137, 204, 146, 204, 155, 204, 141, 204, 134, 205, 157, 204, 134, 205, 130, 204, 140, 204, 133, 205, 134, 204, 191, 205, 132, 205, 129, 205, 129, 205, 151, 205, 157, 204, 142, 204, 134, 205, 157, 204, 138, 205, 146, 204, 128, 205, 160, 204, 144, 205, 145, 204, 130, 205, 138, 204, 136, 204, 191, 204, 147, 204, 142, 205, 131, 204, 140, 205, 131, 205, 152, 204, 128, 204, 145, 205, 146, 205, 129, 205, 155, 205, 155, 204, 128, 205, 130, 204, 133, 204, 132, 205, 139, 204, 130, 205, 131, 205, 139, 204, 169, 205, 154, 204, 157, 204, 166, 204, 153, 115, 204, 182, 204, 129, 204, 138, 204, 140, 204, 138, 204, 171, 204, 174, 204, 151, 205, 142, 204, 171, 204, 170, 205, 149, 204, 164, 204, 160, 204, 173, 204, 153, 204, 173, 204, 177, 205, 149, 204, 158, 204, 172, 205, 156, 204, 187, 204, 178, 204, 150, 204, 187, 205, 141, 204, 173, 204, 164, 204, 156, 204, 152, 204, 159, 204, 177, 205, 147, 204, 187, 204, 169, 204, 152, 205, 147, 204, 152, 205, 149, 205, 156, 204, 150, 205, 133, 205, 142, 204, 152, 204, 166, 204, 174, 204, 185, 204, 165, 204, 161, 204, 170, 204, 175, 204, 163, 205, 150, 204, 170, 204, 157, 205, 133, 204, 162, 204, 176, 204, 160, 116, 204, 182, 205, 139, 205, 151, 204, 150, 205, 136, 204, 163, 204, 176, 204, 174, 205, 142, 204, 151, 204, 188, 205, 148, 204, 157, 204, 168, 204, 179, 204, 168, 205, 149, 204, 152, 204, 173, 204, 156, 204, 160, 204, 174, 205, 133, 205, 149, 205, 153, 204, 161, 204, 187, 204, 150, 204, 152, 204, 153, 204, 178, 204, 170, 204, 151, 205, 154, 204, 173, 204, 176, 204, 152, 204, 173, 204, 169, 205, 147, 205, 154, 204, 151, 204, 161, 204, 163, 204, 159, 204, 178, 205, 150, 205, 136, 204, 168, 205, 136, 204, 172, 205, 142, 205, 156, 204, 171, 205, 137, 204, 158, 204, 163, 205, 153, 204, 164, 204, 161, 204, 162, 205, 153, 204, 158, 204, 170, 205, 149, 205, 137, 204, 170, 204, 187, 205, 142, 204, 170, 205, 149, 204, 152, 205, 153, 205, 148, 205, 137, 204, 169, 204, 171, 205, 154, 205, 156, 204, 173, 204, 169, 205, 147, 204, 165, 44, 204, 182, 204, 154, 204, 171, 204, 188, 204, 177, 205, 135, 204, 165, 204, 185, 204, 174, 205, 154, 205, 154, 204, 185, 204, 174, 204, 161, 204, 150, 204, 188, 204, 167, 204, 156, 205, 148, 204, 169, 204, 169, 204, 165, 204, 168, 205, 148, 204, 177, 204, 177, 204, 151, 205, 148, 205, 150, 204, 178 };
        int[] zalgoBytes2 = { 204, 183, 204, 128, 204, 177, 204, 159, 204, 169, 204, 160, 204, 161, 205, 147, 116, 204, 180, 205, 128, 204, 144, 205, 155, 204, 139, 205, 152, 204, 138, 204, 132, 205, 138, 205, 138, 205, 157, 204, 146, 204, 147, 204, 131, 205, 129, 204, 146, 204, 135, 205, 146, 204, 137, 204, 187, 204, 169, 204, 186, 205, 153, 204, 171, 205, 141, 204, 167, 204, 162, 204, 171, 204, 168, 111, 204, 184, 204, 143, 205, 134, 204, 178, 204, 174, 204, 173, 205, 135, 205, 137, 204, 152, 204, 176, 204, 167, 204, 178, 204, 187, 204, 177, 112, 204, 182, 204, 137, 205, 152, 204, 137, 205, 131, 205, 144, 205, 151, 205, 140, 204, 128, 204, 131, 204, 131, 204, 161, 105, 204, 184, 205, 132, 205, 151, 204, 131, 204, 137, 204, 129, 204, 141, 204, 189, 205, 132, 205, 132, 204, 141, 204, 131, 204, 140, 204, 130, 205, 145, 205, 140, 204, 169, 113, 204, 180, 204, 144, 205, 132, 205, 139, 204, 133, 204, 147, 204, 136, 204, 132, 204, 149, 205, 157, 204, 133, 204, 147, 205, 128, 204, 145, 204, 154, 204, 137, 204, 170, 205, 133, 204, 174, 204, 179, 204, 173, 204, 160, 204, 178, 205, 135, 204, 177, 204, 177, 204, 152, 204, 178, 205, 148, 205, 149, 117, 204, 184, 204, 139, 204, 139, 204, 144, 204, 148, 204, 154, 204, 189, 204, 140, 204, 137, 205, 139, 204, 142, 204, 143, 204, 149, 205, 155, 205, 138, 204, 140, 204, 139, 205, 147, 204, 151, 204, 158, 204, 172, 204, 168, 204, 164, 205, 142, 204, 157, 101, 204, 184, 204, 130, 204, 131, 204, 191, 204, 138, 204, 134, 205, 157, 205, 146, 204, 191, 205, 128, 205, 139, 204, 143, 204, 141, 204, 185, 204, 178, 204, 165 };
        zalgo1 = fromIntArrayOfByteNumbers(zalgoBytes1);
        zalgo2 = fromIntArrayOfByteNumbers(zalgoBytes2);
    }

    @Test
    public void testZalgoTextWithRuleBasedCollator() {
        // NOTE: This fails on Java 8, but succeeds on Java 11.
        //       This may be due to new Unicode characters or combining diacritics being added, or some other reason (e.g. a bugfix).
        //       We don't yet know the first Java version where it succeeds; could be 9, 10 or 11.
        //       If you're trying to build BlackLab on an older Java version and it fails because of this test, you can try @Ignoring it and you
        //       might be fine, depending on your corpus data.
        Assert.assertEquals(1, getBlackLabCollator().compare(zalgo1, zalgo2));
    }

    @Test
    public void testZalgoTextWithDefaultCollator() {
        // This should succeed, even on Java 8, but is here just for completeness.
        Assert.assertEquals(1, getDefaultEnglishCollator().compare(zalgo1, zalgo2));
    }
}
