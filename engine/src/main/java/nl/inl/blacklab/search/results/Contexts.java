package nl.inl.blacklab.search.results;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.StringUtils;
import org.eclipse.collections.api.map.primitive.MutableIntIntMap;
import org.eclipse.collections.api.tuple.primitive.IntIntPair;
import org.eclipse.collections.impl.factory.primitive.IntIntMaps;

import it.unimi.dsi.fastutil.BigList;
import it.unimi.dsi.fastutil.objects.ObjectBigArrayBigList;
import nl.inl.blacklab.Constants;
import nl.inl.blacklab.exceptions.BlackLabRuntimeException;
import nl.inl.blacklab.exceptions.InterruptedSearch;
import nl.inl.blacklab.forwardindex.AnnotationForwardIndex;
import nl.inl.blacklab.forwardindex.Terms;
import nl.inl.blacklab.search.BlackLabIndex;
import nl.inl.blacklab.search.Kwic;
import nl.inl.blacklab.search.TermFrequencyList;
import nl.inl.blacklab.search.indexmetadata.AnnotatedField;
import nl.inl.blacklab.search.indexmetadata.AnnotatedFieldNameUtil;
import nl.inl.blacklab.search.indexmetadata.Annotation;
import nl.inl.blacklab.search.indexmetadata.MatchSensitivity;

/**
 * Some annotation context(s) belonging to a list of hits.
 *
 * This interface is read-only.
 */
public class Contexts implements Iterable<int[]> {

    /** In context arrays, how many bookkeeping ints are stored at the start? */
    private final static int NUMBER_OF_BOOKKEEPING_INTS = 3;

    /**
     * In context arrays, what index after the bookkeeping units indicates the hit
     * start?
     */
    private final static int HIT_START_INDEX = 0;

    /**
     * In context arrays, what index indicates the hit end (start of right part)?
     */
    private final static int RIGHT_START_INDEX = 1;

    /** In context arrays, what index indicates the length of the context? */
    private final static int LENGTH_INDEX = 2;

    /**
     * Retrieves the KWIC information (KeyWord In Context: left, hit and right
     * context) for a number of hits in the same document from the ContentStore.
     *
     * Used by Kwics.retrieveKwics().
     *
     * NOTE: this destroys any existing contexts!
     *
     * @param hits hits in this one document
     * @param forwardIndex Forward index for the words
     * @param punctForwardIndex Forward index for the punctuation
     * @param attrForwardIndices Forward indices for the attributes, or null if none
     * @param wordsAroundHit number of words left and right of hit to fetch
     * @param theKwics where to add the KWICs
     */
    static void makeKwicsSingleDocForwardIndex(
                                               Hits hits,
                                               AnnotationForwardIndex forwardIndex,
                                               AnnotationForwardIndex punctForwardIndex,
                                               Map<Annotation, AnnotationForwardIndex> attrForwardIndices,
                                               ContextSize wordsAroundHit,
                                               Map<Hit, Kwic> theKwics
                                               ) {
        if (hits.size() == 0)
            return;
        HitsInternal hitsInternal = hits.getInternalHits();
        List<String> matchInfoNames = hits.matchInfoNames();

        // OPT: more efficient to get all contexts with one getContextWords() call!

        // Get punctuation context
        int[][] punctContext = null;
        if (punctForwardIndex != null) {
            punctContext = getContextWordsSingleDocument(hitsInternal, 0, hitsInternal.size(), wordsAroundHit, List.of(punctForwardIndex), matchInfoNames);
        }
        Terms punctTerms = punctForwardIndex == null ? null : punctForwardIndex.terms();

        // Get attributes context
        Annotation[] attrName = null;
        Terms[] attrTerms = null;
        int[][][] attrContext = null;
        if (attrForwardIndices != null) {
            int n = attrForwardIndices.size();
            attrName = new Annotation[n];
            AnnotationForwardIndex[] attrFI = new AnnotationForwardIndex[n];
            attrTerms = new Terms[n];
            attrContext = new int[n][][];
            int i = 0;
            for (Map.Entry<Annotation, AnnotationForwardIndex> e: attrForwardIndices.entrySet()) {
                attrName[i] = e.getKey();
                attrFI[i] = e.getValue();
                attrTerms[i] = attrFI[i].terms();
                attrContext[i] = getContextWordsSingleDocument(hitsInternal, 0, hitsInternal.size(), wordsAroundHit, List.of(attrFI[i]), matchInfoNames);
                i++;
            }
        }

        // Get word context
        int[][] wordContext = getContextWordsSingleDocument(hitsInternal, 0, hitsInternal.size(), wordsAroundHit, List.of(forwardIndex), matchInfoNames);
        Terms terms = forwardIndex.terms();

        // Make the concordances from the context
        AnnotatedField field = forwardIndex.annotation().field();
        Annotation concPunctFI = field.annotation(AnnotatedFieldNameUtil.PUNCTUATION_ANNOT_NAME);
        Annotation concWordFI = field.mainAnnotation();
        int hitIndex = -1;
        for (Hit h: hits) {
            hitIndex++;
            List<String> tokens = new ArrayList<>();
            int[] context = wordContext[hitIndex];
            int contextLength = context[Contexts.LENGTH_INDEX];
            int contextRightStart = context[Contexts.RIGHT_START_INDEX];
            int contextHitStart = context[Contexts.HIT_START_INDEX];
            int indexInContext = Contexts.NUMBER_OF_BOOKKEEPING_INTS;
            for (int j = 0; j < contextLength; j++, indexInContext++) {

                // Add punctuation before word
                // (Applications may choose to ignore punctuation before the first word)
                if (punctTerms == null) {
                    // There is no punctuation forward index. Just put a space
                    // between every word.
                    tokens.add(" ");
                } else
                    tokens.add(punctTerms.get(punctContext[hitIndex][indexInContext]));

                // Add extra attributes (e.g. lemma, pos)
                if (attrContext != null) {
                    for (int k = 0; k < attrContext.length; k++) {
                        tokens.add(attrTerms[k].get(attrContext[k][hitIndex][indexInContext]));
                    }
                }

                // Add word
                if (terms != null)
                    tokens.add(terms.get(context[indexInContext]));
                else
                    tokens.add(""); // weird, but make sure the numbers add up at the end

            }
            List<Annotation> annotations = new ArrayList<>();
            annotations.add(concPunctFI);
            if (attrContext != null) {
                annotations.addAll(Arrays.asList(attrName));
            }
            annotations.add(concWordFI); // NOTE: final one is used as main annotation in XML responses!
            Kwic kwic = new Kwic(annotations, tokens, contextHitStart, contextRightStart);
            theKwics.put(h, kwic);
        }
    }

    /**
     * Get context words from the forward index.
     *
     * @param hits the hits
     * @param start first hit to get context words for
     * @param end first hit NOT to get context for (hit after the last to get context for)
     * @param contextSize how many words of context we want
     * @param contextSources forward indices to get context from
     */
    private static int[][] getContextWordsSingleDocument(HitsInternal hits, long start, long end, ContextSize contextSize,
                                                         List<AnnotationForwardIndex> contextSources, List<String> matchInfoNames) {
        if (end - start > Constants.JAVA_MAX_ARRAY_SIZE)
            throw new BlackLabRuntimeException("Cannot handle more than " + Constants.JAVA_MAX_ARRAY_SIZE + " hits in a single doc");
        final int n = (int)(end - start);
        if (n == 0)
            return new int[0][];
        int[] startsOfSnippets = new int[n];
        int[] endsOfSnippets = new int[n];

        EphemeralHit hit = new EphemeralHit();
        for (long i = start; i < end; ++i) {
            hits.getEphemeral(i, hit);
            int j = (int)(i - start);
            contextSize.getSnippetStartEnd(hit, matchInfoNames, false, startsOfSnippets, j, endsOfSnippets, j);
        }

        int fiNumber = 0;
        int doc = hits.doc(start);
        int[][] contexts = new int[n][];
        for (AnnotationForwardIndex forwardIndex: contextSources) {
            // Get all the words from the forward index
            List<int[]> words;
            if (forwardIndex != null) {
                // We have a forward index for this field. Use it.
                words = forwardIndex.retrievePartsInt(doc, startsOfSnippets, endsOfSnippets);
            } else {
                throw new BlackLabRuntimeException("Cannot get context without a forward index");
            }

            // Build the actual concordances
//            int hitNum = 0;
            for (int i = 0; i < n; ++i) {
                long hitIndex = start + i;
                int[] theseWords = words.get(i);
                hits.getEphemeral(hitIndex, hit);

                int firstWordIndex = startsOfSnippets[i];

                if (fiNumber == 0) {
                    // Allocate context array and set hit and right start and context length
                    contexts[i] = new int[NUMBER_OF_BOOKKEEPING_INTS
                            + theseWords.length * contextSources.size()];
                    // Math.min() so we don't go beyond actually retrieved snippet (which may have been limited because of config)!
                    contexts[i][HIT_START_INDEX] = Math.min(theseWords.length, hit.start - firstWordIndex);
                    contexts[i][RIGHT_START_INDEX] = Math.min(theseWords.length, hit.end - firstWordIndex);
                    contexts[i][LENGTH_INDEX] = theseWords.length;
                }
                // Copy the context we just retrieved into the context array
                int copyStart = fiNumber * theseWords.length + NUMBER_OF_BOOKKEEPING_INTS;
                System.arraycopy(theseWords, 0, contexts[i], copyStart, theseWords.length);
            }

            fiNumber++;
        }
        return contexts;
    }

    /**
     * The hit contexts.
     *
     * There may be multiple contexts for each hit. Each
     * int array starts with three bookkeeping integers, followed by the contexts
     * information. The bookkeeping integers are:
     * 0 = hit start, index of the hit word (and length of the left context), counted from the start of the context
     * 1 = right start, start of the right context, counted from the start the context
     * 2 = context length, length of 1 context. As stated above, there may be multiple contexts.
     *
     * The first context therefore starts at index 3.
     */
    private final BigList<int[]> contexts;

    /**
     * If we have context information, this specifies the annotation(s) (i.e. word,
     * lemma, pos) the context came from. Otherwise, it is null.
     */
    private final List<Annotation> annotations;

    /**
     * Retrieve context words for the hits.
     *
     * @param hits hits to find contexts for
     * @param annotations the field and annotations to use for the context
     * @param contextSize how large the contexts need to be
     */
    private Contexts(Hits hits, List<Annotation> annotations, ContextSize contextSize) {
        if (annotations == null || annotations.isEmpty())
            throw new IllegalArgumentException("Cannot build contexts without annotations");

        // Make sure all hits have been read and get access to internal hits
        HitsInternal ha = hits.getInternalHits();
        List<String> matchInfoNames = hits.matchInfoNames();

        List<AnnotationForwardIndex> fis = new ArrayList<>();
        for (Annotation annotation: annotations) {
            fis.add(hits.index().annotationForwardIndex(annotation));
        }

        // Get the context
        // Group hits per document

        // setup first iteration
        final long size = ha.size(); // TODO ugly, might be slow because of required locking
        int prevDoc = size == 0 ? -1 : ha.doc(0);
        int firstHitInCurrentDoc = 0;
        contexts = new ObjectBigArrayBigList<>(hits.size());

        if (size > 0) {
            for (int i = 1; i < size; ++i) { // start at 1: variables already have correct values for primed for hit 0
                final int curDoc = ha.doc(i);
                if (curDoc != prevDoc) {
                    try { hits.threadAborter().checkAbort(); } catch (InterruptedException e) { throw new InterruptedSearch(e); }
                    // Process hits in preceding document:
                    int[][] docContextArray = getContextWordsSingleDocument(ha, firstHitInCurrentDoc, i, contextSize, fis, matchInfoNames);
                    Collections.addAll(contexts, docContextArray);
                    // start a new document
                    prevDoc = curDoc;
                    firstHitInCurrentDoc = i;
                }
            }
            // Process hits in final document
            int[][] docContextArray = getContextWordsSingleDocument(ha, firstHitInCurrentDoc, hits.size(), contextSize, fis, matchInfoNames);
            Collections.addAll(contexts, docContextArray);
        }

        this.annotations = new ArrayList<>(annotations);
    }

    /**
     * Count occurrences of context words around hit.
     *
     * @param hits hits to get collocations for
     * @param annotation annotation to use for the collocations, or null if default
     * @param contextSize how many words around hits to use
     * @param sensitivity what sensitivity to use
     * @param sort whether or not to sort the list by descending frequency
     *
     * @return the frequency of each occurring token
     */
    public synchronized static TermFrequencyList collocations(Hits hits, Annotation annotation, ContextSize contextSize, MatchSensitivity sensitivity, boolean sort) {
        BlackLabIndex index = hits.index();
        if (annotation == null)
            annotation = index.mainAnnotatedField().mainAnnotation();
        if (contextSize == null)
            contextSize = index.defaultContextSize();
        if (sensitivity == null)
            sensitivity = annotation.sensitivity(index.defaultMatchSensitivity()).sensitivity();

        List<Annotation> annotations = List.of(annotation);
        Contexts contexts = new Contexts(hits, annotations, contextSize);
        MutableIntIntMap countPerWord = IntIntMaps.mutable.empty();
        for (int[] context: contexts) {
            // Count words
            int contextHitStart = context[HIT_START_INDEX];
            int contextRightStart = context[RIGHT_START_INDEX];
            int contextLength = context[LENGTH_INDEX];
            int indexInContent = NUMBER_OF_BOOKKEEPING_INTS;
            for (int i = 0; i < contextLength; i++, indexInContent++) {
                if (i >= contextHitStart && i < contextRightStart)
                    continue; // don't count words in hit itself, just around [option..?]
                int wordId = context[indexInContent];
                int count;
                if (!countPerWord.containsKey(wordId))
                    count = 1;
                else
                    count = countPerWord.get(wordId) + 1;
                countPerWord.put(wordId, count);
            }
        }

        // Get the actual words from the sort positions
        Terms terms = index.annotationForwardIndex(contexts.annotations().get(0)).terms();
        Map<String, Integer> wordFreq = new HashMap<>();
        for (IntIntPair e : countPerWord.keyValuesView()) {
            int wordId = e.getOne();
            int count = e.getTwo();
            String word = sensitivity.desensitize(terms.get(wordId));
            // Note that multiple ids may map to the same word (because of sensitivity settings)
            // Here, those groups are merged.
            Integer mergedCount = wordFreq.get(word);
            if (mergedCount == null) {
                mergedCount = 0;
            }
            mergedCount += count;
            wordFreq.put(word, mergedCount);
        }

        // Transfer from map to list
        return new TermFrequencyList(hits.queryInfo(), wordFreq, sort);
    }

    /**
     * Get the field our current concordances were retrieved from
     *
     * @return the field name
     */
    List<Annotation> annotations() {
        return annotations;
    }

    /**
     * Iterate over the context arrays.
     *
     * Note that the order is unspecified.
     *
     * @return iterator
     */
    @Override
    public Iterator<int[]> iterator() {
        return contexts.iterator();
    }

    @Override
    public String toString() {
        return "Contexts(" + StringUtils.join(annotations, ", ") + ")";
    }

}
