package nl.inl.blacklab.search.fimatch;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import org.apache.lucene.index.LeafReaderContext;

import net.jcip.annotations.NotThreadSafe;
import net.jcip.annotations.ThreadSafe;
import nl.inl.blacklab.codec.BLFieldsProducer;
import nl.inl.blacklab.forwardindex.ForwardIndexSegmentReader;
import nl.inl.blacklab.forwardindex.TermsSegmentReader;
import nl.inl.blacklab.search.BlackLabIndex;
import nl.inl.blacklab.search.BlackLabIndexAbstract;
import nl.inl.blacklab.search.indexmetadata.AnnotatedField;
import nl.inl.blacklab.search.indexmetadata.Annotation;
import nl.inl.blacklab.search.indexmetadata.AnnotationSensitivity;
import nl.inl.blacklab.search.indexmetadata.MatchSensitivity;
import nl.inl.blacklab.search.lucene.DocFieldLengthGetter;

/**
 * Allows the forward index matching subsystem to access the forward indices,
 * including an easy and fast way to read any annotation at any position from a
 * document.
 */
@ThreadSafe
public class ForwardIndexAccessorIntegrated extends ForwardIndexAccessorAbstract {

    public ForwardIndexAccessorIntegrated(BlackLabIndex index, AnnotatedField searchField) {
        super(index, searchField);
    }

    @Override
    public ForwardIndexAccessorLeafReader getForwardIndexAccessorLeafReader(LeafReaderContext readerContext) {
        return new ForwardIndexAccessorLeafReaderIntegrated(readerContext);
    }

    /**
     * Forward index accessor for a single LeafReader.
     *
     * Not thread-safe (only used from Spans).
     */
    @NotThreadSafe
    class ForwardIndexAccessorLeafReaderIntegrated implements ForwardIndexAccessorLeafReader {

        protected final LeafReaderContext readerContext;

        private final ForwardIndexSegmentReader forwardIndexReader;

        private final DocFieldLengthGetter lengthGetter;

        private final List<TermsSegmentReader> termsSegmentReaders = new ArrayList<>();

        ForwardIndexAccessorLeafReaderIntegrated(LeafReaderContext readerContext) {
            this.readerContext = readerContext;
            BLFieldsProducer fieldsProducer = BLFieldsProducer.get(readerContext, annotatedField.tokenLengthField());
            forwardIndexReader = fieldsProducer.forwardIndex();
            for (int i = 0; i < luceneFields.size(); i++) {
                termsSegmentReaders.add(forwardIndexReader.terms(luceneFields.get(i)));
            }
            lengthGetter = new DocFieldLengthGetter(readerContext.reader(), annotatedField.name());
        }

        @Override
        public ForwardIndexDocument advanceForwardIndexDoc(int docId) {
            return new ForwardIndexDocumentImpl(this, docId);
        }

        @Override
        public int getDocLength(int docId) {
            // NOTE: we subtract one because we always have an "extra closing token" at the end that doesn't
            //       represent a word, just any closing punctuation after the last word.
            return lengthGetter.getFieldLength(docId)
                    - BlackLabIndexAbstract.IGNORE_EXTRA_CLOSING_TOKEN;
        }

        @Override
        public int[] getChunk(int annotIndex, int docId, int start, int end) {
            Annotation annotation = annotations.get(annotIndex);
            AnnotationSensitivity sensitivity = annotation.hasSensitivity(
                    MatchSensitivity.SENSITIVE) ?
                    annotation.sensitivity(MatchSensitivity.SENSITIVE) :
                    annotation.sensitivity(MatchSensitivity.INSENSITIVE);
            int[] part = forwardIndexReader.retrievePart(sensitivity.luceneField(), docId, start, end);
            return terms.get(annotIndex).segmentIdsToGlobalIds(readerContext.ord, part);
        }

        @Override
        public int getNumberOfAnnotations() {
            return ForwardIndexAccessorIntegrated.this.numberOfAnnotations();
        }

        @Override
        public String getTermString(int annotIndex, int segmentTermId) {
            return termsSegmentReaders.get(annotIndex).get(segmentTermId);
        }

        @Override
        public boolean termsEqual(int annotIndex, int[] segmentTermIds, MatchSensitivity sensitivity) {
            return termsSegmentReaders.get(annotIndex).termsEqual(segmentTermIds, sensitivity);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof ForwardIndexAccessorIntegrated))
            return false;
        ForwardIndexAccessorIntegrated that = (ForwardIndexAccessorIntegrated) o;
        return index.equals(that.index) && annotatedField.equals(that.annotatedField);
    }

    @Override
    public int hashCode() {
        return Objects.hash(index, annotatedField);
    }
}
