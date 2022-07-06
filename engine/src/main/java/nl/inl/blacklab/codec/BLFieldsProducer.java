package nl.inl.blacklab.codec;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.Terms;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.Accountables;

import net.jcip.annotations.ThreadSafe;
import nl.inl.blacklab.forwardindex.ForwardIndexSegmentReader;

/**
 * Adds forward index reading to default FieldsProducer.
 *
 * Each index segment has an instance of BLFieldsProducer.
 * It opens the custom segment files for the forward index
 * (and any other custom files).
 *
 * Delegates all other methods to the default FieldsProducer.
 *
 * Adapted from <a href="https://github.com/meertensinstituut/mtas/">MTAS</a>.
 *
 * Thread-safe. It does store IndexInput which contains state, but those
 * are cloned whenever a thread needs to use them.
 */
@ThreadSafe
public class BLFieldsProducer extends FieldsProducer {

    protected static final Logger logger = LogManager.getLogger(BLFieldsProducer.class);

    /**
     * Get the BLFieldsProducer for the given leafreader.
     *
     * The luceneField must be any existing Lucene field in the index.
     * It doesn't matter which. This is because BLTerms is used as an
     * intermediate to get access to BLFieldsProducer.
     *
     * The returned BLFieldsProducer is not specific for the specified field,
     * but can be used to read information related to any field from the segment.
     *
     * @param lrc leafreader to get the BLFieldsProducer for
     * @param luceneField name of any Lucene field in the index
     * @return BLFieldsProducer for this leafreader
     */
    public static BLFieldsProducer get(LeafReaderContext lrc, String luceneField) {
        try {
            BLTerms terms = (BLTerms)(lrc.reader().terms(luceneField));
            return terms.getFieldsProducer();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /** Our codec name (BLCodec) */
    private final String postingsFormatName;

    /** Name of PF we delegate to (the one from Lucene) */
    private String delegatePostingsFormatName;

    /** The delegate whose functionality we're extending */
    private final FieldsProducer delegateFieldsProducer;

    /** The forward index */
    private final SegmentForwardIndex forwardIndex;

    public BLFieldsProducer(SegmentReadState state, String postingsFormatName)
            throws IOException {
        this.postingsFormatName = postingsFormatName;

        // NOTE: opening the forward index calls openInputFile, which reads
        //       delegatePostingsFormatName, so this must be done first.
        forwardIndex = new SegmentForwardIndex(this, state);

        PostingsFormat delegatePostingsFormat = PostingsFormat.forName(this.delegatePostingsFormatName);
        this.delegateFieldsProducer = delegatePostingsFormat.fieldsProducer(state);
    }

    @Override
    public Iterator<String> iterator() {
        return delegateFieldsProducer.iterator();
    }

    @Override
    public void close() throws IOException {
        forwardIndex.close();
        delegateFieldsProducer.close();
    }

    @Override
    public Terms terms(String field) throws IOException {
        return new BLTerms(delegateFieldsProducer.terms(field), this);
    }

    @Override
    public int size() {
        return delegateFieldsProducer.size();
    }

    @Override
    public long ramBytesUsed() {
        // return BASE_RAM_BYTES_USED + delegateFieldsProducer.ramBytesUsed();

        // TODO: copied from mtas, improve this estimate?
        return 3 * delegateFieldsProducer.ramBytesUsed();
    }

    @Override
    public Collection<Accountable> getChildResources() {
        List<Accountable> resources = new ArrayList<>(delegateFieldsProducer.getChildResources());
        resources.add(Accountables.namedAccountable("delegate", delegateFieldsProducer));
        return Collections.unmodifiableList(resources);
    }

    @Override
    public void checkIntegrity() throws IOException {
        delegateFieldsProducer.checkIntegrity();
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "(delegate=" + delegateFieldsProducer + ")";
    }

    /**
     * Open a custom file for reading and check the header.
     *
     * @param state segment read state
     * @param extension extension of the file to open (will automatically be prefixed with "bl")
     * @return handle to the opened segment file
     */
    IndexInput openIndexFile(SegmentReadState state, String extension) throws IOException {
        String fileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, "bl" + extension);
        IndexInput input = state.directory.openInput(fileName, state.context);
        try {
            // Check index header
            CodecUtil.checkIndexHeader(input, postingsFormatName, BLCodecPostingsFormat.VERSION_START,
                    BLCodecPostingsFormat.VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);

            // Check delegate postings format name
            String delegatePFN = input.readString();
            if (delegatePostingsFormatName == null)
                delegatePostingsFormatName = delegatePFN;
            if (!delegatePostingsFormatName.equals(delegatePFN))
                throw new IOException("Segment file " + fileName +
                        " contains wrong delegate postings format name: " + delegatePFN +
                        " (expected " + delegatePostingsFormatName + ")");

            return input;
        } catch (Exception e) {
            input.close();
            throw e;
        }
    }

    /**
     * Create a forward index reader for this segment.
     *
     * The returned reader is not threadsafe and shouldn't be stored.
     * A single thread may use it for reading from this segment. It
     * can then be discarded.
     *
     * @return forward index segment reader
     */
    public ForwardIndexSegmentReader forwardIndex() {
        return forwardIndex.reader();
    }

}
