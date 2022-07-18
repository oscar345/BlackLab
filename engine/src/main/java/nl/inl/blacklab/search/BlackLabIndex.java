package nl.inl.blacklab.search;

import java.io.File;
import java.io.IOException;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.text.Collator;
import java.util.List;
import java.util.Set;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import nl.inl.blacklab.exceptions.BlackLabRuntimeException;
import nl.inl.blacklab.exceptions.ErrorOpeningIndex;
import nl.inl.blacklab.exceptions.IndexVersionMismatch;
import nl.inl.blacklab.forwardindex.AnnotationForwardIndex;
import nl.inl.blacklab.forwardindex.FiidLookup;
import nl.inl.blacklab.forwardindex.ForwardIndex;
import nl.inl.blacklab.search.indexmetadata.AnnotatedField;
import nl.inl.blacklab.search.indexmetadata.AnnotatedFields;
import nl.inl.blacklab.search.indexmetadata.Annotation;
import nl.inl.blacklab.search.indexmetadata.AnnotationSensitivity;
import nl.inl.blacklab.search.indexmetadata.Field;
import nl.inl.blacklab.search.indexmetadata.IndexMetadata;
import nl.inl.blacklab.search.indexmetadata.MatchSensitivity;
import nl.inl.blacklab.search.indexmetadata.MetadataField;
import nl.inl.blacklab.search.indexmetadata.MetadataFields;
import nl.inl.blacklab.search.lucene.BLSpanQuery;
import nl.inl.blacklab.search.lucene.DocIntFieldGetter;
import nl.inl.blacklab.search.results.ContextSize;
import nl.inl.blacklab.search.results.DocResults;
import nl.inl.blacklab.search.results.Hits;
import nl.inl.blacklab.search.results.SearchSettings;
import nl.inl.blacklab.searches.SearchCache;
import nl.inl.blacklab.searches.SearchEmpty;
import nl.inl.util.VersionFile;
import nl.inl.util.XmlHighlighter.UnbalancedTagsStrategy;

public interface BlackLabIndex extends AutoCloseable {

    /**
     * Default number of context words to return around a hit.
     */
    ContextSize DEFAULT_CONTEXT_SIZE = ContextSize.get(5);

    // Static [factory] methods
    //---------------------------------------------------------------
    
    /**
     * Does the specified directory contain a BlackLab index?
     *
     * NOTE: does NOT follow symlinks.
     * 
     * @param indexDir the directory
     * @return true if it's a BlackLab index, false if not.
     */
    static boolean isIndex(File indexDir) {
        try {
            if (VersionFile.exists(indexDir)) {
                VersionFile vf = VersionFile.read(indexDir);
                String version = vf.getVersion();
                return vf.getType().equals("blacklab") && (version.equals("1") || version.equals("2"));
            }

            if (BlackLab.isFeatureEnabled(BlackLab.FEATURE_INTEGRATE_EXTERNAL_FILES)) {
                // Just see if it's a Lucene index and assume it's a BlackLab index if so.
                // (Lucene index always has a segments_* file)
                try (Directory dir = FSDirectory.open(indexDir.toPath())){
                    return DirectoryReader.indexExists(dir);
                }
            }

            return false;
        } catch (IOException e) {
            throw BlackLabRuntimeException.wrap(e);
        }
    }

    /**
     * Does the specified directory contain a BlackLab index?
     *
     * NOTE: does NOT follow symlinks. Call {@link Path#toRealPath(LinkOption...)}
     * yourself if you want this.
     *
     * @param indexDirPath the directory
     * @return true if it's a BlackLab index, false if not.
     */
    static boolean isIndex(Path indexDirPath) {
        return isIndex(indexDirPath.toFile());
    }

    /**
     * Open an index for reading ("search mode").
     * 
     * @param blackLab our BlackLab instance
     * @param indexDir the index directory
     * @return index object
     * @throws IndexVersionMismatch if the index format is no longer supported
     * @throws ErrorOpeningIndex on any error
     * @deprecated use {@link BlackLab#open(File)} or {@link BlackLabEngine#open(File)} instead
     */
    @Deprecated
    static BlackLabIndex open(BlackLabEngine blackLab, File indexDir) throws ErrorOpeningIndex  {
        return blackLab.open(indexDir);
    }

    /**
     * TODO: consolidate fiid stuff, push down to implementation
     */
    ForwardIndex createForwardIndex(AnnotatedField field);

    /**
     * TODO: consolidate fiid stuff, push down to implementation
     */
    DocIntFieldGetter createFiidGetter(LeafReader reader, Annotation annotation);

    /**
     * Get FiidLookups for the specified annotations.
     *
     * If any of the entries in the list is null, a corresponding null will be added
     * to the result list, so the indexes of the result list will match the indexes
     * of the input list.
     *
     * TODO: consolidate fiid stuff, push down to implementation
     *
     * @param annotations annotations to get FiidLookup for
     * @param enableRandomAccess if true, random access will be enabled for the returned objects
     * @return FiidLookup objects for the specfied annotations
     */
    List<FiidLookup> getFiidLookups(List<Annotation> annotations, boolean enableRandomAccess);

    /**
     * We want to call getFiid() with a Document. Add the fields we'll need.
     *
     * Use this to make sure the required fields will be loaded when retrieving
     * the Lucene Document.
     *
     * May or may not add any fields, depending on the index format.
     *
     * TODO: probably push this down to a separate implementation class for each index format
     *
     * @param annotations annotations we want to access forward index for
     * @param fieldsToLoad (out) required fields will be added here
     */
    void prepareForGetFiidCall(List<Annotation> annotations, Set<String> fieldsToLoad);

    /**
     * Given the Lucene docId, return the forward index id.
     *
     * If all files are contained in the index, the docId and forward
     * index id are the same.
     *
     * TODO: probably push this down to a separate implementation class for each index format
     *
     * @param annotation annotation to get the fiid for
     * @param docId Lucene doc id
     * @param doc Lucene document if available, or null otherwise
     * @return the forward index id
     */
    int getFiid(Annotation annotation, int docId, Document doc);

    // Basic stuff, low-level access to index
    //---------------------------------------------------------------
    
    @Override
    boolean equals(Object obj);
    
    @Override
    int hashCode();
    
    /**
     * Finalize the index object. This closes the IndexSearcher and (depending on
     * the constructor used) may also close the index reader.
     */
    @Override
    void close();

    /**
     * Is this a newly created, empty index?
     * 
     * @return true if it is, false if not
     */
    boolean isEmpty();

    /**
     * Is the document id in range, and not a deleted document?
     * @param docId document id to check
     * @return true if it is an existing document
     */
    boolean docExists(int docId);

    /**
     * Perform a task on each (non-deleted) Lucene Document.
     * 
     * @param task task to perform
     */
    void forEachDocument(DocTask task);


    // Search the index
    //---------------------------------------------------------------------------
    
    /**
     * Find hits for a pattern in a field.
     * 
     * Uses the default search settings.
     *
     * @param query the pattern to find
     * @return the hits found
     */
    default Hits find(BLSpanQuery query) {
        return find(query, null);
    }

    /**
     * Find hits for a pattern in a field.
     * 
     * @param query the pattern to find
     * @param settings search settings, or null for default
     * @return the hits found
     */
    Hits find(BLSpanQuery query, SearchSettings settings);

    /**
     * Perform a document query only (no hits)
     * 
     * @param documentFilterQuery the document-level query
     * @return the matching documents
     */
    DocResults queryDocuments(Query documentFilterQuery);

    /**
     * Determine the term frequencies for an annotation sensitivity.
     * 
     * @param annotSensitivity the annotation + sensitivity indexing we want the term frequency for
     * @param filterQuery document filter, or null for all documents
     * @param terms a list of terms to retrieve frequencies for, or null/empty to retrieve frequencies for all terms
     * @return term frequencies
     */
    TermFrequencyList termFrequencies(AnnotationSensitivity annotSensitivity, Query filterQuery, Set<String> terms);

    /**
     * Explain how a SpanQuery is rewritten to an optimized version to be executed
     * by Lucene.
     *
     * @param query the query to explain
     * @return the explanation
     */
    QueryExplanation explain(BLSpanQuery query);
    
    /**
     * Start building a Search.
     *
     * @param field field to search
     * @param useCache whether to use the cache or bypass it
     * @return empty search object
     */
    SearchEmpty search(AnnotatedField field, boolean useCache);

    /**
     * Start building a Search. 
     * 
     * @param field field to search
     * @return empty search object
     */
    default SearchEmpty search(AnnotatedField field) {
        return search(field, true);
    }
    
    /**
     * Start building a Search.
     *
     * Uses the main annotated field, e.g. usually called "contents".
     *
     * @return empty search object
     */
    default SearchEmpty search() {
        return search(mainAnnotatedField());
    }

    
    // Access the different modules of the index
    //---------------------------------------------------------------------------
    
    /**
     * Get the Lucene index reader we're using.
     *
     * @return the Lucene index reader
     */
    IndexReader reader();

    IndexSearcher searcher();

    /**
     * Get the content accessor for a field.
     * 
     * @param field the field
     * @return the content accessor, or null if there is no content accessor for this field
     */
    ContentAccessor contentAccessor(Field field);

    /**
     * Tries to get the ForwardIndex object for the specified field name.
     *
     * Looks for an already-opened forward index first. If none is found, and if
     * we're in "create index" mode, may create a new forward index. Otherwise,
     * looks for an existing forward index and opens that.
     *
     * @param annotation the annotation for which we want the forward index
     * @return the ForwardIndex if found/created
     * @throws BlackLabRuntimeException if the annotation has no forward index
     */
    AnnotationForwardIndex annotationForwardIndex(Annotation annotation);

    /**
     * Get forward index for the specified annotated field.
     * 
     * @param field field to get forward index for
     * @return forward index
     */
    ForwardIndex forwardIndex(AnnotatedField field);


    
    // Information about the index
    //---------------------------------------------------------------------------
    
    /**
     * Get the index name.
     * 
     * Usually the name of the directory the index is in.
     * 
     * @return index name
     */
    String name();

    /**
     * Get the index directory.
     * 
     * @return index directory
     */
    File indexDirectory();

    /**
     * Get information about the structure of the BlackLab index.
     *
     * @return the structure object
     */
    IndexMetadata metadata();

    /**
     * Get a field (either an annotated or a metadata field).
     * 
     * @param fieldName name of the field
     * @return the field
     */
    default Field field(String fieldName) {
        Field field = annotatedField(fieldName);
        if (field == null)
            field = metadataField(fieldName);
        return field;
    }

    default AnnotatedFields annotatedFields() {
        return metadata().annotatedFields();
    }

    default AnnotatedField annotatedField(String fieldName) {
        return metadata().annotatedField(fieldName);
    }

    default AnnotatedField mainAnnotatedField() {
        return metadata().mainAnnotatedField();
    }
    
    default MetadataFields metadataFields() {
        return metadata().metadataFields();
    }

    /**
     * Get the specified metadata field config.
     *
     * @param fieldName metadata field name
     * @return metadata field config
     * @throws IllegalArgumentException if field not found
     */
    default MetadataField metadataField(String fieldName) {
        return metadata().metadataField(fieldName);
    }

    
    // Get settings
    //---------------------------------------------------------------------------

    /**
     * The default settings for all new Hits objects.
     *
     * You may change these settings; this will affect all new Hits objects.
     *
     * @return settings object
     */
    SearchSettings searchSettings();

    /**
     * How do we fix well-formedness for snippets of XML?
     * 
     * @return the setting: either adding or removing unbalanced tags
     */
    UnbalancedTagsStrategy defaultUnbalancedTagsStrategy();

    MatchSensitivity defaultMatchSensitivity();

    /**
     * Get the default initial query execution context.
     *
     * @param field field to search
     * @return query execution context
     */
    QueryExecutionContext defaultExecutionContext(AnnotatedField field);

    /**
     * Get the collator being used for sorting.
     *
     * @return the collator
     */
    Collator collator();

    /**
     * Get the analyzer for indexing and searching.
     * 
     * @return the analyzer
     */
    Analyzer analyzer();
    
    /**
     * Get the default context size.
     * @return default context size
     */
    ContextSize defaultContextSize();

    /**
     * Are we running in index mode?
     * @return true if we are, false if not
     */
    boolean indexMode();


    // Methods that mutate settings
    //---------------------------------------------------------------------------

    /**
     * Set the collator used for sorting.
     *
     * The default collator is for English.
     *
     * @param collator the collator
     */
    void setCollator(Collator collator);

    /**
     * Set the default sensitivity for queries.
     * @param m default match sensitivity
     */
    void setDefaultMatchSensitivity(MatchSensitivity m);

    /**
     * Set the maximum number of hits to process/count.
     * @param settings desired settings
     */
    void setSearchSettings(SearchSettings settings);
    
    /**
     * Set the default context size.
     * @param size default context size
     */
    void setDefaultContextSize(ContextSize size);

    /**
     * Set the object BlackLab should use as cache.
     * 
     * BlackLab will notify the cache of search results and will ask
     * the cache for results before executing a search.
     * 
     * It is up to the application to implement an effective cache, deciding
     * whether to cache a result and ensuring the cache doesn't grow too large.
     * 
     * @param cache cache object to use
     */
    void setCache(SearchCache cache);

    SearchCache cache();

    /**
     * Get the BlackLab instance that created us.
     * @return BlackLab instance
     */
    BlackLabEngine blackLab();

    default Document luceneDoc(int docId) {
        try {
            return reader().document(docId);
        } catch (IOException e) {
            throw new BlackLabRuntimeException(e);
        }
    }
}
