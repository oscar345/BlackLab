package nl.inl.blacklab.search.lucene;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.lucene.search.spans.FilterSpans;

/**
 * Captures all relations within a match info span.
 *
 * Note that the match info spans we're trying to capture relations in
 * may not overlap! If they do overlap, some relations may be skipped over.
 */
class SpansCaptureRelationsWithinSpan extends BLFilterSpans<BLSpans> {

    private final SpansRelations relations;

    /** Name of match info spans to capture relations from */
    final String toCapture;

    /** Group index of toCapture */
    private int toCaptureIndex = -1;

    /** Match info name for the list of captured relations */
    final String captureAs;

    /** Group index of captureAs */
    private int captureAsIndex = -1;

    /** Our hit query context */
    private HitQueryContext context;

    /** Match info for current hit */
    private MatchInfo[] matchInfo;

    /** List of relations captured for current hit */
    private List<RelationInfo> capturedRelations = new ArrayList<>();

    /**
     * Construct a SpansCaptureRelationsWithinSpan.
     *
     * @param clause clause we're capturing from
     * @param relations relations to capture
     * @param toCapture name of the match info span to capture relations from
     * @param captureAs name to capture the list of relations as
     */
    public SpansCaptureRelationsWithinSpan(BLSpans clause, SpansRelations relations, String toCapture, String captureAs) {
        super(clause);
        this.relations = relations;
        this.toCapture = toCapture;
        this.captureAs = captureAs;
    }

    @Override
    protected FilterSpans.AcceptStatus accept(BLSpans candidate) throws IOException {
        if (matchInfo == null) {
            matchInfo = new MatchInfo[context.getMatchInfoNames().size()];
        } else {
            Arrays.fill(matchInfo, null);
        }

        // We can "only" get match info for our own clause, but that should be enough
        // (we can only capture relations from match info captured within own clause)
        candidate.getMatchInfo(matchInfo);
        MatchInfo mi = matchInfo[toCaptureIndex];
        // We can only capture relations if we have the span
        if (mi != null) {
            int start = mi.getSpanStart();
            int end = mi.getSpanEnd();

            // Capture all relations within the toCapture span
            capturedRelations.clear();
            int docId = relations.docID();
            if (docId < candidate.docID())
                docId = relations.advance(candidate.docID());
            if (docId == candidate.docID()) {
                if (relations.startPosition() < start)
                    relations.advanceStartPosition(start);
                while (relations.endPosition() < end) {
                    capturedRelations.add(relations.getRelationInfo().copy());
                    relations.nextStartPosition();
                }
            }
            capturedRelations.sort(RelationInfo::compareTo);
            matchInfo[captureAsIndex] = new RelationListInfo(capturedRelations);
        }

        return FilterSpans.AcceptStatus.YES;
    }

    @Override
    public String toString() {
        return "RCAP(" + in + ", " + relations + ", " + toCapture + ", " + captureAs + ")";
    }

    @Override
    protected void passHitQueryContextToClauses(HitQueryContext context) {
        super.passHitQueryContextToClauses(context);
        this.context = context;
        this.toCaptureIndex = context.registerMatchInfo(toCapture);
        this.captureAsIndex = context.registerMatchInfo(captureAs);
    }

    @Override
    public void getMatchInfo(MatchInfo[] matchInfo) {
        // We've already gathered matchInfo in accept(); just copy it over
        int n = Math.min(matchInfo.length, this.matchInfo.length);
        for (int i = 0; i < n; i++) {
            if (this.matchInfo[i] != null)
                matchInfo[i] = this.matchInfo[i];
        }
    }

    @Override
    public boolean hasMatchInfo() {
        return true;
    }
}