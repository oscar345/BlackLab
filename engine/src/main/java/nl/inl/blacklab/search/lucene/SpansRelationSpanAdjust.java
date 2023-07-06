package nl.inl.blacklab.search.lucene;

import java.io.IOException;
import java.util.Arrays;

import org.apache.lucene.search.spans.FilterSpans;

/**
 * Adjust relations spans to match source, target, or entire relation.
 */
class SpansRelationSpanAdjust extends BLFilterSpans<BLSpans> {

    /** how to adjust spans */
    private final RelationInfo.SpanMode mode;
    
    /** Adjusted start position of current hit */
    private int startAdjusted = -1;

    /** Adjusted end position of current hit */
    private int endAdjusted = -1;

    private HitQueryContext context;

    private MatchInfo[] matchInfo;

    /**
     * Constructs a SpansRelationSpanAdjust.
     *
     * @param in spans to adjust
     * @param mode how to adjust spans
     */
    public SpansRelationSpanAdjust(BLSpans in, RelationInfo.SpanMode mode) {
        super(in, SpanQueryRelationSpanAdjust.createGuarantees(in.guarantees(), mode));
        this.mode = mode;
    }

    @Override
    protected FilterSpans.AcceptStatus accept(BLSpans candidate) throws IOException {
        if (mode == RelationInfo.SpanMode.SOURCE && in.getRelationInfo().isRoot()) {
            // Need source, but this has no source
            return FilterSpans.AcceptStatus.NO;
        }
        setAdjustedStartEnd();
        return FilterSpans.AcceptStatus.YES;
    }

    @Override
    public int nextDoc() throws IOException {
        assert docID() != NO_MORE_DOCS;
        startAdjusted = endAdjusted = -1;
        return super.nextDoc();
    }

    @Override
    public int advance(int target) throws IOException {
        assert target >= 0 && target > docID();
        startAdjusted = endAdjusted = -1;
        return super.advance(target);
    }

    @Override
    protected boolean twoPhaseCurrentDocMatches() throws IOException {
        assert docID() >= 0 && docID() != NO_MORE_DOCS;
        startAdjusted = endAdjusted = -1;
        return super.twoPhaseCurrentDocMatches();
    }

    @Override
    public int startPosition() {
        return atFirstInCurrentDoc ? -1 : startAdjusted;
    }

    @Override
    public int endPosition() {
        return atFirstInCurrentDoc ? -1 : endAdjusted;
    }

    @Override
    public int nextStartPosition() throws IOException {
        assert startPosition() != NO_MORE_POSITIONS;
        super.nextStartPosition();
        setAdjustedStartEnd();
        return startAdjusted;
    }

    @Override
    protected void passHitQueryContextToClauses(HitQueryContext context) {
        super.passHitQueryContextToClauses(context);
        this.context = context;
    }

    private void setAdjustedStartEnd() {
        if (startPos == NO_MORE_POSITIONS) {
            startAdjusted = endAdjusted = NO_MORE_POSITIONS;
        } else if (atFirstInCurrentDoc || startPos < 0) {
            startAdjusted = endAdjusted = -1;
        } else {
            if (mode == RelationInfo.SpanMode.ALL_SPANS) {
                // We need all match info because we want the full span including all matched relations
                if (matchInfo == null)
                    matchInfo = new MatchInfo[context.numberOfMatchInfos()];
                else
                    Arrays.fill(matchInfo, null);
                in.getMatchInfo(matchInfo);
                startAdjusted = Integer.MAX_VALUE;
                endAdjusted = Integer.MIN_VALUE;
                for (int i = 0; i < matchInfo.length; i++) {
                    MatchInfo info = matchInfo[i];
                    if (info != null && info.getType() == MatchInfo.Type.RELATION) {
                        // This is a relations match. Take this into account for the full span.
                        // (capture groups are not taken into account, but should already fall into the span anyway)
                        if (info.getSpanStart() < startAdjusted)
                            startAdjusted = info.getSpanStart();
                        if (info.getSpanEnd() > endAdjusted)
                            endAdjusted = info.getSpanEnd();
                    }
                }
                if (startAdjusted == Integer.MAX_VALUE) {
                    // No relations matched; use the original span
                    startAdjusted = in.startPosition();
                    endAdjusted = in.endPosition();
                }
            } else {
                RelationInfo relationInfo = in.getRelationInfo();
                if (relationInfo == null) {
                    // No relation info available; use the original span
                    startAdjusted = in.startPosition();
                    endAdjusted = in.endPosition();
                } else {
                    startAdjusted = relationInfo.spanStart(mode);
                    endAdjusted = relationInfo.spanEnd(mode);
                }
            }
        }
    }

    @Override
    public void getMatchInfo(MatchInfo[] matchInfo) {
        if (this.matchInfo != null) {
            // We've already retrieved our clause's match info. Use that.
            for (int i = 0; i < matchInfo.length; i++) {
                if (matchInfo[i] == null)
                    matchInfo[i] = this.matchInfo[i];
            }
        } else {
            super.getMatchInfo(matchInfo);
        }
    }

    @Override
    public RelationInfo getRelationInfo() {
        return super.getRelationInfo();
    }

    @Override
    public int advanceStartPosition(int target) throws IOException {
        assert target > startPosition();
        if (atFirstInCurrentDoc && startPos >= target) {
            // Our cached hit is the one we want.
            return nextStartPosition();
        }
        if (mode != RelationInfo.SpanMode.FULL_SPAN) {
            // We can't skip because the spans we produce are not guaranteed to be sorted by start position.
            // Call the naive implementation.
            if (BLSpans.naiveAdvanceStartPosition(this, target) == NO_MORE_POSITIONS) {
                startPos = startAdjusted = endAdjusted = NO_MORE_POSITIONS;
            }
        } else {
            // We know our spans will be in order, so we can use the more efficient advanceStartPosition()
            if (in.advanceStartPosition(target) == NO_MORE_POSITIONS) {
                startPos = startAdjusted = endAdjusted = NO_MORE_POSITIONS;
            } else {
                setAdjustedStartEnd();
            }
        }
        return startAdjusted;
    }

    @Override
    public String toString() {
        return "RSPAN(" + in + ", " + mode + ")";
    }

}