/*******************************************************************************
 * Copyright (c) 2010, 2012 Institute for Dutch Lexicology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
package nl.inl.blacklab.resultproperty;

import java.util.Arrays;
import java.util.List;

import nl.inl.blacklab.forwardindex.Terms;
import nl.inl.blacklab.search.BlackLabIndex;
import nl.inl.blacklab.search.indexmetadata.AnnotatedField;
import nl.inl.blacklab.search.indexmetadata.AnnotatedFieldNameUtil;
import nl.inl.blacklab.search.indexmetadata.Annotation;
import nl.inl.blacklab.search.indexmetadata.MatchSensitivity;
import nl.inl.blacklab.search.results.Contexts;
import nl.inl.blacklab.search.results.Hits;

/**
 * A hit property for grouping on the context of the hit. Requires
 * HitConcordances as input (so we have the hit text available).
 */
public class HitPropertyWordLeft extends HitProperty {

    private String luceneFieldName;

    private Annotation annotation;

    private Terms terms;

    private MatchSensitivity sensitivity;

    public HitPropertyWordLeft(Hits hits, Annotation annotation, MatchSensitivity sensitivity) {
        super(hits);
        BlackLabIndex index = hits.queryInfo().index();
        this.annotation = annotation == null ? hits.queryInfo().field().annotations().main(): annotation;
        this.luceneFieldName = this.annotation.luceneFieldPrefix();
        this.terms = index.forwardIndex(this.annotation).terms();
        this.sensitivity = sensitivity;
    }

    public HitPropertyWordLeft(Hits hits, Annotation annotation) {
        this(hits, annotation, hits.queryInfo().index().defaultMatchSensitivity());
    }

    public HitPropertyWordLeft(Hits hits, MatchSensitivity sensitivity) {
        this(hits, null, sensitivity);
    }

    public HitPropertyWordLeft(Hits hits) {
        this(hits, null, hits.queryInfo().index().defaultMatchSensitivity());
    }

    public HitPropertyWordLeft(BlackLabIndex index, Annotation annotation, MatchSensitivity sensitivity) {
        super(null);
        this.annotation = annotation == null ? index.mainAnnotatedField().annotations().main(): annotation;
        this.terms = index.forwardIndex(this.annotation).terms();
        this.sensitivity = sensitivity;
    }

    public HitPropertyWordLeft(BlackLabIndex index, MatchSensitivity sensitivity) {
        this(index, null, sensitivity);
    }

    @Override
    public HitProperty copyWithHits(Hits newHits) {
        return new HitPropertyWordLeft(newHits, annotation, sensitivity);
    }

    @Override
    public HitPropValueContextWord get(int hitNumber) {
        int[] context = contexts.get(hitNumber);
        int contextHitStart = context[Contexts.HIT_START_INDEX];
        //int contextRightStart = context[Contexts.CONTEXTS_RIGHT_START_INDEX];
        int contextLength = context[Contexts.LENGTH_INDEX];

        if (contextHitStart <= 0)
            return new HitPropValueContextWord(hits, annotation, -1, sensitivity);
        int contextStart = contextLength * contextIndices.get(0) + Contexts.NUMBER_OF_BOOKKEEPING_INTS;
        return new HitPropValueContextWord(hits, annotation, context[contextStart
                + contextHitStart - 1], sensitivity);
    }

    @Override
    public int compare(Object i, Object j) {
        int[] ca = contexts.get((Integer) i);
        int caHitStart = ca[Contexts.HIT_START_INDEX];
        int caLength = ca[Contexts.LENGTH_INDEX];
        int[] cb = contexts.get((Integer) j);
        int cbHitStart = cb[Contexts.HIT_START_INDEX];
        int cbLength = cb[Contexts.LENGTH_INDEX];

        if (caHitStart <= 0)
            return cbHitStart <= 0 ? 0 : (reverse ? 1 : -1);
        if (cbHitStart <= 0)
            return reverse ? -1 : 1;
        // Compare one word to the left of the hit
        int contextIndex = contextIndices.get(0);

        int cmp = terms.compareSortPosition(
                ca[contextIndex * caLength + caHitStart - 1 + Contexts.NUMBER_OF_BOOKKEEPING_INTS],
                cb[contextIndex * cbLength + cbHitStart - 1 + Contexts.NUMBER_OF_BOOKKEEPING_INTS],
                sensitivity);
        return reverse ? -cmp : cmp;
    }

    @Override
    public List<Annotation> needsContext() {
        return Arrays.asList(annotation);
    }

    @Override
    public String getName() {
        return "word left";
    }

    @Override
    public List<String> getPropNames() {
        return Arrays.asList("word left: " + annotation.name());
    }

    @Override
    public String serialize() {
        String[] parts = AnnotatedFieldNameUtil.getNameComponents(luceneFieldName);
        String thePropName = parts.length > 1 ? parts[1] : "";
        return serializeReverse() + PropValSerializeUtil.combineParts("wordleft", thePropName, sensitivity.luceneFieldSuffix());
    }

    public static HitPropertyWordLeft deserialize(Hits hits, String info) {
        String[] parts = PropValSerializeUtil.splitParts(info);
        AnnotatedField field = hits.queryInfo().field();
        String propName = parts[0];
        if (propName.length() == 0)
            propName = AnnotatedFieldNameUtil.getDefaultMainAnnotationName();
        MatchSensitivity sensitivity = parts.length > 1 ? MatchSensitivity.fromLuceneFieldSuffix(parts[1]) : MatchSensitivity.SENSITIVE;
        Annotation annotation = field.annotations().get(propName);
        return new HitPropertyWordLeft(hits, annotation, sensitivity);
    }

}
