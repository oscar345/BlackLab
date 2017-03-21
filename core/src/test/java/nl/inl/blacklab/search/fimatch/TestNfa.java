package nl.inl.blacklab.search.fimatch;

import org.junit.Assert;
import org.junit.Test;

public class TestNfa {

	class ForwardIndexDocumentString extends ForwardIndexDocument {

		private String input;

		ForwardIndexDocumentString(String input) {
			this.input = input;
		}

		@Override
		public int getToken(int propIndex, int pos) {
			if (!validPos(pos))
				return -1;
			return input.charAt(pos);
		}

		public boolean validPos(int pos) {
			return pos >= 0 && pos < input.length();
		}
	}

	@Test
	public void testNfaSimple() {
		// Test simple NFA matching ab|ba
		NfaState ab = NfaState.token(0, 'a', NfaState.token(0, 'b', null, "b"), "a");
		NfaState ba = NfaState.token(0, 'b', NfaState.token(0, 'a', null, "a"), "b");
		NfaState start = NfaState.or(ab, ba);

		ForwardIndexDocumentString fiDoc = new ForwardIndexDocumentString("abatoir");
		Assert.assertTrue(start.matches(fiDoc, 0, 1));
		Assert.assertTrue(start.matches(fiDoc, 1, 1));
		Assert.assertFalse(start.matches(fiDoc, 2, 1));
		Assert.assertFalse(start.matches(fiDoc, 6, 1));
	}

	@Test
	public void testNfaRepetition() {
		// Test NFA matching ac*e
		NfaState c = NfaState.token(0, 'c', null, "c");
		NfaState split = NfaState.or(c, NfaState.token(0, 'e', null, "e"));
		NfaState start = NfaState.token(0, 'a', split, "a");
		c.setNextState(0, split); // loopback

		// Forward matching
		Assert.assertTrue(start.matches(new ForwardIndexDocumentString("access"), 0, 1));
		Assert.assertTrue(start.matches(new ForwardIndexDocumentString("aces"), 0, 1));
		Assert.assertTrue(start.matches(new ForwardIndexDocumentString("aether"), 0, 1));
		Assert.assertFalse(start.matches(new ForwardIndexDocumentString("acquire"), 0, 1));
		Assert.assertFalse(start.matches(new ForwardIndexDocumentString("cesium"), 0, 1));

		// Backward matching
		Assert.assertTrue(start.matches(new ForwardIndexDocumentString("ideaal"), 3, -1));
	}

}
