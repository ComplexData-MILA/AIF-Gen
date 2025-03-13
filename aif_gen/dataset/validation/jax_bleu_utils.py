import jax
import jax.numpy as jnp
from typing import Any

@jax.jit
def calculate_bleu_similarity(
    tokenized_a: jnp.ndarray,
    tokenized_b: jnp.ndarray,
    pad_token_id: int,
) -> jnp.ndarray:
    """Compute BLEU score across multiple reference sentences in parallel.
    
    Each sentence in `tokenized_a` is compared against all sentences in `tokenized_b`
    to produce one BLEU score per candidate sentence.
    
    The BLEU score is computed as:
      BLEU = BP * exp(average_{n=1..4}(log(p_n + eps)))
    where p_n is the modified n-gram precision for n-grams of order n and BP is the
    brevity penalty. N‑gram counts are computed by first hashing each n‑gram into a single
    integer value, ignoring any n‑grams that contain the pad token.
    
    :param tokenized_a: jnp.ndarray of shape (num_seq_a, width_a) for candidate sentences.
    :param tokenized_b: jnp.ndarray of shape (num_seq_b, width_b) for reference sentences.
    :param pad_token_id: The integer id used for padding.
    :return: jnp.ndarray of shape (num_seq_a,) containing one BLEU score per candidate.
    """
    # Compute a base for hashing (assumes token ids are non-negative integers)
    global_max = jnp.maximum(jnp.max(tokenized_a), jnp.max(tokenized_b))
    base = global_max + 1

    def get_ngrams(sentence: jnp.ndarray, n: int, base: int, pad_token_id: int) -> jnp.ndarray:
        """Extract hashed n-grams from a tokenized sentence, ignoring n-grams with pad tokens.
        
        A sliding window is used over the fixed-length sentence (e.g. 512 tokens) to extract
        all n-grams. Any n-gram that contains the pad token is replaced with -1.
        
        :param sentence: 1D jnp.ndarray of tokens.
        :param n: n-gram order.
        :param base: Base for hashing (usually max(token)+1).
        :param pad_token_id: Token id used for padding.
        :return: 1D jnp.ndarray of hashed n-grams with shape (sentence_length - n + 1,).
        """
        L = sentence.shape[0]
        num_ngrams = L - n + 1
        if num_ngrams <= 0:
            return jnp.empty((0,), dtype=sentence.dtype)
        # Compute indices for sliding windows.
        indices = jnp.arange(num_ngrams)[:, None] + jnp.arange(n)[None, :]
        ngrams = sentence[indices]  # shape: (num_ngrams, n)
        # Determine which n-grams are fully valid (none of the tokens is a pad token).
        valid_mask = jnp.all(ngrams != pad_token_id, axis=1)
        # Hash each n-gram using a weighted sum.
        weights = base ** jnp.arange(n)
        hashed = jnp.sum(ngrams * weights, axis=1)
        # Replace any invalid n-gram with -1.
        return jnp.where(valid_mask, hashed, -1)

    def compute_precision(candidate: jnp.ndarray, n: int, base: int, pad_token_id: int) -> jnp.ndarray:
        """Compute the modified n-gram precision for a candidate sentence.
        
        For each unique candidate n-gram (ignoring any n-gram hashed to -1), the maximum count 
        over all reference sentences is computed and the candidate counts are clipped accordingly.
        A fixed maximum size and padding is specified when calling jnp.unique.
        
        :param candidate: 1D jnp.ndarray for the candidate sentence.
        :param n: n-gram order.
        :param base: Base for hashing.
        :param pad_token_id: Token id used for padding.
        :return: Modified n-gram precision as a scalar.
        """
        cand_ngrams = get_ngrams(candidate, n, base, pad_token_id)
        valid_mask = cand_ngrams != -1
        total = jnp.sum(valid_mask)
        total = jnp.maximum(total, 1)  # Avoid division by zero.
        max_unique_size = cand_ngrams.shape[0]
        unique_ngrams, counts = jnp.unique(
            cand_ngrams, return_counts=True, size=max_unique_size, fill_value=-1
        )

        def max_ref_count(ngram: int) -> int:
            """Return the maximum count of the given ngram across all reference sentences.
            
            If the ngram is -1 (i.e. invalid), return 0.
            """
            def compute_count() -> int:
                def count_in_ref(ref: jnp.ndarray) -> int:
                    ref_ngrams = get_ngrams(ref, n, base, pad_token_id)
                    return jnp.sum(ref_ngrams == ngram)
                counts_in_refs = jax.vmap(count_in_ref)(tokenized_b)
                return jnp.max(counts_in_refs)
            return jax.lax.cond(ngram < 0, lambda: 0, lambda: compute_count())

        max_counts = jax.vmap(max_ref_count)(unique_ngrams)
        clipped = jnp.minimum(counts, max_counts)
        clipped_sum = jnp.sum(clipped)
        return clipped_sum / total

    def compute_bleu_for_candidate(candidate: jnp.ndarray) -> jnp.ndarray:
        """Compute the BLEU score for a single candidate sentence.
        
        Modified n-gram precisions (for n = 1, 2, 3, 4) are computed, log-averaged, and then
        combined with a brevity penalty. The candidate and reference lengths are computed by 
        counting tokens that are not equal to the pad token.
        
        :param candidate: 1D jnp.ndarray for the candidate sentence.
        :return: BLEU score as a scalar.
        """
        eps = 1e-8  # Smoothing constant to avoid log(0)
        precisions = []
        for n in range(1, 5):
            p_n = compute_precision(candidate, n, base, pad_token_id)
            precisions.append(jnp.log(p_n + eps))
        geo_mean = jnp.exp(jnp.mean(jnp.stack(precisions)))
        
        # Compute candidate length as the count of non-pad tokens.
        cand_len = jnp.sum(candidate != pad_token_id)
        # For each reference sentence, compute its valid length.
        ref_lens = jnp.sum(tokenized_b != pad_token_id, axis=1)
        diff = jnp.abs(ref_lens - cand_len)
        best_ref_len = ref_lens[jnp.argmin(diff)]
        bp = jnp.where(cand_len > best_ref_len, 1.0, jnp.exp(1 - best_ref_len / cand_len))
        
        return bp * geo_mean

    # Compute BLEU scores for all candidate sentences in parallel.
    bleu_scores = jax.vmap(compute_bleu_for_candidate)(tokenized_a)
    return bleu_scores
