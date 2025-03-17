import numpy as np
import jax
import jax.numpy as jnp
from typing import Any, List, Tuple, Callable, Optional


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
    brevity penalty.

    :param tokenized_a: jnp.ndarray of shape (num_seq_a, width_a) for candidate sentences.
    :param tokenized_b: jnp.ndarray of shape (num_seq_b, width_b) for reference sentences.
    :param pad_token_id: The integer id used for padding.
    :return: jnp.ndarray of shape (num_seq_a,) containing one BLEU score per candidate.
    """
    # Maximum n-gram order to consider
    max_n = 4

    def get_ngrams(sentence: jnp.ndarray, n: int, pad_token_id: int) -> jnp.ndarray:
        """Extract n-grams from a tokenized sentence.

        A sliding window is used over the fixed-length sentence to extract all n-grams.

        :param sentence: 1D jnp.ndarray of tokens.
        :param n: n-gram order.
        :param pad_token_id: Token id used for padding.
        :return: n-grams array of shape (seq_len-n+1, n)
        """
        L = sentence.shape[0]
        num_ngrams = L - n + 1
        # If sentence is too short, return empty fixed-size array
        empty_ngrams = jnp.full((num_ngrams, n), pad_token_id, dtype=sentence.dtype)

        def get_valid_ngrams() -> jnp.ndarray:
            # Create indices for sliding windows
            indices = jnp.arange(num_ngrams)[:, None] + jnp.arange(n)[None, :]
            return jnp.take(sentence, indices, axis=0)

        return jax.lax.cond(num_ngrams <= 0, lambda: empty_ngrams, get_valid_ngrams)

    def is_valid_ngram(ngram: jnp.ndarray, pad_token_id: int) -> jnp.bool_:
        """Check if an n-gram is valid (contains no padding tokens)."""
        return jnp.all(ngram != pad_token_id)

    def compute_precision(
        candidate: jnp.ndarray, n: int, pad_token_id: int
    ) -> jnp.ndarray:
        """Compute the modified n-gram precision for a candidate sentence."""
        # Extract candidate n-grams
        cand_ngrams = get_ngrams(candidate, n, pad_token_id)

        # Create validity mask for candidate n-grams
        cand_valid_mask = jax.vmap(is_valid_ngram, in_axes=(0, None))(
            cand_ngrams, pad_token_id
        )

        # Count total valid n-grams in candidate
        total_valid = jnp.sum(cand_valid_mask)
        total_valid = jnp.maximum(total_valid, 1)  # Avoid division by zero

        def empty_case() -> jnp.ndarray:
            return jnp.array(0.0, dtype=jnp.float32)

        def normal_case() -> jnp.ndarray:
            # Create a counter for clipped counts
            clipped_count_sum = jnp.array(0, dtype=jnp.int32)

            # Process each candidate n-gram individually to avoid dynamic shapes
            def process_ngram(i, acc) -> jnp.ndarray:
                ngram = cand_ngrams[i]
                is_valid = cand_valid_mask[i]

                # Count how many times this n-gram appears in candidate
                cand_count = jnp.sum(
                    jnp.all(cand_ngrams == ngram[None, :], axis=1) & cand_valid_mask
                )

                # Maximum count across references
                max_ref_count = jnp.array(0, dtype=jnp.int32)

                # For each reference, count matches
                def count_in_references(max_count, ref):
                    ref_ngrams = get_ngrams(ref, n, pad_token_id)
                    ref_valid_mask = jax.vmap(is_valid_ngram, in_axes=(0, None))(
                        ref_ngrams, pad_token_id
                    )

                    # Count matches in this reference
                    ref_count = jnp.sum(
                        jnp.all(ref_ngrams[:, None] == ngram[None, :], axis=2)
                        & ref_valid_mask[:, None]
                    )

                    return jnp.maximum(max_count, ref_count)

                max_ref_count = jax.lax.fori_loop(
                    0,
                    tokenized_b.shape[0],
                    lambda j, count: count_in_references(count, tokenized_b[j]),
                    max_ref_count,
                )

                # Add clipped count only if n-gram is valid
                clipped = jnp.minimum(cand_count, max_ref_count)
                return acc + (clipped * is_valid)

            # Process all n-grams
            clipped_sum = jax.lax.fori_loop(
                0, cand_ngrams.shape[0], process_ngram, clipped_count_sum
            )

            return clipped_sum / total_valid

        return jax.lax.cond(total_valid > 0, normal_case, empty_case)

    def find_closest_ref_length(
        ref_lens: jnp.ndarray, cand_len: jnp.ndarray
    ) -> jnp.ndarray:
        """Find the closest reference length following NLTK's approach."""
        # Calculate absolute differences between reference lengths and candidate length
        diffs = jnp.abs(ref_lens - cand_len)

        # Find the minimum difference
        min_diff = jnp.min(diffs)

        # Create a mask for references with the minimum difference
        is_min_diff = diffs == min_diff

        # Among those with min difference, take the shortest reference
        min_len_refs = jnp.where(is_min_diff, ref_lens, jnp.max(ref_lens) + 1)
        closest_ref_len = jnp.min(min_len_refs)

        return closest_ref_len

    def compute_bleu_for_candidate(candidate: jnp.ndarray) -> jnp.ndarray:
        """Compute the BLEU score for a single candidate sentence."""
        eps = 1e-8  # Smoothing constant to avoid log(0)

        # Compute precisions for n-grams of orders 1 to 4
        precisions = []
        for n in range(1, max_n + 1):
            p_n = compute_precision(candidate, n, pad_token_id)
            precisions.append(jnp.log(p_n + eps))  # Apply smoothing

        # Compute geometric mean of precisions
        geo_mean = jnp.exp(jnp.mean(jnp.stack(precisions)))

        # Compute candidate length as the count of non-pad tokens
        cand_len = jnp.sum(candidate != pad_token_id)

        # Compute valid length for each reference sentence
        ref_lens = jnp.sum(tokenized_b != pad_token_id, axis=1)

        # Find closest reference length using NLTK's approach
        closest_ref_len = find_closest_ref_length(ref_lens, cand_len)

        # Compute brevity penalty
        bp = jnp.where(
            cand_len > closest_ref_len,
            1.0,
            jnp.exp(1 - closest_ref_len / jnp.maximum(cand_len, 1)),
        )

        return bp * geo_mean

    # Compute BLEU scores for all candidate sentences in parallel
    bleu_scores = jax.vmap(compute_bleu_for_candidate)(tokenized_a)
    return bleu_scores


def preprocess_for_jax_bleu(
    texts: List[str],
    tokenizer,
    max_length: int = 512,
    min_length: int = 5,
    pad_token_id: int = -1,
) -> np.ndarray:
    """Preprocess texts using NLTK's tokenizer for JAX BLEU computation.

    :param texts: List of text strings.
    :param tokenizer: NLTK tokenizer function.
    :param max_length: Maximum sequence length.
    :param min_length: Minimum token count for a sequence to be included.
    :param pad_token_id: Appended to the right to reach max_length.
    :return: JAX array with padded token ids.
    """
    # Tokenize using NLTK
    tokenized_texts = [tokenizer(text) for text in texts]

    # Convert to token ids (assuming tokenizer doesn't do this)
    # Create vocabulary mapping from tokens to ids
    all_tokens = set()
    for tokens in tokenized_texts:
        all_tokens.update(tokens)

    token_to_id = {token: idx + 1 for idx, token in enumerate(sorted(all_tokens))}

    # Convert tokens to ids and pad
    tokenized_arrays = []
    for tokens in tokenized_texts:
        ids = [
            token_to_id.get(token, 1) for token in tokens
        ]  # Use 1 (UNK) for unseen tokens
        if len(ids) > max_length:
            ids = ids[:max_length]
        elif len(ids) < min_length:
            continue
        else:
            ids = ids + [pad_token_id] * (max_length - len(ids))
        tokenized_arrays.append(ids)

    # Convert to JAX array
    return np.array(tokenized_arrays, dtype=np.int32)
