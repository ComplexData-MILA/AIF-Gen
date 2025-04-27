### Seed Word Vocabularies

______________________________________________________________________

The seed word vocabularies were generated as follows:

**Model**: [Claude 3.5 Sonnet](https://claude.ai/new)

**Prompt**:

<blockquote>

Generate 100 distinct words related to the following topic:  \<topic_name>. Return the list of words in a single comma-seperated response.

</blockquote>

The _\<topic_name>_ was substituted according to the corresponding python file. (e.g. _education_ seed words in `education.py`)

_Note_: Resulting vocabulary lists were post-processed by removing duplicate words.
