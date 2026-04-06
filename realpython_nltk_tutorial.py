from __future__ import annotations

import contextlib
import io
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


def write_heading(buffer: io.StringIO, title: str) -> None:
    line = "=" * len(title)
    print(f"\n{title}\n{line}", file=buffer)


def write_block(buffer: io.StringIO, label: str, value) -> None:
    print(f"{label}:", file=buffer)
    print(value, file=buffer)
    print("", file=buffer)


def capture_stdout(func, *args, **kwargs) -> str:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        func(*args, **kwargs)
    return stream.getvalue().strip()


def ensure_nltk_data() -> None:
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "maxent_ne_chunker",
        "maxent_ne_chunker_tab",
        "words",
        "wordnet",
        "omw-1.4",
        "book",
    ]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            # Some resource names vary by NLTK version. We ignore failures here
            # and let the actual feature usage surface anything still missing.
            pass


def main() -> None:
    ensure_nltk_data()

    report = io.StringIO()

    write_heading(report, "Natural Language Processing With Python's NLTK Package")
    print(
        "Source tutorial: https://realpython.com/nltk-nlp-python/\n",
        file=report,
    )

    write_heading(report, "1. Tokenizing")
    example_string = """
Muad'Dib learned rapidly because his first training was in how to learn.
And the first lesson of all was the basic trust that he could learn.
It's shocking to find how many people do not believe they can learn,
and how many more believe learning to be difficult.
"""
    write_block(report, "Sentences", sent_tokenize(example_string))
    write_block(report, "Words", word_tokenize(example_string))

    write_heading(report, "2. Filtering Stop Words")
    worf_quote = "Sir, I protest. I am not a merry man!"
    words_in_quote = word_tokenize(worf_quote)
    stop_words = set(stopwords.words("english"))
    filtered_list_loop = []
    for word in words_in_quote:
        if word.casefold() not in stop_words:
            filtered_list_loop.append(word)
    filtered_list_comp = [
        word for word in words_in_quote if word.casefold() not in stop_words
    ]
    write_block(report, "Original tokens", words_in_quote)
    write_block(report, "Filtered with loop", filtered_list_loop)
    write_block(report, "Filtered with list comprehension", filtered_list_comp)

    write_heading(report, "3. Stemming")
    stemmer = PorterStemmer()
    string_for_stemming = (
        "The crew of the USS Discovery discovered many discoveries. "
        "Discovering is what explorers do."
    )
    words = word_tokenize(string_for_stemming)
    stemmed_words = [stemmer.stem(word) for word in words]
    write_block(report, "Original words", words)
    write_block(report, "Stemmed words", stemmed_words)

    write_heading(report, "4. Tagging Parts of Speech")
    sagan_quote = """
If you wish to make an apple pie from scratch,
you must first invent the universe.
"""
    words_in_sagan_quote = word_tokenize(sagan_quote)
    sagan_pos_tags = nltk.pos_tag(words_in_sagan_quote)
    write_block(report, "Carl Sagan POS tags", sagan_pos_tags)

    jabberwocky_excerpt = """
'Twas brillig, and the slithy toves did gyre and gimble in the wabe:
all mimsy were the borogoves, and the mome raths outgrabe.
"""
    words_in_excerpt = word_tokenize(jabberwocky_excerpt)
    jabberwocky_pos_tags = nltk.pos_tag(words_in_excerpt)
    write_block(report, "Jabberwocky POS tags", jabberwocky_pos_tags)

    write_heading(report, "5. Lemmatizing")
    lemmatizer = WordNetLemmatizer()
    string_for_lemmatizing = "The friends of DeSoto love scarves."
    lemmatized_words = [
        lemmatizer.lemmatize(word)
        for word in word_tokenize(string_for_lemmatizing)
    ]
    write_block(report, 'lemmatizer.lemmatize("scarves")', lemmatizer.lemmatize("scarves"))
    write_block(report, "Lemmatized sentence", lemmatized_words)
    write_block(report, 'lemmatizer.lemmatize("worst")', lemmatizer.lemmatize("worst"))
    write_block(
        report,
        'lemmatizer.lemmatize("worst", pos="a")',
        lemmatizer.lemmatize("worst", pos="a"),
    )

    write_heading(report, "6. Chunking")
    lotr_quote = "It's a dangerous business, Frodo, going out your door."
    words_in_lotr_quote = word_tokenize(lotr_quote)
    lotr_pos_tags = nltk.pos_tag(words_in_lotr_quote)
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    chunk_parser = nltk.RegexpParser(grammar)
    tree = chunk_parser.parse(lotr_pos_tags)
    write_block(report, "LOTR POS tags", lotr_pos_tags)
    write_block(report, "Chunk parse tree", tree)

    write_heading(report, "7. Chinking")
    chink_grammar = """
Chunk: {<.*>+}
       }<JJ>{
"""
    chink_parser = nltk.RegexpParser(chink_grammar)
    chink_tree = chink_parser.parse(lotr_pos_tags)
    write_block(report, "Chink parse tree", chink_tree)

    write_heading(report, "8. Named Entity Recognition (NER)")
    quote = """
Men like Schiaparelli watched the red planet—it is odd, by-the-bye, that
for countless centuries Mars has been the star of war—but failed to
interpret the fluctuating appearances of the markings they mapped so well.
All that time the Martians must have been getting ready.

During the opposition of 1894 a great light was seen on the illuminated
part of the disk, first at the Lick Observatory, then by Perrotin of Nice,
and then by other observers. English readers heard of it first in the
issue of Nature dated August 2.
"""

    def extract_ne(text: str) -> set[str]:
        tagged_words = nltk.pos_tag(word_tokenize(text))
        ne_tree = nltk.ne_chunk(tagged_words, binary=True)
        return {
            " ".join(token for token, _ in chunk)
            for chunk in ne_tree
            if hasattr(chunk, "label") and chunk.label() == "NE"
        }

    write_block(report, "Named entities", sorted(extract_ne(quote)))

    write_heading(report, "9. Getting Text to Analyze")
    nltk_book_import_output = capture_stdout(exec, "from nltk.book import *", globals())
    write_block(report, "Importing nltk.book", nltk_book_import_output)

    from nltk.book import text2, text8  # noqa: PLC0415

    write_heading(report, "10. Using a Concordance")
    concordance_man = capture_stdout(text8.concordance, "man")
    concordance_woman = capture_stdout(text8.concordance, "woman")
    write_block(report, 'text8.concordance("man")', concordance_man)
    write_block(report, 'text8.concordance("woman")', concordance_woman)

    write_heading(report, "11. Making a Dispersion Plot")
    text8.dispersion_plot(
        ["woman", "lady", "girl", "gal", "man", "gentleman", "boy", "guy"]
    )
    plt.close()
    write_block(report, "text8.dispersion_plot()", "Generated successfully")

    text2.dispersion_plot(["Allenham", "Whitwell", "Cleveland", "Combe"])
    plt.close()
    write_block(report, "text2.dispersion_plot()", "Generated successfully")

    write_heading(report, "12. Making a Frequency Distribution")
    frequency_distribution = FreqDist(text8)
    meaningful_words = [word for word in text8 if word.casefold() not in stop_words]
    meaningful_frequency_distribution = FreqDist(meaningful_words)
    write_block(report, "FreqDist(text8)", frequency_distribution)
    write_block(report, "Top 20 words in text8", frequency_distribution.most_common(20))
    write_block(
        report,
        "Top 20 non-stop-words in text8",
        meaningful_frequency_distribution.most_common(20),
    )
    meaningful_frequency_distribution.plot(20, cumulative=True)
    plt.close()
    write_block(report, "frequency_distribution.plot()", "Generated successfully")

    write_heading(report, "13. Finding Collocations")
    collocations_original = capture_stdout(text8.collocations)
    write_block(report, "text8.collocations()", collocations_original)
    lemmatized_text8_words = [lemmatizer.lemmatize(word) for word in text8]
    new_text = nltk.Text(lemmatized_text8_words)
    collocations_lemmatized = capture_stdout(new_text.collocations)
    write_block(report, "Lemmatized text8 collocations", collocations_lemmatized)

    print(report.getvalue())


if __name__ == "__main__":
    main()
