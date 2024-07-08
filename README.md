# SearchArray 

[![Python package](https://github.com/softwaredoug/searcharray/actions/workflows/test.yml/badge.svg)](https://github.com/softwaredoug/searcharray/actions/workflows/test.yml) | Discuss at [Relevance Slack](https://join.slack.com/t/relevancy/shared_invite/zt-2ccxvzn1w-2_50zf9xBOSv3n06Cu15jg)

SearchArray turns Pandas string columns into a term index. It alows efficient BM25 / TFIDF scoring of phrases and individual tokens.

Think Lucene, but as a Pandas column.

```python
from searcharray import SearchArray
import pandas as pd

df['title_indexed'] = SearchArray.index(df['title'])
np.sort(df['title_indexed'].array.score(['cat', 'in', 'the', 'hat']))   # Search w/ phrase

> BM25 scores:
> array([ 0.        ,  0.        ,  0.        , ..., 15.84568033, 15.84568033, 15.84568033])
```

## Docs | Guide

SearchArray is documented in these notebooks:

[SearchArray Guide](https://colab.research.google.com/drive/1gmgVz53fDPTJakUHb6Mttevqry7gKHLM) | [SearchArray Offline Experiment](https://colab.research.google.com/drive/1w_Ajn5rHzcISKhdCuPhhVFav3zrvKWn1) | [About internals](https://softwaredoug.com/blog/2024/01/21/search-array-phrase-algorithm)

## Installation

```
pip install searcharray
```

## Features

* Search w/ terms by passing a string
* Search w/ a phrase by passing a list[str]
* Search w/ a [phrase w/ edit-distance](https://lucene.apache.org/core/9_6_0/core/org/apache/lucene/search/PhraseQuery.html) by passing slop=N.
* Access raw stats arrays in termfreqs / docfreqs methods on the array
* Bring your own tokenizer. Pass any (`def tokenize(value: str) -> List[str]`) when indexing.
* Memory map by passing `data_dir` to index for memory mapped index
* Accepts any python function to compute similarity. Here's [one similarity](https://github.com/softwaredoug/searcharray/blob/main/searcharray/similarity.py#L103)
* Scores the entire dataframe, allowing combination w/ other ranking attributes (recency, popularity, etc) or scores from other fields (ie boolean queries)
* Implement's Solr's [edismax query parser](https://github.com/softwaredoug/searcharray/blob/main/searcharray/solr.py) for efficient prototyping

## Motivation

To simplify lexical search in the Python data stack.

Many ML / AI practitioners reach for a vector search solution, then realize they need to sprinkle in some degree of BM25 / lexical search. Let's get traditional full-text search to behave like other parts of the data stack.

SearchArray creates a Pandas-centric way of creating and using a search index as just part of a Pandas array. In a sense, it builds a search engine in Pandas - to allow anyone to prototype ideas and perform reranking, without external systems. 

You can see a full end-to-end search relevance experiment in this [colab notebook](https://colab.research.google.com/drive/1w_Ajn5rHzcISKhdCuPhhVFav3zrvKWn1?usp=sharing)

IE, let's take a dataframe that has a bunch of text, like movie title and overviews:

```
In[1]: df = pd.DataFrame({'title': titles, 'overview': overviews}, index=ids)
Out[1]:
                                        title                                           overview
374430          Black Mirror: White Christmas  This feature-length special consists of three ...
19404   The Brave-Hearted Will Take the Bride  Raj is a rich, carefree, happy-go-lucky second...
278                  The Shawshank Redemption  Framed in the 1940s for the double murder of h...
372058                             Your Name.  High schoolers Mitsuha and Taki are complete s...
238                             The Godfather  Spanning the years 1945 to 1955, a chronicle o...
...                                       ...                                                ...
65513                          They Came Back  The lives of the residents of a small French t...
65515                       The Eleventh Hour  An ex-Navy SEAL, Michael Adams, (Matthew Reese...
65521                      Pyaar Ka Punchnama  Outspoken and overly critical Nishant Agarwal ...
32767                                  Romero  Romero is a compelling and deeply moving look ...
```

Index the text:

```
In[2]: df['title_indexed'] = SearchArray.index(df['title'])
       df

Out[2]:
                                        title                                           overview                                      title_indexed
374430          Black Mirror: White Christmas  This feature-length special consists of three ...  Terms({'Black', 'Mirror:', 'White'...
19404   The Brave-Hearted Will Take the Bride  Raj is a rich, carefree, happy-go-lucky second...  Terms({'The', 'Brave-Hearted', 'Wi...
278                  The Shawshank Redemption  Framed in the 1940s for the double murder of h...  Terms({'The', 'Shawshank', 'Redemp...
372058                             Your Name.  High schoolers Mitsuha and Taki are complete s...  Terms({'Your', 'Name.'}, {'Your': ...
238                             The Godfather  Spanning the years 1945 to 1955, a chronicle o...  Terms({'The', 'Godfather'}, {'The'...
...                                       ...                                                ...                                                ...
65513                          They Came Back  The lives of the residents of a small French t...  Terms({'Back', 'They', 'Came'},...
65515                       The Eleventh Hour  An ex-Navy SEAL, Michael Adams, (Matthew Reese...  Terms({'The', 'Hour', 'Eleventh': ...
65521                      Pyaar Ka Punchnama  Outspoken and overly critical Nishant Agarwal ...  Terms({'Ka', 'Pyaar', 'Punchnama':...
32767                                  Romero  Romero is a compelling and deeply moving look ...        Terms({'Romero'})
65534                                  Poison  Paul Braconnier and his wife Blandine only hav...        Terms({'Poison'})```
```

(notice the dumb tokenization - no worries you can pass your own tokenizer).

Then search, getting top N with `Cat`

```
In[3]: np.sort(df['title_indexed'].array.score('Cat'))
Out[3]: array([ 0.        ,  0.        ,  0.        , ..., 15.84568033,
                15.84568033, 15.84568033])

In[4]: df['title_indexed'].score('Cat').argsort()
Out[4]: 

array([0, 18561, 18560, ..., 15038, 19012,  4392])
```

And since its just pandas, we can, of course just retrieve the top matches

```
In[5]: df.iloc[top_n_cat[-10:]]
Out[5]:
                  title                                           overview                                      title_indexed
24106     The Black Cat  American honeymooners in Hungary are trapped i...  Terms({'Black': 1, 'The': 1, 'Cat': 1}, ...
12593     Fritz the Cat  A hypocritical swinging college student cat ra...  Terms({'Cat': 1, 'the': 1, 'Fritz': 1}, ...
39853  The Cat Concerto  Tom enters from stage left in white tie and ta...  Terms({'The': 1, 'Cat': 1, 'Concerto': 1...
75491   The Rabbi's Cat  Based on the best-selling graphic novel by Joa...  Terms({'The': 1, 'Cat': 1, "Rabbi's": 1}...
57353           Cat Run  When a sexy, high-end escort holds the key evi...  Terms({'Cat': 1, 'Run': 1}, {'Cat': [0],...
25508        Cat People  Sketch artist Irena Dubrovna (Simon) and Ameri...  Terms({'Cat': 1, 'People': 1}, {'Cat': [...
11694        Cat Ballou  A woman seeking revenge for her murdered fathe...  Terms({'Cat': 1, 'Ballou': 1}, {'Cat': [...
25078          Cat Soup  The surreal black comedy follows Nyatta, an an...  Terms({'Cat': 1, 'Soup': 1}, {'Cat': [0]...
35888        Cat Chaser  A Miami hotel owner finds danger when be becom...  Terms({'Cat': 1, 'Chaser': 1}, {'Cat': [...
6217         Cat People  After years of separation, Irina (Nastassja Ki...  Terms({'Cat': 1, 'People': 1}, {'Cat': [...
```

More use cases can be seen [in the colab notebook](https://colab.research.google.com/drive/1w_Ajn5rHzcISKhdCuPhhVFav3zrvKWn1)

## Goals 

The overall goals are to recreate a lot of the lexical features (term / phrase search) of a search engine like Solr or Elasticsearch, but in a Pandas dataframe. 

### Memory efficient and fast text index

We want the index to be as memory efficient and fast at searching as possible. We want using it to have a minimal overhead.

We want you to be able to work with a reasonable dataset (100X-1M docs) relatively efficiently for offline evaluation. And 1000s for fast reranking in a service.

### Experimentation, reranking, functionality over scalability

Instead of building for 'big data' our goal is to build for *small-data*. That is, focus on capabilities and expressiveness of Pandas, over limiting functionality in favor of scalability.

To this end, the applications of searcharray will tend to be focused on experimentation and top N candidate reranking. For experimentation, we want any ideas expressed in Pandas to have a somewhat clear path / "contract" in how they'd be implemented in a classical lexical search engine. For reranking, we want to load some top N results from a base system and be able to modify them.

### Make lexical search compatible with the data stack

We know in search, RAG, and other retrieval problems [hybrid search](https://www.pinecone.io/learn/hybrid-search-intro/) techniques dominate. Yet often its cast in terms of a giant, weird, big data lexical search engine that looks odd to most data scientists being joined with a vector database. We want lexical search to be more approachable to data scientists and ML engineers building these systems.

## Non-goals

### You need to bring your own tokenization

Python libraries [already do tokenization really well](https://github.com/snowballstem). Even exceeding what Lucene can do... giving you the ability to simulate and/or exceed the abilities of Lucene's tokenization.

In SearchArray, a tokenizer is a function takes a string and emits a series of tokens. IE dumb, default whitespace tokenization:

```python
def ws_tokenizer(string):
    return string.split()
```

And you can pass any tokenizer that matches this signature to index:

```python
def ws_lowercase_tokenizer(string):
    return string.lower().split()

df['title_indexed'] = SearchArray.index(df['title'], tokenizer=ws_lowercase_tokenizer)
```

Create your own using stemming libraries, or whatever Python functionality you want.

### Use Pandas instead of function queries

Solr has its [own unique function query syntax]()https://solr.apache.org/guide/7_7/function-queries.html. Elasticsearch has [Painless](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-scripting-painless.html).

Instead of recreating these, simply use Pandas on existing Pandas columns. Then later, if you need to implement this in Solr or Elasticsearch, attempt to recreate the functionality. Arguably what's in Solr / ES would be a subset of what you could do in Pandas.

```
# Calculate the number of hours into the past
df['hrs_into_past'] = (now - df['timestamp']).dt.total_seconds() / 3600
```

Then multiply by BM25 if you want:

```
df['score'] = df['title_indexed'].score('Cat') * df['hrs_into_past']
```

### Vector search

We focus on the lexical, ie "BM25-ish" and adjacent problems. There are other great tools for vector search out there.

## Need help?

Visit the [#searcharray channel on Relevance Slack](https://join.slack.com/t/relevancy/shared_invite/zt-2ccxvzn1w-2_50zf9xBOSv3n06Cu15jg)

## TODOs / Future Work / Known issues

* Always more efficient
* Support tokenizers with overlapping positions (ie synonyms, etc)
* Improve support for phrase slop
* Helper functions (like [this start at edismax](https://github.com/softwaredoug/searcharray/blob/main/searcharray/solr.py) that help recreate Solr / Elasticsearch lexical queries)
* Fuzzy search
* Efficient way to "slurp" some top N results from retrieval system into a dataframe
