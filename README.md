# SearchArray 

[![Python package](https://github.com/softwaredoug/searcharray/actions/workflows/test.yml/badge.svg)](https://github.com/softwaredoug/searcharray/actions/workflows/test.yml)

SearchArray turns Pandas string columns into a term index. It alows efficient BM25 scoring of phrases and individual tokens.

Think Lucene, but as a Pandas column.

```python
In[3]:  df['title_indexed'] = PostingsArray.index(df['title'])
        np.sort(df['title_indexed'].array.bm25('Cat'))
Out[3]: array([ 0.        ,  0.        ,  0.        , ..., 15.84568033,
                15.84568033, 15.84568033])
```

## Installation

```
pip install searcharray
```

## Motivation

Why do we treat Lucene-based, and other lexical search systems, like a special snowflake in the data stack? Many ML practitioners reach for a vector search solution, then realize they need to sprinkle in some degree of traditional lexical matching for the best solution. Indeed, in search, [hybrid search](https://www.pinecone.io/learn/hybrid-search-intro/) of vector+lexical solutions has shown to be most performant.

Let's break down the esoteric mystique of these systems, and tame them, so they just behave like other parts of the data stack.

SearchArray creates a Pandas-centric way of creating and using a search index as just part of a Pandas array. In a sense, it builds a search engine in Pandas - to allow anyone to prototype ideas, without external systems. 

You can see a full end-to-end search relevance experiment in this [colab notebook](https://colab.research.google.com/drive/1w_Ajn5rHzcISKhdCuPhhVFav3zrvKWn1?usp=sharing)

IE, take a dataframe that has a bunch of text, like movie title and overviews:

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
In[2]: df['title_indexed'] = PostingsArray.index(df['title'])
       df

Out[2]:
                                        title                                           overview                                      title_indexed
374430          Black Mirror: White Christmas  This feature-length special consists of three ...  PostingsRow({'Black': 1, 'Mirror:': 1, 'White'...
19404   The Brave-Hearted Will Take the Bride  Raj is a rich, carefree, happy-go-lucky second...  PostingsRow({'The': 1, 'Brave-Hearted': 1, 'Wi...
278                  The Shawshank Redemption  Framed in the 1940s for the double murder of h...  PostingsRow({'The': 1, 'Shawshank': 1, 'Redemp...
372058                             Your Name.  High schoolers Mitsuha and Taki are complete s...  PostingsRow({'Your': 1, 'Name.': 1}, {'Your': ...
238                             The Godfather  Spanning the years 1945 to 1955, a chronicle o...  PostingsRow({'The': 1, 'Godfather': 1}, {'The'...
...                                       ...                                                ...                                                ...
65513                          They Came Back  The lives of the residents of a small French t...  PostingsRow({'Back': 1, 'They': 1, 'Came': 1},...
65515                       The Eleventh Hour  An ex-Navy SEAL, Michael Adams, (Matthew Reese...  PostingsRow({'The': 1, 'Hour': 1, 'Eleventh': ...
65521                      Pyaar Ka Punchnama  Outspoken and overly critical Nishant Agarwal ...  PostingsRow({'Ka': 1, 'Pyaar': 1, 'Punchnama':...
32767                                  Romero  Romero is a compelling and deeply moving look ...        PostingsRow({'Romero': 1}, {'Romero': [0]})
65534                                  Poison  Paul Braconnier and his wife Blandine only hav...        PostingsRow({'Poison': 1}, {'Poison': [0]})```
```

Then search, getting top N with `Cat`

```
In[3]: np.sort(df['title_indexed'].array.bm25('Cat'))
Out[3]: array([ 0.        ,  0.        ,  0.        , ..., 15.84568033,
                15.84568033, 15.84568033])

In[4]: df['title_indexed'].bm25('Cat').argsort()
Out[4]: 

array([0, 18561, 18560, ..., 15038, 19012,  4392])
```

And since its just pandas, we can, of course just retrieve the top matches

```
In[5]: df.iloc[top_n_cat[-10:]]
Out[5]:
                  title                                           overview                                      title_indexed
24106     The Black Cat  American honeymooners in Hungary are trapped i...  PostingsRow({'Black': 1, 'The': 1, 'Cat': 1}, ...
12593     Fritz the Cat  A hypocritical swinging college student cat ra...  PostingsRow({'Cat': 1, 'the': 1, 'Fritz': 1}, ...
39853  The Cat Concerto  Tom enters from stage left in white tie and ta...  PostingsRow({'The': 1, 'Cat': 1, 'Concerto': 1...
75491   The Rabbi's Cat  Based on the best-selling graphic novel by Joa...  PostingsRow({'The': 1, 'Cat': 1, "Rabbi's": 1}...
57353           Cat Run  When a sexy, high-end escort holds the key evi...  PostingsRow({'Cat': 1, 'Run': 1}, {'Cat': [0],...
25508        Cat People  Sketch artist Irena Dubrovna (Simon) and Ameri...  PostingsRow({'Cat': 1, 'People': 1}, {'Cat': [...
11694        Cat Ballou  A woman seeking revenge for her murdered fathe...  PostingsRow({'Cat': 1, 'Ballou': 1}, {'Cat': [...
25078          Cat Soup  The surreal black comedy follows Nyatta, an an...  PostingsRow({'Cat': 1, 'Soup': 1}, {'Cat': [0]...
35888        Cat Chaser  A Miami hotel owner finds danger when be becom...  PostingsRow({'Cat': 1, 'Chaser': 1}, {'Cat': [...
6217         Cat People  After years of separation, Irina (Nastassja Ki...  PostingsRow({'Cat': 1, 'People': 1}, {'Cat': [...
```

More use cases can be seen [in the colab notebook](https://colab.research.google.com/drive/1w_Ajn5rHzcISKhdCuPhhVFav3zrvKWn1)

## Goals 

The overall goals are to recreate a lot of the lexical features (term / phrase search) of a search engine like Solr or Elasticsearch, but in a Pandas dataframe. 

### Memory efficient and fast text index

We want the index to be as memory efficient and fast at searching as possible. We want using it to have a minimal overhead.

We want you to be able to work with a reasonable dataset (1M-10M docs) relatively efficiently.

### Experimentation, reranking, functionality over scalability

Instead of building for 'big data' our goal is to build for for *small-data*. That is, focus on capabilities and expressiveness of Pandas, over limiting functionality in favor of scalability.

To this end, the applications of searcharray will tend to be focused on experimentation and top N candidate reranking. For experimentation, we want any ideas expressed in Pandas to have a somewhat clear path / "contract" in how they'd be implemented in a classical lexical search engine. For reranking, we want to load some top N results from a base system and be able to modify them.

### Make lexical search not a special snowflake in the ML stack

We know in search systems [hybrid search](https://www.pinecone.io/learn/hybrid-search-intro/) techniques dominate. Yet often its cast in terms of a giant, weird, big data lexical search engine that looks odd to most data scientists being joined with a vector database. We want lexical search to be more approachable to data scientists and ML engineers building these systems.

## Non-goals

### You need to bring your own tokenization

Currently tokenization (ie text analysis) is out of scope. There's enough Python libraries [that do this really well](https://github.com/snowballstem). Even exceeding what Lucene can do.

In SearchArray, a tokenizer is a function takes a string and emits a series of tokens. IE dumb, default whitespace tokenization:

```python
def ws_tokenizer(string):
    return string.split()
```

And you can pass any tokenizer that matches this signature to index:


```python
def ws_lowercase_tokenizer(string):
    return string.lower().split()

df['title_indexed'] = PostingsArray.index(df['title'], tokenizer=ws_lowercase_tokenizer)
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
df['score'] = df['title_indexed'].bm25('Cat') * df['hrs_into_past']
```

## TODOs / Future Work / Known issues

* Always more efficient
* Support tokenizers with overlapping positions (ie synonyms, etc)
* Add support for loading global term stats (ie doc freq) from external sources for more accurate representation
* Add minimum should match to each function
* Dumb vector search? Guessing other tools do this at small scale well enough.
