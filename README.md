# SearchArray 

[![Python package](https://github.com/softwaredoug/searcharray/actions/workflows/test.yml/badge.svg)](https://github.com/softwaredoug/searcharray/actions/workflows/test.yml)

⛔️ Proceed with caution. Prototype code

Making search experimentation colab-notebook-able

Anytime I run an offline search relevance experiment, I have to standup a lot of systems. Something like Solr or Elasticsearch, maybe other services, components, vector databases, whatever.

Imagine the drain to velocity this entails.

This project creates a Pandas-centric way of creating and using a search index as just part of a Pandas array. In a sense, it simulates the functionality of the search engine, to allow anyone to prototype ideas, without external systems

You can see examples at this [colab notebook](https://colab.research.google.com/drive/1w_Ajn5rHzcISKhdCuPhhVFav3zrvKWn1?usp=sharing)

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

More use cases, like phrase search, can be seen [in the tests](https://github.com/softwaredoug/searcharray/blob/main/test/test_extension_array.py#L197)

## Goals 

This project is very much in prototype stage. 

The overall goals are to recreate a lot of the lexical features (term / phrase search) of a search engine like Solr or Elasticsearch, but in a dataframe. This includes more tokenization / text analysis features. As well as some wrappers that emulate the lexical query DSL of these search engines.

In the future, however, naive vector search likely will be added to assist in prototyping.

We care right now about relatively small scale "local" (or in colab environnment) prototyping of search ideas that could be promising for deeper investigation 100k-1m docs. We want to prioritize the offline / testing use case right now.

## TODOs / Future Work

* Make more memory efficient - underlying we use a Scipy sparse matrix, one for term freqs, another for positions. This can be cleaned up further.
* Flesh out wrapper functions that recreate most Solr / Elasticsearch query DSL functionality around term matching
* Testing on larger amounts of data
* Clean up the very janky code. This is very much a first pass
