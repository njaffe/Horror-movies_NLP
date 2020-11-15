# Project_4: Using NLP to topic model horror movies

My Metis project 4.

## 1. Background
In this project, used topic modeling techniques to identify subgenres within the horror genre.

My goal was to see if I could model specific subgenres from script data; that is, people speaking.

## 2. Workflow
- Webscraped 125 horror movie scripts from [IMSDB](https://www.imsdb.com/) using BeautifulSoup
- Cleaned and tokenized corpus using regular expressions and NLTK
- Vectorized scripts using CV and TFIDF
- Performed topic modeling using PCA, SVD and NMF
- Interpreted topics by examining most expressed words in each topic and most expressed topic in each movie script, and by using domain knowledge 
- Performed K-means clustering to determine relationships among movies in SVD space
- Visualized high-dimensionality SVD space in 2 dimensions using t-SNE

## 3. Results summary
- Six SVD components yielded clearest subgenres:
  - Suburbia, “trapped in a house” trope, miscellaneous
  - Gothic horror
  - Science fiction horror
  - "Magical realism" horror
  - Gory, campy horror
  - Horror centered around a central, charismatic protagonist or centered around animals
  
 - Gothic horror and science fiction horror were most clearest topics

## 4. Takeaways and future work
Horror movies allow viewers to face their fears in a safe space. Thus, specific subgenres appear to represent deep-seated fears, such as the unknown, death, madness, and intrusion into one's home or neighborhood by violent forces. Using NLP, I was able to gain insight into these genres and get an idea of which movies belong to which subgenre.

The next step in this project could be to further delve into the less clearly defined subgenres and using tools such as [CorEx](https://pypi.org/project/corextopic/) anchoring to split the corpus into further subgenres that may be more cohesive and clearly defined.

Additionaly, horror movies represent society's fears, and these fears change over time. Thus an interesting next step of this project could be to examine my topics through time and assess whether different subgenres tend to be associated with different time periods. 


## 5. Tools and techniques used
- [Jupyter](https://jupyter.org/)
- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/index.html)
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [NLTK](https://www.nltk.org/)
- [Spacy](https://spacy.io/)
- [count vectorization](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 
- [term frequency–inverse document frequency vectorization](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [word2vec with Gensim](https://radimrehurek.com/gensim/models/word2vec.html)
- [SVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
- [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
- [K-means clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

Special thanks to Brian McGarry and Richard Chiou for assistance with experimental design and implementation, and to Anterra Kennedy for help with pipeline and experimental design.

