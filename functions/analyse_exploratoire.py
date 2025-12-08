import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ! $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def repartition_classes(df, col, type=""):
    df[col].value_counts().plot(kind='bar')

    plt.title(f"Répartition des classes - {type}")
    plt.xlabel(col)
    plt.ylabel("Nombre d'articles")
    plt.show()
 
 
# ! $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def analyse_len_text(df, col="text"):
    df['text_length'] = df[col].apply(len)
    
    print(f"======== \n{df.head(5)}")
    
    print("======== \n",df['text_length'].describe())
    
    plt.hist(df['text_length'], bins=50)
    plt.title("Distribution de la longueur des textes")
    plt.xlabel("Longueur du texte")
    plt.ylabel("Fréquence")
    plt.show()
    
# ! $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def plot_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

# ! $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$