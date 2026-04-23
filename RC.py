import os
import re 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

current_folder = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(current_folder, 'articles.parquet')).head(10000).copy()

columns_to_combine = ['prod_name', 'product_type_name', 'colour_group_name', 'detail_desc']
for col in columns_to_combine:
    df[col] = df[col].fillna('')

df['features'] = df['prod_name'] + " " + df['product_type_name'] + " " + df['colour_group_name'] + " " + df['detail_desc']

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['cleaned_features'] = df['features'].apply(clean_text) 

tfidf_vectorizer = TfidfVectorizer(stop_words='english') 
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_features'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['prod_name']).drop_duplicates()

def get_recommendations(prod_name, k=5, cosine_similarity=cosine_sim):
    if prod_name not in indices:
        return f"Item '{prod_name}' not found."

    idx = indices[prod_name]

    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    sim_scores = list(enumerate(cosine_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    unique_recommendations = []
    seen_names = set([prod_name]) 
    
    for i in sim_scores:
        item_idx = i[0] 
        name = df['prod_name'].iloc[item_idx] 
        
        if name not in seen_names: 
            unique_recommendations.append(item_idx)
            seen_names.add(name) 
            
        if len(unique_recommendations) == k: 
            break
            
    return df[['prod_name', 'product_type_name', 'detail_desc']].iloc[unique_recommendations]


test_product = df['prod_name'].iloc[4599]
test_product_2 = df['prod_name'].iloc[8888]


print("-" * 50)
print(f"CLIENT VIEWING PRODUCT: {test_product}")
print(f"Description: {df['detail_desc'].iloc[4599]}")
print("-" * 50)
print("SYSTEM RECOMMENDS TOP-5 SIMILAR PRODUCTS:")
print(get_recommendations(test_product))

print("-" * 50)
print(f"CLIENT VIEWING PRODUCT: {test_product_2}")
print(f"Description: {df['detail_desc'].iloc[8888]}")
print("-" * 50)
print("SYSTEM RECOMMENDS TOP-5 SIMILAR PRODUCTS:")
print(get_recommendations(test_product_2))
