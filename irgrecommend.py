import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data_path = r"C:\Users\Rajaguru Irusan\Documents\DS_Project\Final_Project\Recommendation_system\Sample_Items.xlsx"
data = pd.read_excel(data_path)

# Preprocess the data
data['Item Name'] = data['Item Name'].fillna('').str.lower()
data['Category Name'] = data['Category Name'].fillna('').str.lower()
data['combined_features'] = data['Item Name'] + ' ' + data['Category Name']

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(data['combined_features'])

# Streamlit app
st.title("GUVI Final Project: Product Recommendation System - IRG")

# Input product name with typing suggestion
product_input = st.text_input("Type the product", value="", placeholder="Start typing...")

if product_input:
    product_input = product_input.lower()
    matching_products = data[data['Item Name'].str.contains(product_input, na=False)]['Item Name'].head(50).tolist()
    if matching_products:
        st.write("Suggestions:")
        for suggestion in matching_products:
            st.markdown(f"- **{suggestion}**")

if st.button("Get Recommendation"):
    if product_input:
        product_input = product_input.lower()

        # Check if the input product exists in the dataset
        if product_input in data['Item Name'].values:
            # Compute similarity scores
            input_index = data[data['Item Name'] == product_input].index[0]
            similarity_scores = cosine_similarity(
                feature_matrix[input_index], feature_matrix
            )

            # Get top 5 recommendations
            similar_indices = similarity_scores[0].argsort()[-6:-1][::-1]
            recommended_products = data.iloc[similar_indices]

            # Sort recommendations by selling price (ascending) as an enhancement
            recommended_products = recommended_products.sort_values(by='Selling Price', ascending=True)

            # Display recommendations
            st.subheader("Recommended Products:")
            for _, row in recommended_products.iterrows():
                st.write(f"{row['Item Name']} (Price: {row['Selling Price']})")
        else:
            st.error("Product not found in the dataset. Please try another.")
    else:
        st.error("Please enter a product name.")
