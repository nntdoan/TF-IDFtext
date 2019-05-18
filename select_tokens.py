import pandas as pd
# Extract the 50 highest-scored verbs for each document (romance and non-romance)
# given this verb scores 0 in the other document

# Making dataframe
df = pd.read_csv("output/tf_idf_romance_matrix_v4.csv")

# 50 highest-scored verbs for Romance, while scoring 0 for Non-romance
df_romance = df.loc[df.non_romance == 0.0]
select_romance_verbs = df_romance.nlargest(54, ['romance'])
export_selected_romance = select_romance_verbs.to_csv(f'output/highest_tf_idf_romance_verbs_x50.csv', header=True)

# 50 highest-scored verbs for Non-romance, while scoring 0 for Romance
df_non_romance = df.loc[df.romance == 0.0]
select_non_romance_verbs = df_non_romance.nlargest(50, ['non_romance'])
export_selected_non_romance = select_non_romance_verbs.to_csv(f'output/highest_tf_idf_non_romance_verbs_x50.csv', header=True)

"""
First attempt of selecting 'romance' word resulted in 04 words LSA does not recognize:
For romance corpus:
Can't find any terms from text: 'contort' 
Can't find any terms from text: 'dishevel' 
Can't find any terms from text: 'enrol' 
Can't find any terms from text: 'appal' 
So, in second attempt, 54 highest were extract, and these 4 words were mannually deleted that LSA space.

"""
"""
Further on working with dataframe through Pandas:
https://www.geeksforgeeks.org/python-pandas-dataframe-where/
https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.html
https://datatofish.com/select-rows-pandas-dataframe/
This can also be achieved using NumPy:
https://stackoverflow.com/questions/41298073/how-to-get-the-most-representative-features-in-the-following-tfidf-model
"""